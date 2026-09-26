"""Measured-speed motor guard: directional boundaries, real integration and parity.

Failure contract written first: never suppress braking or fabricate a safe state;
never exceed the manufacturer envelope; disabled behavior stays bit-identical;
scalar/native/Warp agree and exported effective actuation identifies the guard.
"""
import mujoco
import numpy as np
import pytest
import torch

from k1_motion.actuation import action_settings
from k1_motion.actuators import ACTUATOR_PROFILE, SAFETY_FIELDS, actuator_contract
from k1_motion.cpu_physics import CpuParallelPhysics
from k1_motion.robot import K1Model
from k1_motion.runtime import Controller
from k1_motion.servo import step_pd
from k1_motion.tracking_env import MujocoPhysics
from test_actuator_contract import reset_reference, settings


def guarded_settings(robot):
    return {**settings(robot), 'operating_speed_guard': True}


def test_setting_requires_explicit_boolean_and_contract_preserves_legacy():
    robot = K1Model()
    original = settings(robot)
    assert 'operating_speed_guard' not in action_settings(robot, original)
    assert actuator_contract(robot, original) == actuator_contract(
        robot, {**original, 'operating_speed_guard': False})
    for invalid in (0, 1, .1, 'true', [], None):
        with pytest.raises(ValueError, match='operating_speed_guard'):
            action_settings(robot, {**original, 'operating_speed_guard': invalid})
    contract = actuator_contract(robot, guarded_settings(robot))
    assert contract['operating_speed_guard']['version'] == 'accelerating-torque-taper-v1'
    assert contract['operating_speed_guard']['onset_fraction'] == .9
    assert contract['operating_speed_guard']['state_clipping'] is False
    assert contract['profile'] == ACTUATOR_PROFILE
    class Policy:
        metadata = dict(action_settings=guarded_settings(robot), actuator_contract=contract)
    Controller(robot, Policy())
    Policy.metadata = {**Policy.metadata, 'actuator_contract': actuator_contract(robot, original)}
    with pytest.raises(ValueError, match='actuator contract'):
        Controller(robot, Policy())


def test_guard_signed_boundaries_preserve_braking_and_zero_speed():
    from k1_motion.actuators import speed_guard_torque
    speed = np.array([0., 8., 9., 9.5, 10., 11., -8., -9., -9.5, -10., -11.])
    accelerating = np.where(speed < 0, -20., 20.)
    expected = np.array([20., 20., 20., 10., 0., 0., -20., -20., -10., 0., 0.])
    actual = speed_guard_torque(accelerating, speed, 10.)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-13)
    np.testing.assert_array_equal(speed_guard_torque(-accelerating, speed, 10.), -accelerating)
    np.testing.assert_array_equal(speed_guard_torque(np.zeros_like(speed), speed, 10.), 0.)
    assert np.all(abs(actual) <= abs(accelerating))
    assert np.all(actual*accelerating >= 0)


def test_disabled_servo_is_exact_and_active_guard_preserves_measured_violations():
    a, b = K1Model(), K1Model()
    for robot in (a, b):
        robot.substeps = 1
        robot.data.qpos[2] += 2.
        robot.data.qvel[6+13] = 11.
        mujoco.mj_forward(robot.model, robot.data)
    target = a.neutral.copy()
    target[13] = 2.
    old = step_pd(a, target, np.zeros(22), settings=settings(a))
    disabled = step_pd(b, target, np.zeros(22), settings={**settings(b), 'operating_speed_guard': False})
    np.testing.assert_array_equal(a.data.qpos, b.data.qpos)
    np.testing.assert_array_equal(a.data.qvel, b.data.qvel)
    assert old == disabled
    c = K1Model()
    c.substeps = 1
    c.data.qpos[2] += 2.
    c.data.qpos[7+13] = 1.
    c.data.qvel[6+13] = 11.
    mujoco.mj_forward(c.model, c.data)
    result = step_pd(c, target, np.zeros(22), settings=guarded_settings(c))
    assert c.data.ctrl[13] == 0.  # Requested torque would accelerate overspeed.
    assert c.data.qvel[6+13] > .8*c.velocity_limit[13]
    assert result['safety']['operating_speed_fraction'] > 0
    assert result['safety']['operating_speed_max_ratio'] > 1


def test_guard_real_knee_acceleration_canary(tmp_path):
    traces = {}
    for enabled in (False, True):
        robot = K1Model()
        robot.substeps = 1
        robot.data.qpos[2] += 2.
        robot.data.qpos[7+13] = 1.
        operating = .8*robot.velocity_limit[13]
        robot.data.qvel[6+13] = .97*operating
        mujoco.mj_forward(robot.model, robot.data)
        target = robot.neutral.copy()
        target[13] = 2.1
        rows = []
        for _ in range(15):
            before = robot.data.qvel[6+13]
            step_pd(robot, target, np.zeros(22), settings={**settings(robot), 'operating_speed_guard': enabled})
            rows.append([before, robot.data.qvel[6+13], robot.data.ctrl[13]])
        traces[str(enabled)] = np.array(rows)
    np.savez_compressed(tmp_path/'knee-speed-guard-canary.npz', **traces)
    assert traces['False'][:, 1].max() > operating
    assert traces['True'][:, 1].max() < operating
    assert traces['True'][0, 2] < traces['False'][0, 2]


def test_native_scalar_active_guard_substep_and_reset_parity():
    torch.set_num_threads(1)
    robot = K1Model()
    robot.substeps = 1
    robot.control_dt = .002
    config = guarded_settings(robot)
    scalar = MujocoPhysics(robot, 4, 'cpu', config)
    native = CpuParallelPhysics(robot, 4, 'cpu', config, workers=2)
    reference = reset_reference(robot, 4)
    reference['joint_position'][:, 13] = 1.
    reference['joint_velocity'][:, 13] = torch.tensor([.95, -.95, 1.1, -1.1])*(.8*robot.velocity_limit[13])
    try:
        for physics in (scalar, native):
            physics.reset(torch.arange(4), reference)
        for tick in range(8):
            target = reference['joint_position'].clone()
            target[:, 13] = torch.tensor([2., 0., 0., 2.]) if tick < 4 else torch.tensor([0., 2., 2., 0.])
            for physics in (scalar, native):
                physics.step(target)
            for name, value in scalar.state().items():
                torch.testing.assert_close(native.state()[name], value, rtol=0, atol=0)
            for name in SAFETY_FIELDS:
                torch.testing.assert_close(native.safety_per_env[name], scalar.safety_per_env[name],
                                           rtol=2e-6, atol=2e-7)
        for physics in (scalar, native):
            physics.reset(torch.tensor([1]), {name:value[1:2] for name,value in reference.items()})
            assert all(value[1] == 0 for value in physics.safety_per_env.values())
        assert scalar.contract['actuator'] == native.contract['actuator']
    finally:
        scalar.close()
        native.close()


def test_warp_guard_kernel_signed_torque_matches_scalar_and_preserves_envelope():
    import warp as wp
    from k1_motion.actuators import speed_guard_torque, torque_envelope, manufacturer_parameters
    from k1_motion.warp_physics import _speed_guard
    wp.init()
    params = manufacturer_parameters()
    operating = (.8*params['velocity']).astype(np.float32)
    speed = np.array([0., .89, .9, .95, 1., 1.1, -.89, -.95, -1., -1.1],np.float32)[:,None]*operating
    qvel = np.zeros((len(speed),28),np.float32)
    qvel[:,6:] = speed
    available = torque_envelope(speed, params['effort'], params['velocity'], params['knee']).astype(np.float32)
    torque = available.copy()
    torque[::2] *= -1
    power = (torque/params['effort'].astype(np.float32))**2
    def array(value):
        return wp.array(np.asarray(value,np.float32), dtype=float, device='cpu')
    ctrl, energy = array(torque), array(power)
    wp.launch(_speed_guard, dim=speed.shape, device='cpu', inputs=[
        array(qvel), array(operating), array(params['effort']), 1., ctrl, energy])
    expected = speed_guard_torque(torque, speed, operating)
    np.testing.assert_allclose(ctrl.numpy(), expected, rtol=2e-6, atol=2e-5)
    np.testing.assert_allclose(energy.numpy(), (expected/params['effort'])**2, rtol=2e-6, atol=2e-7)
    assert np.all(abs(ctrl.numpy()) <= available+1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='NVIDIA CUDA required for real Warp parity')
def test_warp_native_active_guard_physical_parity():
    from k1_motion.warp_physics import WarpPhysics
    torch.set_num_threads(1)
    robot = K1Model()
    robot.substeps = 1
    robot.control_dt = .002
    config = guarded_settings(robot)
    native = CpuParallelPhysics(robot, 2, 'cpu', config, workers=2)
    warp = WarpPhysics(robot, 2, 'cuda:0', action_settings=config, epa_horizon=96)
    reference = reset_reference(robot)
    reference['joint_position'][:,13] = 1.
    reference['joint_velocity'][:,13] = torch.tensor([.95, -.95])*(.8*robot.velocity_limit[13])
    try:
        native.reset(torch.arange(2), reference)
        warp.reset(torch.arange(2, device='cuda:0'), {k:v.cuda() for k,v in reference.items()})
        for _ in range(5):
            target = reference['joint_position'].clone()
            target[:,13] = torch.tensor([2., 0.])
            native.step(target)
            warp.step(target.cuda())
            for name in ('q', 'dq', 'position', 'velocity'):
                torch.testing.assert_close(warp.state()[name].cpu(), native.state()[name], rtol=3e-4, atol=2e-5)
            for name in SAFETY_FIELDS:
                torch.testing.assert_close(warp.safety_per_env[name].cpu(), native.safety_per_env[name],
                                           rtol=5e-4, atol=5e-4)
    finally:
        native.close()
        warp.close()
