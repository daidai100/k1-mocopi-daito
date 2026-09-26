"""Official actuator math, opt-in dynamics and measured-substep safety contracts."""
import copy
import json

import mujoco
import numpy as np
import pytest
import torch

from k1_motion.actuation import action_settings, command_velocity_limits
from k1_motion.actuators import (
    ACTUATOR_PROFILE, SAFETY_FIELDS, actuator_parameters, configure_robot_actuators,
    manufacturer_parameters, torque_envelope,
)
from k1_motion.cpu_physics import CpuParallelPhysics
from k1_motion.robot import K1Model, ROOT
from k1_motion.servo import step_pd
from k1_motion.tracking_env import MujocoPhysics


def settings(robot, limit=None):
    return {"actuator_profile": ACTUATOR_PROFILE,
            "command_velocity_limit": (.8 * robot.velocity_limit).tolist() if limit is None else limit}


def reset_reference(robot, count=2):
    reference = robot.neutral_reference()
    keys = ("root_position", "root_orientation", "joint_position", "root_velocity", "joint_velocity")
    result = {key: torch.tensor(np.tile(getattr(reference, key), (count, 1)), dtype=torch.float32)
              for key in keys}
    result['root_position'][:, 2] += 2.0  # Compare actuation without contact-solver variation.
    return result


def test_official_joint_map_and_ankle_wrapper():
    values = manufacturer_parameters()
    np.testing.assert_array_equal(values['effort'], [6]*2+[14]*8+[68,76,38.3,112,38.3,38.3]*2)
    np.testing.assert_array_equal(values['velocity'], K1Model().velocity_limit)
    np.testing.assert_array_equal(values['knee'], [7.85]*2+[5.24]*8+[1.88,2.62,7.85,2.09,7.85,7.85]*2)
    assert values['armature'][0] == .001
    np.testing.assert_array_equal(values['armature'][[14,15,20,21]], [.0282528*2]*4)


def test_torque_envelope_matches_symmetric_official_breakpoints_and_zero_braking():
    velocity = np.array([-21.,-20.,-15.,-10.,0.,10.,15.,20.,21.])
    np.testing.assert_array_equal(torque_envelope(velocity, 80., 20., 10.), [0,0,40,80,80,80,40,0,0])
    # Manufacturer clamps a knee beyond nominal speed to nominal, then uses 1e-6 denominator.
    np.testing.assert_array_equal(torque_envelope(np.array([0.,7.,7.85,8.]), 6., 7.85, 10.47), [6,6,0,0])


def test_explicit_command_vector_keeps_legacy_default_and_nominal_envelope_separate():
    robot = K1Model()
    np.testing.assert_array_equal(command_velocity_limits(robot, action_settings(robot)), [6.]*22)
    values = action_settings(robot, settings(robot))
    np.testing.assert_array_equal(command_velocity_limits(robot, values), .8*robot.velocity_limit)
    params = actuator_parameters(robot, values)
    np.testing.assert_array_equal(params['velocity'], robot.velocity_limit)
    path = ROOT/'configs/controller-pv-official80-v1.json'
    config = action_settings(robot, json.loads(path.read_text()))
    np.testing.assert_array_equal(command_velocity_limits(robot, config), .8*robot.velocity_limit)
    assert config['actuator_profile'] == ACTUATOR_PROFILE
    for value in ([1]*21, [float('nan')]*22, [0]*22, 'invalid'):
        with pytest.raises(ValueError):
            action_settings(robot, {'command_velocity_limit': value})
    with pytest.raises(ValueError):
        action_settings(robot, {'actuator_profile': 'unversioned'})


def test_profile_copies_effective_model_without_mutating_reference_signature_or_geometry():
    original = K1Model()
    robot = copy.copy(original)
    robot.data = mujoco.MjData(original.model)
    mujoco.mj_copyData(robot.data, original.model, original.data)
    configure_robot_actuators(robot, settings(robot))
    assert robot.model is not original.model
    assert robot.signature == original.signature
    assert original.effort[11] == 43
    assert original.model.actuator_forcerange[11,1] == 43
    assert robot.effort[11] == 76
    assert robot.model.actuator_forcerange[11,1] == 76
    np.testing.assert_array_equal(robot.model.dof_armature[6:], manufacturer_parameters()['armature'])
    np.testing.assert_array_equal(robot.model.body_mass, original.model.body_mass)
    np.testing.assert_array_equal(robot.model.body_inertia, original.model.body_inertia)
    np.testing.assert_array_equal(robot.data.qpos, original.data.qpos)
    rebuilt = copy.copy(robot.model)
    mujoco.mj_setConst(rebuilt, mujoco.MjData(rebuilt))
    for key in ('dof_M0', 'dof_invweight0', 'body_invweight0'):
        np.testing.assert_array_equal(getattr(robot.model,key), getattr(rebuilt,key))
    # A repeated servo call must not recreate dynamics or erase warm-start data.
    model, data = robot.model, robot.data
    configure_robot_actuators(robot, settings(robot))
    assert robot.model is model and robot.data is data


def test_actual_hip_roll_torque_can_exceed_legacy43_but_not_official76():
    legacy, official = K1Model(), K1Model()
    for robot in (legacy, official):
        robot.substeps = 1
    command = legacy.neutral.copy()
    command[11] += 1.
    step_pd(legacy, command, np.zeros(22))
    diagnostics = step_pd(official, command, np.zeros(22), settings=settings(official))
    assert legacy.data.ctrl[11] == 43.
    assert 43 < official.data.ctrl[11] <= 76
    assert 43 < official.data.qfrc_actuator[17] <= 76
    assert set(diagnostics['safety']) == set(SAFETY_FIELDS)


def test_scalar_and_native_match_actuation_safety_and_partial_reset():
    torch.set_num_threads(1)
    robot = K1Model()
    config = settings(robot)
    native = MujocoPhysics(robot, 2, 'cpu', config)
    cpp = CpuParallelPhysics(robot, 2, 'cpu', config, workers=2)
    reference = reset_reference(robot)
    reference['joint_velocity'][0,10] = 13.
    reference['joint_velocity'][1,13] = -13.
    try:
        for physics in (native, cpp):
            physics.reset(torch.arange(2), reference)
        for tick in range(5):
            command = reference['joint_position']+.3*torch.sin(torch.arange(44).reshape(2,22)+tick)
            for physics in (native, cpp):
                physics.step(command, torch.full((2,22), 100.))
            for key, value in native.state().items():
                torch.testing.assert_close(cpp.state()[key], value, rtol=0, atol=0)
            for key in SAFETY_FIELDS:
                torch.testing.assert_close(cpp.safety_per_env[key], native.safety_per_env[key], rtol=2e-6, atol=2e-7)
            assert native.contract['actuator']['profile'] == ACTUATOR_PROFILE
            assert cpp.contract['actuator']['profile'] == ACTUATOR_PROFILE
        for physics in (native, cpp):
            before = {key: value.clone() for key,value in physics.safety_per_env.items()}
            physics.reset(torch.tensor([1]), {key:value[1:] for key,value in reference.items()})
            for key in SAFETY_FIELDS:
                assert physics.safety_per_env[key][1] == 0
                assert physics.safety_per_env[key][0] == before[key][0]
    finally:
        native.close()
        cpp.close()


@pytest.mark.parametrize('backend', ['scalar', 'native'])
def test_safety_samples_final_physics_state_without_clipping_actual_velocity(backend):
    robot = K1Model()
    robot.substeps = 1
    robot.control_dt = .002
    config = settings(robot, .001)
    cls = MujocoPhysics if backend == 'scalar' else CpuParallelPhysics
    kwargs = {} if backend == 'scalar' else {'workers': 1}
    physics = cls(robot, 1, 'cpu', config, **kwargs)
    reference = reset_reference(robot, 1)
    try:
        physics.reset(torch.tensor([0]), reference)
        physics.step(reference['joint_position']+.1)
        speed = physics.state()['dq'].abs()
        assert speed.max() > .001  # A command cap must never become a qvel clamp.
        expected_fraction = (speed > .001).float().mean(-1)
        torch.testing.assert_close(physics.safety_per_env['operating_speed_fraction'], expected_fraction)
        torch.testing.assert_close(physics.safety_per_env['operating_speed_max_ratio'], (speed/.001).max(-1).values,
                                   rtol=2e-6, atol=2e-6)
    finally:
        physics.close()
