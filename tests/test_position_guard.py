"""Command guard failures and end-to-end physical/runtime parity.

Failure cases fixed before implementation: invalid or collapsed ranges, jumps
when enabling an interior envelope, asymmetric inward/outward feedforward,
zero-margin behavior drift, arm feedback escaping the envelope, runtime versus
training divergence, and diagnostics hidden by clipping physical state.
"""
from dataclasses import replace

import mujoco
import numpy as np
import pytest
import torch

from k1_motion.actuation import action_settings
from k1_motion.contracts import MotionClip
from k1_motion.robot import K1Model
from k1_motion.runtime import Controller
from k1_motion.servo import step_command
from k1_motion.tracking_env import TrackerEnv
from test_training import make_library


def test_position_margin_validation():
    robot = K1Model()
    assert 'command_position_margin_rad' not in action_settings(robot)
    assert action_settings(robot, {'command_position_margin_rad': 0})['command_position_margin_rad'] == 0
    assert action_settings(robot, {'command_position_margin_rad': [.02]*22})['command_position_margin_rad'] == [.02]*22
    for invalid in (-.1, float('nan'), [0]*21, [float('inf')]*22,
                    (robot.limits[:, 1]-robot.limits[:, 0])/2):
        with pytest.raises(ValueError, match='command_position_margin_rad'):
            action_settings(robot, {'command_position_margin_rad': invalid})


def test_position_guard_zero_and_vector_zero_are_exact_noops():
    from k1_motion.actuation import position_guard_tensor
    target = torch.arange(44, dtype=torch.float32).reshape(2, 22)/100
    velocity = target-0.1
    limits = torch.tensor([[-1., 1.]]*22)
    for margin in (torch.zeros(22), torch.tensor(0.)):
        actual = position_guard_tensor(target, velocity, target, target, limits, .01, margin)
        assert torch.equal(actual[0], target)
        assert torch.equal(actual[1], velocity)
    margin = torch.zeros(22)
    margin[0] = .1
    q, v = position_guard_tensor(target, velocity, target, target, limits, .01, margin)
    assert torch.equal(q[:, 1:], target[:, 1:])
    assert torch.equal(v[:, 1:], velocity[:, 1:])


def test_position_guard_keeps_slew_while_entering_interior_envelope():
    from k1_motion.actuation import position_guard_tensor
    limits = torch.tensor([[-1., 1.]]*22)
    previous = torch.tensor([[1.]*22, [-1.]*22])
    q = previous.clone()
    for step in range(5):
        target, velocity = position_guard_tensor(q, torch.zeros_like(q), previous, q,
                                                limits, .02, torch.tensor(.05))
        assert torch.max(torch.abs(target-previous)) <= .020001
        assert torch.all(target.abs() <= previous.abs())
        previous = target
    torch.testing.assert_close(target.abs(), torch.full_like(target, .95))
    assert torch.equal(q, torch.tensor([[1.]*22, [-1.]*22]))


def test_position_guard_tapers_outward_velocity_without_blocking_recovery():
    from k1_motion.actuation import position_guard_tensor
    limits = torch.tensor([[-1., 1.]]*22)
    # At the upper safe boundary, halfway through its warning band, and well inside.
    q = torch.tensor([[.9]*22, [.85]*22, [.5]*22, [1.01]*22])
    target = torch.full_like(q, .5)
    for sign in (1., -1.):
        args = (target*sign, torch.ones_like(q)*sign, target*sign, q*sign, limits, 1., torch.tensor(.1))
        guarded, outward = position_guard_tensor(*args)
        torch.testing.assert_close(outward[:, 0], torch.tensor([0., .5, 1., 0.])*sign)
        _, inward = position_guard_tensor(target*sign, -torch.ones_like(q)*sign,
            target*sign, q*sign, limits, 1., torch.tensor(.1))
        torch.testing.assert_close(inward, -torch.ones_like(q)*sign)
        assert torch.all(guarded.abs() <= .9)
    # Feedforward cannot push against an already clipped desired target either.
    _, velocity = position_guard_tensor(torch.ones_like(q), torch.ones_like(q),
        q, torch.zeros_like(q), limits, 1., torch.tensor(.1))
    assert torch.count_nonzero(velocity) == 0


def test_partial_guard_preserves_unenabled_runtime_double_precision_targets():
    robot = K1Model()
    class Policy:
        def __init__(self, margin):
            self.metadata = {'action_settings': {'command_position_margin_rad': margin}}
        def __call__(self, observation):
            return np.ones(22, dtype=np.float32)
    reference = robot.neutral_reference()
    commands = []
    for margin in (0., [.04]+[0.]*21):
        controller = Controller(robot, Policy(margin))
        controller.calibrate(reference, robot.state(0), True, 0)
        controller.arm(robot.state(0), 0)
        controller.previous_target[10] += .0123456789123
        commands.append(controller.tick(robot.state(0), 0))
    np.testing.assert_array_equal(commands[0].targets[1:], commands[1].targets[1:])
    np.testing.assert_array_equal(commands[0].velocities[1:], commands[1].velocities[1:])


@pytest.mark.parametrize('backend,profile', [('mujoco', 'causal'), ('mujoco_cpp', 'causal'),
    ('mujoco_cpp', 'preview'), pytest.param('warp', 'preview', marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason='NVIDIA CUDA required for guarded Warp replay'))])
def test_guarded_controller_training_replay_parity_and_substep_safety(tmp_path, backend, profile):
    torch.set_num_threads(1)
    robot, directory = make_library(tmp_path)
    path = directory/'standing.npz'
    clip = MotionClip.load(path)
    refs = []
    for i in range(50):
        ref = robot.neutral_reference(i*.02)
        q, dq = ref.joint_position.copy(), np.zeros(22)
        q[1] = robot.limits[1, 1]-.003
        dq[1] = .8
        if backend == 'warp':
            # Isolate actuator/command parity from contact-solver differences.
            ref = replace(ref, root_position=ref.root_position+np.array([0., 0., 2.]))
        refs.append(replace(ref, joint_position=q, joint_velocity=dq))
    MotionClip.from_references(refs, clip.metadata).save(path)
    settings = {'command_position_margin_rad': [.04]*10+[.02]*12,
                'target_velocity_scale': .5, 'upper_body_residual_scale': 0.,
                'arm_collision_clearance': .025}
    from k1_motion.actuators import ACTUATOR_PROFILE, actuator_contract
    from k1_motion.observations import observation_contract
    settings['actuator_profile'] = ACTUATOR_PROFILE

    class Policy:
        metadata = {'action_settings': settings, 'actuator_contract': actuator_contract(robot, settings),
                    'observation': observation_contract(4, profile)}
        seen = []

        def __call__(self, observation):
            self.seen.append(observation.copy())
            return np.zeros(22, dtype=np.float32)

    device = 'cuda:0' if backend == 'warp' else 'cpu'
    env = TrackerEnv(directory, num_envs=1, device=device, backend=backend,
        action_settings=settings, observation_profile=profile,
        **({'physics_options': {'workers': 1}} if backend=='mujoco_cpp' else
           {'physics_options': {'epa_horizon': 96}} if backend=='warp' else {}))
    zero = torch.zeros(1, dtype=torch.long, device=device)
    observed, _ = env.reset(clips=zero, frames=zero)
    robot.reset(refs[0])
    robot.data.qvel[6:] = refs[0].joint_velocity
    mujoco.mj_forward(robot.model, robot.data)
    policy = Policy()
    controller = Controller(robot, policy)
    controller.calibrate(refs[0], robot.state(0), True, 0)
    delay = .3 if profile=='preview' else 0.
    if delay:
        for ref in refs[1:16]:
            controller.set_reference(ref)
    controller.arm(robot.state(delay), delay)
    captured = []
    original_step = env.physics.step
    def capture_step(targets, velocities):
        captured.append((targets.detach().cpu().numpy().copy(), velocities.detach().cpu().numpy().copy()))
        return original_step(targets, velocities)
    env.physics.step = capture_step
    try:
        for i in range(12):
            now = delay+i*.02
            if i:
                controller.set_reference(refs[i+15 if delay else i])
            command = controller.tick(robot.state(now), now)
            assert command.targets[1] <= robot.limits[1, 1]-.04+1e-6
            assert command.velocities[1] == 0
            if profile=='preview':
                np.testing.assert_allclose(policy.seen[-1][-120:], observed[0, -120:].cpu(), atol=2e-5)
                assert np.all(policy.seen[-1][-120:].reshape(3, 40)[:, -1] == 1)
            observed, _, _, _, _ = env.step(torch.zeros((1, 22), device=device), auto_reset=False)
            diagnostics = step_command(robot, command, controller.action_settings)
            state = env.physics.state()
            np.testing.assert_allclose(state['q'][0].cpu().numpy(), robot.data.qpos[7:], atol=2e-6)
            np.testing.assert_allclose(state['dq'][0].cpu().numpy(), robot.data.qvel[6:], atol=2e-5)
            np.testing.assert_allclose(env.previous_target[0].cpu().numpy(), command.targets, atol=2e-7)
            np.testing.assert_allclose(captured[-1][0][0], command.targets, atol=2e-7)
            np.testing.assert_allclose(captured[-1][1][0], command.velocities, atol=2e-6)
            for key, value in diagnostics['safety'].items():
                assert float(env.physics.safety_per_env[key][0]) == pytest.approx(value, abs=5e-5)
        # The target envelope is not an artificial clamp on actual position.
        assert robot.data.qpos[1+7] != command.targets[1]
    finally:
        env.close()


def test_guard_does_not_hide_actual_overspeed_or_range_diagnostics():
    from k1_motion.servo import step_pd
    robot = K1Model()
    robot.substeps = 1
    robot.data.qpos[2] += 2.
    robot.data.qpos[8] = robot.limits[1, 1]+.01
    robot.data.qvel[7] = 1.
    mujoco.mj_forward(robot.model, robot.data)
    diagnostics = step_pd(robot, robot.neutral, np.zeros(22), settings={
        'command_position_margin_rad': .04, 'command_velocity_limit': .001})
    assert robot.data.qpos[8] > robot.limits[1, 1]
    assert diagnostics['safety']['joint_limit_fraction'] > 0
    assert diagnostics['safety']['operating_speed_max_ratio'] > 1
