"""Reward-series failure cases, followed by native physics/PPO/export checks.

Risks: root centering hides travel; frame transforms change scores; windows leak
across partial resets; angle removal changes other terms; contact absence masks
penalties; ankle motion is mistaken for contact slip; native queries mutate
physics; reward contracts are lost on save/resume; launch placement/budget drift.
Run: .venv/bin/python -m pytest tests/test_reward_series.py --basetemp=artifacts/reward-series-tests
"""
from dataclasses import replace
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from k1_motion.spatial_rewards import SpatialTrackingReward, SERIES_PROFILES
from k1_motion.observations import quat_apply, quat_mul
from k1_motion.tracking_env import TrackerEnv, MujocoPhysics
from k1_motion.cpu_physics import CpuParallelPhysics
from test_training import make_library


def sample(batch=2):
    point = torch.arange(51, dtype=torch.float32).reshape(1, 17, 3).repeat(batch, 1, 1) / 100
    orientation = torch.tensor([[1., 0, 0, 0]]).repeat(batch, 1)
    state = dict(q=torch.zeros(batch, 22), position=torch.zeros(batch, 3),
                 orientation=orientation, velocity=torch.zeros(batch, 3), landmarks=point,
                 body_orientation=orientation[:, None].repeat(1, 17, 1))
    reference = {key: value.clone() for key, value in dict(joint_position=state['q'],
        root_position=state['position'], root_orientation=state['orientation'],
        root_velocity=torch.zeros(batch, 6), landmarks=point,
        body_orientation=state['body_orientation'], landmark_velocity=torch.zeros_like(point),
        contacts=torch.tensor([[1., 0.]]).repeat(batch, 1),
        contact_confidence=torch.tensor([[1., 0.]]).repeat(batch, 1)).items()}
    support = dict(contact=reference['contacts'].bool(), slip_speed=torch.zeros(batch, 2),
                   normal_force=reference['contacts'] * 10)
    return state, reference, support


@pytest.mark.parametrize('profile,total', list(zip(SERIES_PROFILES, (13.5, 15.5, 16.5, 15.0))))
def test_peak_world_translation_and_common_heading_invariance(profile, total):
    state, ref, support = sample()
    reward = SpatialTrackingReward(profile, 2, 'cpu', .02)
    assert sum(reward.contract['weights'].values()) == total
    reward.reset(torch.arange(2), state['position'], ref['root_position'])
    value, _ = reward.step(state, ref, ref['landmark_velocity'], support)
    torch.testing.assert_close(value, torch.full((2,), 9.5))
    state['position'][:, 0] += .4
    state['landmarks'][:, :, 0] += .4
    lagged, parts = reward.step(state, ref, ref['landmark_velocity'], support)
    assert (lagged < value - .5).all()
    torch.testing.assert_close(parts['body_position'], torch.ones(2))
    torch.testing.assert_close(parts['root_xy'], torch.full((2,), np.exp(-(.4/.25)**2)))
    yaw = torch.tensor([[np.sqrt(.5), 0, 0, np.sqrt(.5)]], dtype=torch.float32).repeat(2, 1)
    shift = torch.tensor([3., -4., 0.])
    for root_key, ori_key, obj in [('position', 'orientation', state),
                                 ('root_position', 'root_orientation', ref)]:
        obj[root_key] = quat_apply(yaw, obj[root_key]) + shift
        obj['landmarks'] = quat_apply(yaw[:, None].expand(2, 17, 4), obj['landmarks']) + shift
        obj[ori_key] = quat_mul(yaw, obj[ori_key])
        obj['body_orientation'] = quat_mul(yaw[:, None].expand(2, 17, 4), obj['body_orientation'])
    reward.reset(torch.arange(2), state['position'], ref['root_position'])
    rotated, _ = reward.step(state, ref, ref['landmark_velocity'], support)
    torch.testing.assert_close(rotated, lagged, atol=3e-6, rtol=0)


def test_progress_penalizes_wrong_speed_direction_and_resets_only_selected_world():
    state, ref, support = sample()
    reward = SpatialTrackingReward('spatial-s3-v1', 2, 'cpu', .02)
    reward.reset(torch.arange(2), state['position'], ref['root_position'])
    ref['root_velocity'][:, 0] = 1.
    for tick in range(1, 26):
        ref['root_position'][:, 0] = tick * .02
        _, parts = reward.step(state, ref, ref['landmark_velocity'], support)
        torch.testing.assert_close(parts['progress_enabled'], torch.full((2,), float(tick >= 25)))
    torch.testing.assert_close(parts['progress'], torch.full((2,), np.exp(-(.5/.15)**2)))
    assert (parts['velocity'] < .02).all()
    reward.reset(torch.tensor([0]), state['position'][:1], ref['root_position'][:1])
    _, parts = reward.step(state, ref, ref['landmark_velocity'], support)
    torch.testing.assert_close(parts['progress_enabled'], torch.tensor([0., 1.]))


def test_s5_removes_only_angle_term_and_unknown_support_cannot_change_denominator():
    state, ref, support = sample()
    state['q'][:] = .3
    values = {}
    for profile in ('spatial-s3-v1', 'spatial-s5-v1'):
        reward = SpatialTrackingReward(profile, 2, 'cpu', .02)
        reward.reset(torch.arange(2), state['position'], ref['root_position'])
        values[profile], _ = reward.step(state, ref, ref['landmark_velocity'], support)
    assert (values['spatial-s3-v1'] < 9.5).all()
    torch.testing.assert_close(values['spatial-s5-v1'], torch.full((2,), 9.5))
    state['q'].zero_()
    reward = SpatialTrackingReward('spatial-s4-v1', 2, 'cpu', .02)
    reward.reset(torch.arange(2), state['position'], ref['root_position'])
    support['contact'][0] = torch.tensor([False, True])
    support['slip_speed'][1, 0] = .2
    value, parts = reward.step(state, ref, ref['landmark_velocity'], support)
    torch.testing.assert_close(parts['support_score'], torch.tensor([0., (.5 + .5*np.exp(-4))], dtype=torch.float32))
    assert (value < 9.5).all()
    torch.testing.assert_close(parts['support_enabled'], torch.ones(2))
    ref['contact_confidence'].fill_(.5)
    masked, parts = reward.step(state, ref, ref['landmark_velocity'], support)
    torch.testing.assert_close(masked, torch.full((2,), 9.5))
    torch.testing.assert_close(parts['support_enabled'], torch.zeros(2))


def test_native_contact_patch_force_and_slip_match_scalar_without_mutation(tmp_path):
    spec, _ = make_library(tmp_path)
    scalar = MujocoPhysics(spec, 3, 'cpu')
    cpp = CpuParallelPhysics(spec, 3, 'cpu', workers=2)
    reference = spec.neutral_reference()
    fields = ('root_position', 'root_orientation', 'joint_position', 'root_velocity', 'joint_velocity')
    ref = {k: torch.tensor(np.tile(getattr(reference, k), (3, 1)), dtype=torch.float32) for k in fields}
    ref['root_position'][:, 2] -= .002
    ref['root_position'][1, 2] += 1.
    ref['root_velocity'][2, 0] = .4
    try:
        for physics in (scalar, cpp):
            physics.reset(torch.arange(3), ref)
        for tick in range(5):
            before = {k: v.clone() for k, v in cpp.state().items()}
            expected, actual = scalar.foot_support(), cpp.foot_support()
            for key in expected:
                torch.testing.assert_close(actual[key], expected[key], atol=1e-6, rtol=1e-6)
            for key in before:
                torch.testing.assert_close(cpp.state()[key], before[key], atol=0, rtol=0)
            assert not actual['contact'][1].any()
            if tick == 0:
                assert actual['contact'][0].all()
                torch.testing.assert_close(actual['slip_speed'][2], torch.full((2,), .4), atol=1e-5, rtol=0)
            for physics in (scalar, cpp):
                physics.step(ref['joint_position'])
    finally:
        scalar.close()
        cpp.close()


@pytest.mark.parametrize('profile', SERIES_PROFILES)
def test_series_cpp_ppo_checkpoint_export_and_changed_reward_resume_rejection(tmp_path, profile):
    from k1_motion.learning import TrainConfig, train, Policy
    from k1_motion.export import export_checkpoint
    from k1_motion.contracts import MotionClip
    from k1_motion.control_validation import replay_clip
    robot, library = make_library(tmp_path)
    torch.set_num_threads(1)
    config = TrainConfig(stage='student', iterations=2, horizon=32, epochs=1, minibatch=64,
                         hidden_sizes=(32, 16), evaluation_interval=0, checkpoint_interval=1, bc_weight=0)
    settings = dict(history=3, observation_profile='planar', reward_profile=profile,
                    self_collision_weight=1., physics_options={'workers': 2}, reference_storage='packed')
    env = TrackerEnv(library, 2, 'cpu', 'mujoco_cpp', **settings)
    try:
        result = train(env, tmp_path/'training', config)
        assert result['finite_updates'] and result['checkpoint_reload_max_error'] == 0
        assert result['reward_settings']['spatial_tracking']['version'] == profile
        assert result['transitions'] == 128 and result['optimizer_steps'] == 2
        export_checkpoint(tmp_path/'training/checkpoint.pt', tmp_path/'export/actor.pt')
        replay = replay_clip(robot, Policy(tmp_path/'export/actor.pt', robot.signature),
                             MotionClip.load(library/'standing.npz'))
        assert replay['resets_during_trial'] == 0
        changed_profile = 'spatial-s3-v1' if profile != 'spatial-s3-v1' else 'spatial-s5-v1'
        changed = TrackerEnv(library, 2, 'cpu', 'mujoco_cpp', **{**settings, 'reward_profile': changed_profile})
        try:
            with pytest.raises(ValueError, match='reward settings'):
                train(changed, tmp_path/'changed', replace(config, iterations=1),
                      resume_checkpoint=tmp_path/'training/checkpoint.pt')
        finally:
            changed.close()
    finally:
        env.close()


def test_launcher_exact_four_profiles_placement_ten_hour_budget_and_warp_capacity(tmp_path):
    path = Path(__file__).resolve().parents[1] / 'scripts/run_reward_series.py'
    spec = importlib.util.spec_from_file_location('reward_series_launcher', path)
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    assert launcher.NAMES == ('s2_positions', 's3_timing', 's4_support', 's5_no_angles_warp')
    for i, name in enumerate(launcher.NAMES, 2):
        command = launcher.learner_command(tmp_path, tmp_path/name, tmp_path/'cache.pt', name, 2048, False)
        def option(flag):
            return command[command.index(flag) + 1]
        assert option('--max-seconds') == '36000'
        assert option('--reward-profile') == f'spatial-s{i}-v1'
        assert option('--observation-profile') == 'planar'
        assert option('--backend') == ('warp' if i == 5 else 'mujoco_cpp')
        assert '--root-velocity-weight' not in command and '--tracking-huber' not in command
        if i == 5:
            assert option('--epa-horizon') == '96'
        else:
            assert '--epa-horizon' not in command
