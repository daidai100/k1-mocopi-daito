"""Failure cases: lag hidden, poor recovery sensitivity, implicit weight dilution,
changed-reward resume, invalid scale/rotation, and lost export/runtime contract.
End-to-end artifact: pytest temporary training/checkpoint.pt and export/actor.pt.
Reproduce: .venv/bin/python -m pytest tests/test_simple_rewards.py
"""
from dataclasses import replace

import numpy as np
import pytest
import torch

from k1_motion.simple_rewards import SCREEN_PROFILES, SimpleTrackingReward
from k1_motion.spatial_rewards import SpatialTrackingReward
from test_reward_series import sample
from test_training import make_library


def test_long_tail_penalizes_world_lag_without_diluting_other_terms():
    state, ref, _ = sample()
    reward = SimpleTrackingReward('simple-track-v1', 2, 'cpu', .02)
    perfect, before = reward.step(state, ref, ref['landmark_velocity'])
    torch.testing.assert_close(perfect, torch.full((2,), 9.5))
    state['position'][:, 0] += 1.
    state['landmarks'][:, :, 0] += 1.
    lagged, after = reward.step(state, ref, ref['landmark_velocity'])
    torch.testing.assert_close(after['anchor'], torch.full((2,), .2))
    torch.testing.assert_close(lagged, perfect - 1.6)
    for key in ('velocity', 'body_position', 'orientation', 'joint', 'body_velocity'):
        torch.testing.assert_close(before['weighted/'+key], after['weighted/'+key])
    assert reward.contract['weights']['velocity'] == 4.
    state['position'][:, 0] -= .5
    state['landmarks'][:, :, 0] -= .5
    improving, _ = reward.step(state, ref, ref['landmark_velocity'])
    assert (improving > lagged + .5).all()
    state['position'][:, 2] += .08
    state['landmarks'][:, :, 2] += .08
    lowered, _ = reward.step(state, ref, ref['landmark_velocity'])
    assert (lowered < improving).all()


def test_no_angle_ablation_preserves_other_coefficients_and_frame_invariance():
    state, ref, _ = sample()
    base = SimpleTrackingReward('simple-track-v1', 2, 'cpu', .02)
    ablated = SimpleTrackingReward('simple-track-no-joint-v1', 2, 'cpu', .02)
    first, parts = base.step(state, ref, ref['landmark_velocity'])
    second, removed = ablated.step(state, ref, ref['landmark_velocity'])
    torch.testing.assert_close(first-second, torch.full((2,), .5))
    for name in base.contract['weights']:
        if name != 'joint':
            torch.testing.assert_close(parts['weighted/'+name], removed['weighted/'+name])
    # Common rigid translation never changes scores.
    for obj, key in [(state, 'position'), (ref, 'root_position')]:
        obj[key] += torch.tensor([3., -5., 0.])
        obj['landmarks'] += torch.tensor([3., -5., 0.])
    translated, _ = base.step(state, ref, ref['landmark_velocity'])
    torch.testing.assert_close(translated, first)


def test_s3_tail_changes_only_xy_kernel_with_same_history_and_masks():
    state, ref, _ = sample()
    old = SpatialTrackingReward('spatial-s3-v1', 2, 'cpu', .02)
    new = SimpleTrackingReward('screen-s3-tail-v1', 2, 'cpu', .02)
    for obj in (old, new):
        obj.reset(torch.arange(2), state['position'], ref['root_position'])
    state['position'][:, 0] += 1.
    state['landmarks'][:, :, 0] += 1.
    for _ in range(26):
        _, previous = old.step(state, ref, ref['landmark_velocity'])
        _, changed = new.step(state, ref, ref['landmark_velocity'])
    torch.testing.assert_close(changed['root_xy'], torch.full((2,), 1/17))
    for name in old.weights:
        if name != 'root_xy':
            torch.testing.assert_close(previous['weighted/'+name], changed['weighted/'+name])


def test_v2_height_signal_is_independent_of_large_xy_lag():
    reward = SimpleTrackingReward('simple-track-v2', 2, 'cpu', .02)
    penalties = []
    for lag in (0., 2.):
        state, ref, _ = sample()
        state['position'][:, 0] += lag
        state['landmarks'][:, :, 0] += lag
        before, _ = reward.step(state, ref, ref['landmark_velocity'])
        state['position'][:, 2] += .08
        state['landmarks'][:, :, 2] += .08
        after, parts = reward.step(state, ref, ref['landmark_velocity'])
        penalties.append(before-after)
        torch.testing.assert_close(parts['height'], torch.full((2,), float(np.exp(-1))))
    torch.testing.assert_close(penalties[0], penalties[1])
    assert (penalties[1] > .6).all()
    assert sum(reward.contract['weights'].values()) == 9.5


@pytest.mark.parametrize('profile', SCREEN_PROFILES)
def test_reward_ppo_export_and_strict_resume_contract(tmp_path, profile):
    from k1_motion.tracking_env import TrackerEnv
    from k1_motion.learning import TrainConfig, train, Policy
    from k1_motion.export import export_checkpoint
    from k1_motion.contracts import MotionClip
    from k1_motion.control_validation import replay_clip
    robot, library = make_library(tmp_path)
    torch.set_num_threads(1)
    config = TrainConfig(stage='student', iterations=2, horizon=8, epochs=1, minibatch=16,
                         hidden_sizes=(32, 16), evaluation_interval=0, checkpoint_interval=1, bc_weight=0)
    settings = dict(history=3, observation_profile='planar', reward_profile=profile,
                    self_collision_weight=1., physics_options={'workers': 2}, reference_storage='packed')
    env = TrackerEnv(library, 2, 'cpu', 'mujoco_cpp', **settings)
    try:
        result = train(env, tmp_path/'training', config)
        assert result['finite_updates'] and result['checkpoint_reload_max_error'] == 0
        assert result['reward_settings']['spatial_tracking']['version'] == profile
        assert result['transitions'] == 32
        export_checkpoint(tmp_path/'training/checkpoint.pt', tmp_path/'export/actor.pt')
        replay = replay_clip(robot, Policy(tmp_path/'export/actor.pt', robot.signature),
                             MotionClip.load(library/'standing.npz'))
        assert replay['resets_during_trial'] == 0
        assert np.isfinite(replay['root_velocity_rmse_m_s'])
        different = 'simple-track-v1' if profile != 'simple-track-v1' else 'simple-track-wide-v1'
        other = TrackerEnv(library, 2, 'cpu', 'mujoco_cpp', **{**settings, 'reward_profile': different})
        try:
            with pytest.raises(ValueError, match='reward settings'):
                train(other, tmp_path/'invalid_resume', replace(config, iterations=1),
                      resume_checkpoint=tmp_path/'training/checkpoint.pt')
        finally:
            other.close()
    finally:
        env.close()
