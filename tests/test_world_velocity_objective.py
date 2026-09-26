"""Isolated root velocity shaping, then native PPO/export/replay contract.

Failure modes: adding velocity silently reweights world tracking; distant XY
lag extinguishes the velocity signal; only XY velocity is used; extreme finite
errors yield invalid rewards; the default legacy profile changes; checkpoints
lose the reward identity or resume an optimizer across different objectives.
"""
from dataclasses import replace

import numpy as np
import pytest
import torch

from k1_motion.world_objective import WorldBodyTracking
from test_reward_series import sample
from test_training import make_library


def test_velocity_addition_is_exact_with_no_reweighting_or_legacy_drift():
    state, reference, _ = sample()
    legacy = WorldBodyTracking()
    before_contract = dict(legacy.contract)
    state['landmarks'][:, :, 0] += .2
    state['q'][:, 3] += .1
    state['velocity'] = torch.tensor([[.3, .4, 0.], [0., 0., .5]])
    reference['root_velocity'].zero_()
    old, old_parts = legacy.step(state, reference, reference['landmark_velocity'])
    candidate = WorldBodyTracking('world-velocity-v1')
    new, new_parts = candidate.step(state, reference, reference['landmark_velocity'])
    expected = 2 * torch.exp(-torch.ones(2))
    torch.testing.assert_close(new-old, expected)
    torch.testing.assert_close(new_parts['weighted/root_velocity'], expected)
    for key in old_parts:
        if key != 'tracking_per_second':
            torch.testing.assert_close(new_parts[key], old_parts[key], rtol=0, atol=0)
    assert legacy.contract == before_contract
    assert legacy.contract['weights'] == dict(world_position=9., joint_posture=.25, root_orientation=.25)
    assert legacy.contract['maximum_tracking_per_second'] == 9.5
    assert candidate.contract['maximum_tracking_per_second'] == 11.5
    assert candidate.contract['normalization'] == legacy.contract['normalization']
    assert candidate.contract['scales']['root_velocity_m_s'] == .5
    assert candidate.contract['weights']['root_velocity'] == 2.


@pytest.mark.parametrize('lag_m', [0., .15, 1., 10.])
def test_velocity_signal_survives_near_and_far_world_position_error(lag_m):
    state, reference, _ = sample()
    state['landmarks'][:, :, 0] += lag_m
    state['position'][:, 0] += lag_m
    state['velocity'] = torch.tensor([[0., 0., 0.], [.5, 0., 0.]])
    reference['root_velocity'].zero_()
    old, _ = WorldBodyTracking().step(state, reference, reference['landmark_velocity'])
    new, _ = WorldBodyTracking('world-velocity-v1').step(state, reference, reference['landmark_velocity'])
    torch.testing.assert_close(new-old, 2*torch.exp(-torch.tensor([0., 1.])))


def test_velocity_score_is_bounded_finite_and_opt_in():
    state, reference, _ = sample()
    reference['root_velocity'].zero_()
    state['velocity'] = torch.tensor([[0., 0., 0.], [1e30, -1e30, 1e30]])
    value, parts = WorldBodyTracking('world-velocity-v1').step(
        state, reference, reference['landmark_velocity'])
    torch.testing.assert_close(value, torch.tensor([11.5, 9.5]))
    assert torch.isfinite(value).all()
    assert all(torch.isfinite(part).all() for part in parts.values())
    del state['velocity']
    # Existing world-body callers do not need the new velocity input.
    old, parts = WorldBodyTracking().step(state, reference, reference['landmark_velocity'])
    torch.testing.assert_close(old, torch.full((2,), 9.5))
    assert 'root_velocity' not in parts
    with pytest.raises(ValueError, match='profile'):
        WorldBodyTracking('unknown')


def test_native_velocity_reward_ppo_export_replay_and_resume_contract(tmp_path):
    from k1_motion.tracking_env import TrackerEnv
    from k1_motion.learning import TrainConfig, train, Policy
    from k1_motion.export import export_checkpoint
    from k1_motion.contracts import MotionClip
    from k1_motion.control_validation import replay_clip

    robot, library = make_library(tmp_path)
    torch.set_num_threads(1)
    config = TrainConfig(stage='student', iterations=2, horizon=8, epochs=1, minibatch=16,
        hidden_sizes=(32, 16), evaluation_interval=0, checkpoint_interval=1, bc_weight=0)
    settings = dict(history=3, observation_profile='preview', preview_horizon_s=.3,
        reward_profile='world-velocity-v1', safety_profile='casual-safe-v1',
        physics_options={'workers': 2},
        action_settings={'actuator_profile': 'booster-train-k1-actuator-v1',
                         'command_velocity_limit': (.8*robot.velocity_limit).tolist()})
    env = TrackerEnv(library, 2, 'cpu', 'mujoco_cpp', **settings)
    try:
        result = train(env, tmp_path/'training', config)
        assert result['finite_updates'] and result['checkpoint_reload_max_error'] == 0
        contract = result['reward_settings']['spatial_tracking']
        assert contract['version'] == 'world-velocity-v1'
        assert contract['weights']['root_velocity'] == 2.
        export_checkpoint(tmp_path/'training/checkpoint.pt', tmp_path/'export/actor.pt')
        replay = replay_clip(robot, Policy(tmp_path/'export/actor.pt', robot.signature),
                             MotionClip.load(library/'standing.npz'))
        assert np.isfinite(replay['world_body_rmse_m'])
        assert replay['resets_during_trial'] == 0
        assert replay['preview']['preview_horizon_s'] == .3
        other = TrackerEnv(library, 2, 'cpu', 'mujoco_cpp',
                           **{**settings, 'reward_profile': 'world-body-v1'})
        try:
            with pytest.raises(ValueError, match='reward settings'):
                train(other, tmp_path/'invalid', replace(config, iterations=1),
                      resume_checkpoint=tmp_path/'training/checkpoint.pt')
        finally:
            other.close()
    finally:
        env.close()
