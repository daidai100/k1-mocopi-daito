"""Continuous recovery experience and smooth catch-up pressure.

Failure cases: pose gates or timeouts cut off recovery; long-lasting XY drift
is free; the position cost saturates, clips or has a dead zone; a handoff
reanchors away world error; a partial reset clears another world; falls stop
being terminal; the old checkpoint's task changes; PPO/reload/resume lose the
new objective. An isolated reward check verifies the error sensitivity that
native physics cannot hold constant while other reward terms change.
"""
from dataclasses import replace
import json
import math
import os
from pathlib import Path

import pytest
import torch

from k1_motion.learning import TrainConfig, train
from k1_motion.tracking_env import TrackerEnv
from test_shuffled_survival import mixed_library


def environment(path, profile='survival-position-v2', num_envs=2):
    return TrackerEnv(path, num_envs, 'cpu', 'mujoco_cpp', history=3,
        observation_profile='preview', reference_storage='packed',
        reward_profile=profile, safety_profile='casual-safe-v1', physics_options={'workers': 2},
        scene_transitions=dict(mode='shuffle', episode_seconds=120., random_initial_heading=False,
            max_heading_change_rad=0., pause_seconds=[.04, .06], candidate_count=32))


def start(env):
    ids = torch.zeros(env.num_envs, dtype=torch.long)
    return env.reset(clips=ids, frames=ids.clone())


def test_native_vague_pose_continues_and_fall_is_immediate(tmp_path):
    torch.set_num_threads(1)
    directory = mixed_library(tmp_path/'library')
    old, new = environment(directory, 'survival-position-v1'), environment(directory)
    try:
        for env in (old, new):
            start(env)
            env.scene_transitions.yaw[0] = torch.tensor([math.cos(math.pi/4), 0, 0, math.sin(math.pi/4)])
        for tick in range(8):
            _, _, _, old_done, _ = old.step(torch.zeros(2, 22), auto_reset=False)
            _, _, _, new_done, _ = new.step(torch.zeros(2, 22), auto_reset=False)
            assert not new_done.any()
            if tick >= 5:
                assert old_done.tolist() == [True, False]
            for key, value in new.physics.state().items():
                torch.testing.assert_close(value, old.physics.state()[key], rtol=0, atol=0)
        # Real inverted-body perturbation, without altering the tracking timer.
        ref = {k: v[:1].clone() for k, v in new.reference_frames(new.frames).items()}
        ref['root_position'][:, 2] = 1.
        ref['root_orientation'][:] = torch.tensor([0., 1., 0., 0.])
        new.physics.reset(torch.tensor([0]), ref)
        _, _, reward, done, info = new.step(torch.zeros(2, 22), auto_reset=False)
        assert done.tolist() == [True, False] and info['falls'] == 1
        assert new.last_step['failure_penalty'].tolist() == [10., 0.]
        assert new.last_step['survival_reward'].tolist() == pytest.approx([0., .04])
        assert reward[0] < -9
    finally:
        old.close()
        new.close()


def test_native_persistent_error_continues_and_recovery_is_rewarded(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(246)
    directory = mixed_library(tmp_path/'library')
    env = environment(directory)
    try:
        start(env)
        env.scene_transitions.shift[:, 0] += 1.3
        handoffs = 0
        for tick in range(600):
            if tick == 250:
                # The command comes back toward world 0. World 1 remains
                # displaced for twelve seconds, with no reset or deadline.
                env.scene_transitions.shift[0, 0] -= 1.
            _, _, _, done, info = env.step(torch.zeros(2, 22), auto_reset=False)
            handoffs += int(info['scene_transition_starts'])
            assert not done.any()
            assert info['tracking_failures'] == 0 and info['falls'] == 0
            assert env.last_step['failure_penalty'].tolist() == [0., 0.]
        assert handoffs > 50 and env.episode_steps.tolist() == [600, 600]
        ref = env.reference_frames(env.frames)
        state = env.physics.state()
        _, parts = env.spatial_reward.step(state, ref, torch.zeros(2, 17, 3))
        assert parts['root_xy_error_m'][1] > 1.2
        assert parts['weighted/root_xy_error'][0] > parts['weighted/root_xy_error'][1]+.2
        before = ref['root_position'][1].clone()
        env.reset(torch.tensor([0]), clips=torch.tensor([0]), frames=torch.tensor([0]))
        assert env.episode_steps.tolist() == [0, 600]
        torch.testing.assert_close(env.reference_frames(env.frames)['root_position'][1], before)
        artifact = Path(os.environ.get('K1_CONSISTENCY_ARTIFACT', tmp_path/'consistency.json'))
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(json.dumps(dict(version='relaxed-tracking-mechanics-v1', worlds=2,
            handoffs=handoffs, continuous_seconds=12., tracking_terminations=0,
            persistent_lag_m=float(parts['root_xy_error_m'][1]),
            position_cost_per_second=parts['weighted/root_xy_error'].tolist(),
            fall_penalty=10., tracking_timeout=None,
            fixture='native standing/head commands with controlled reference offsets',
            behaviorally_accepted=False), indent=2)+'\n')
    finally:
        env.close()


@pytest.mark.parametrize('error_m', [0., .01, .5, 1., 10., 100.])
def test_position_cost_is_continuous_and_keeps_sensitivity_far_away(error_m):
    from k1_motion.world_objective import WorldBodyTracking
    from test_causal_balanced_reward import inputs
    state, reference, _ = inputs()
    position = state['position'].clone()
    position[:, 0] += error_m
    position.requires_grad_()
    state['position'] = position
    objective = WorldBodyTracking('survival-position-v2')
    _, parts = objective.step(state, reference, reference['landmark_velocity'])
    cost = parts['weighted/root_xy_error']
    gradient, = torch.autograd.grad(cost.sum(), position)
    assert torch.isfinite(cost).all() and torch.isfinite(gradient).all()
    assert cost.max() <= 0
    if error_m == 0:
        torch.testing.assert_close(cost, torch.zeros_like(cost))
        torch.testing.assert_close(gradient, torch.zeros_like(gradient))
    else:
        assert (gradient[:, 0] < 0).all()
        if error_m >= 1:
            assert (gradient[:, 0].abs() >= .4).all()
        # A repeated discrepancy contributes in proportion to elapsed time.
        assert float((cost.detach()[0]*.02).repeat(250).sum()) == pytest.approx(float(cost.detach()[0])*5)


def test_native_ppo_export_resume_preserves_relaxed_contract(tmp_path):
    from k1_motion.export import export_checkpoint
    from k1_motion.training_validation import checkpoint_environment_settings
    torch.set_num_threads(1)
    directory = mixed_library(tmp_path/'library')
    env = environment(directory)
    config = TrainConfig(stage='student', iterations=2, horizon=32, epochs=1, minibatch=64,
        hidden_sizes=(16, 8), bc_weight=0, evaluation_interval=0, checkpoint_interval=1,
        gamma=math.exp(-.02/60), gae_lambda=.99)
    try:
        report = train(env, tmp_path/'training', config)
        assert report['finite_updates'] and report['checkpoint_reload_max_error'] == 0
        assert 'weighted/root_xy_error' in report['last_metrics']['reward_components']
        saved = torch.load(tmp_path/'training/checkpoint.pt', weights_only=True)
        assert saved['reward_settings']['spatial_tracking']['tracking_termination'] is False
        assert checkpoint_environment_settings(saved)['reward_profile'] == 'survival-position-v2'
        export_checkpoint(tmp_path/'training/checkpoint.pt', tmp_path/'export/actor.pt')
        metadata = json.loads((tmp_path/'export/actor.json').read_text())
        assert metadata['observation'] == saved['observation']
        resumed = train(env, tmp_path/'resumed', replace(config, iterations=1),
                        resume_checkpoint=tmp_path/'training/checkpoint.pt')
        assert resumed['transitions'] == report['transitions']+64
        old = environment(directory, 'survival-position-v1')
        try:
            with pytest.raises(ValueError, match='reward settings'):
                train(old, tmp_path/'wrong', config, resume_checkpoint=tmp_path/'training/checkpoint.pt')
        finally:
            old.close()
    finally:
        env.close()
