"""Repeatable native-physics composition -> PPO -> export -> resume checks.

Failure cases: seams reset body/history/survival; worlds share the same shuffle;
pauses freeze unsupported poses; input jumps bypass actuator limits; seam
velocities use recording-reset zeros; headings/world origins jump arbitrarily;
composition mutates sources or enters held-out replay; short clips terminate
long scenes; joint errors sum over DoFs or get integrated twice; falls receive
alive credit; rewards/sampling depend on learning progress; resume changes the
discount horizon silently. Synthetic fixtures verify mechanics, not locomotion.
"""
from dataclasses import replace
import json
import math
import os
from pathlib import Path

import mujoco
import pytest
import torch

from k1_motion.contracts import MotionClip
from k1_motion.learning import TrainConfig, train
from k1_motion.tracking_env import TrackerEnv
from test_scene_transitions import scene_library


def mixed_library(path):
    from k1_motion.robot import K1Model
    from k1_motion.standing_padding import endpoint_support
    scene_library(path, count=8)
    rows = [json.loads(line) for line in (path/'index.jsonl').read_text().splitlines()]
    for row, family in zip(rows, ('gesture', 'other', 'walk', 'walk', 'idle_stance', 'bow', 'run', 'run')):
        clip = MotionClip.load(path/row['reference_path'])
        row['family'] = family
        row['standing_padding'] = dict(finish=endpoint_support(clip, -1, K1Model()))
        clip.metadata.update(row)
        clip.save(path/row['reference_path'])
    (path/'index.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    return path


def environment(directory, *, seconds=2., num_envs=4, backend='mujoco_cpp', **settings):
    return TrackerEnv(directory, num_envs, 'cpu', backend, history=3,
        observation_profile='preview', reward_profile='survival-position-v1',
        safety_profile='casual-safe-v1', reference_storage='packed',
        **({'physics_options': {'workers': 2}} if backend == 'mujoco_cpp' else {}),
        scene_transitions=dict(mode='shuffle', episode_seconds=seconds,
            random_initial_heading=False, max_heading_change_rad=0.,
            pause_seconds=[.04, .06], candidate_count=32, **settings))


def test_two_minute_shuffled_native_scene(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(123)
    directory = mixed_library(tmp_path/'library')
    env = environment(directory, seconds=120.)
    artifact = Path(os.environ.get('K1_SHUFFLE_ARTIFACT', tmp_path/'receipt.json'))
    try:
        env.reset(clips=torch.zeros(4, dtype=torch.long), frames=torch.zeros(4, dtype=torch.long))
        source = {k: v.clone() for k, v in env.library.values.items()}
        alive_credit = torch.zeros(4)
        seen, pauses, handoffs = set(), 0, 0
        for tick in range(6000):
            previous = env.clips.clone()
            _, _, _, done, info = env.step(torch.zeros(4, 22), auto_reset=False)
            assert not done.any() if tick < 5999 else done.all()
            assert info['falls'] == 0 and info['tracking_failures'] == 0
            alive_credit += env.last_step['survival_reward']
            handoffs += int(info['scene_transition_starts'])
            pauses += int(info['scene_bridge_steps'])
            for i in (previous != env.clips).nonzero().flatten().tolist():
                slot = int(env.scene_transitions.slots[i])
                role = env.scene_transition_contract['pattern'][slot]
                family = env.library.rows[int(env.clips[i])]['family']
                assert family in env.scene_transition_contract['families'][role]
                seen.add((i, int(env.clips[i]), slot))
        torch.testing.assert_close(alive_credit, torch.full((4,), 240.), rtol=1e-4, atol=.02)
        assert env.episode_steps.tolist() == [6000]*4
        assert handoffs > 500 and pauses > 0 and len(seen) > 32
        assert info['scene_episode_limits'] == 4
        for k, v in env.library.values.items():
            torch.testing.assert_close(v, source[k], rtol=0, atol=0)
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(json.dumps(dict(version='shuffled-survival-mechanics-v1',
            physical_seconds_per_world=120, worlds=4, handoffs=handoffs,
            pause_ticks=pauses, falls=0, tracking_failures=0,
            survival_return_per_world=alive_credit.tolist(),
            originals=8, fixture='synthetic neutral/head commands with role labels',
            behaviorally_accepted=False), indent=2)+'\n')
    finally:
        env.close()


def test_handoff_preserves_physics_and_history_and_bounds_motor_commands(tmp_path):
    torch.set_num_threads(1)
    directory = mixed_library(tmp_path/'library')
    env = environment(directory, num_envs=2)
    control = TrackerEnv(directory, 2, 'cpu', 'mujoco_cpp', history=3,
        observation_profile='preview', reference_storage='packed', physics_options={'workers': 2})
    try:
        for e in (env, control):
            e.reset(clips=torch.zeros(2, dtype=torch.long), frames=torch.full((2,), 14))
        action = torch.full((2, 22), .001)
        _, _, _, done, info = env.step(action, auto_reset=False)
        control.step(action, auto_reset=False)
        assert not done.any() and info['scene_transition_starts'] == 2
        for k, v in env.physics.state().items():
            torch.testing.assert_close(v, control.physics.state()[k], rtol=0, atol=0)
        torch.testing.assert_close(env.previous_action, action)
        assert (env.episode_steps == 1).all()
        for _ in range(20):
            target = env.previous_target.clone()
            previous_ref = env.reference_frames(env.frames)
            _, _, _, _, info = env.step(action, auto_reset=False)
            assert ((env.previous_target-target).abs() <= env.command_step_limit+1e-6).all()
            if (env.frames == 0).any():
                ids = env.frames == 0
                ref = env.reference_frames(env.frames)
                torch.testing.assert_close(ref['joint_velocity'][ids],
                    (ref['joint_position'][ids]-previous_ref['joint_position'][ids])/.02,
                    atol=1e-5, rtol=1e-4)
        other = env.reference_frames(env.frames)['root_position'][1].clone()
        env.reset(torch.tensor([0]), clips=torch.tensor([0]), frames=torch.tensor([0]))
        torch.testing.assert_close(env.reference_frames(env.frames)['root_position'][1], other)
        assert env.episode_steps.tolist() == [0, 21]
    finally:
        env.close()
        control.close()


def test_fixed_reward_priorities_joint_mean_auc_and_fall(tmp_path):
    from k1_motion.world_objective import WorldBodyTracking
    from test_causal_balanced_reward import inputs
    objective = WorldBodyTracking('survival-position-v1')
    state, reference, support = inputs()
    maximum, baseline = objective.step(state, reference, reference['landmark_velocity'], support)
    state['q'][0, 0] += .22
    state['q'][1, :] += .01
    _, changed = objective.step(state, reference, reference['landmark_velocity'], support)
    torch.testing.assert_close(changed['joint_mae_rad'], torch.full((2,), .01), atol=1e-7, rtol=1e-5)
    torch.testing.assert_close(changed['weighted/joint_error'], torch.full((2,), -.0025))
    state, reference, support = inputs()
    state['position'][:, 0] += .5
    state['landmarks'][:, :, 0] += .5
    shifted, _ = objective.step(state, reference, reference['landmark_velocity'], support)
    torch.testing.assert_close(maximum-shifted, torch.full((2,), 4.))
    assert 'curriculum' not in objective.contract or objective.contract['curriculum'] is False
    # Integrating dt * mean joint absolute error is an AUC; splitting the
    # interval or changing the logging frequency must not change its weight.
    assert float((changed['joint_mae_rad'][0]*.02).repeat(50).sum()) == pytest.approx(.01)
    directory = mixed_library(tmp_path/'library')
    env = environment(directory, backend='mujoco', num_envs=2)
    try:
        env.reset(clips=torch.zeros(2, dtype=torch.long), frames=torch.zeros(2, dtype=torch.long))
        for _ in range(2):
            env.step(torch.zeros(2, 22), auto_reset=False)
        assert env.episode_joint_error_auc.min() > 0
        robot = env.physics.robots[0]
        robot.data.qpos[2] = 1.
        robot.data.qpos[3:7] = [0, 1, 0, 0]
        mujoco.mj_forward(robot.model, robot.data)
        _, _, reward, done, info = env.step(torch.zeros(2, 22), auto_reset=False)
        assert done.tolist() == [True, False] and info['falls'] == 1
        assert env.last_step['survival_reward'].tolist() == pytest.approx([0., .04])
        assert env.last_step['failure_penalty'].tolist() == [10., 0.]
        assert reward[0] < -9.
        prior = env.episode_joint_error_auc[1].clone()
        env.reset(torch.tensor([0]), clips=torch.tensor([0]), frames=torch.tensor([0]))
        assert env.episode_joint_error_auc[0] == 0
        assert env.episode_joint_error_auc[1] == prior
    finally:
        env.close()


def test_random_headings_and_unsupported_pause_and_empty_role(tmp_path):
    torch.set_num_threads(1)
    directory = mixed_library(tmp_path/'library')
    rows = [json.loads(line) for line in (directory/'index.jsonl').read_text().splitlines()]
    for r in rows:
        r['standing_padding']['finish']['feasible'] = False
    (directory/'index.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    env = TrackerEnv(directory, 32, 'cpu', 'mujoco_cpp', reference_storage='packed',
        physics_options={'workers': 2}, scene_transitions=dict(mode='shuffle'))
    try:
        env.reset(clips=torch.zeros(32, dtype=torch.long), frames=torch.full((32,), 14))
        assert env.physics.state()['orientation'][:, 3].std() > .3
        previous_position = env.reference_frames(env.frames+1)['root_position']
        _, _, _, _, info = env.step(torch.zeros(32, 22), auto_reset=False)
        assert info['scene_transition_starts'] == 32
        assert (env.frames == 0).all()  # No unsupported synthetic holds.
        after = env.reference_frames(env.frames)['root_position']
        torch.testing.assert_close(after[:, :2], previous_position[:, :2], atol=2e-6, rtol=0)
    finally:
        env.close()
    rows = [r for r in rows if r['family'] != 'run']
    (directory/'index.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError, match='run'):
        environment(directory)


def test_resets_rotate_the_fixed_pattern_to_preserve_locomotion_exposure(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(246)
    directory = mixed_library(tmp_path/'library')
    env = environment(directory, num_envs=32)
    try:
        env.reset()
        slots = env.scene_transitions.slots.clone()
        roles = [env.scene_transition_contract['pattern'][i] for i in slots.tolist()]
        assert set(roles) == {'motion', 'walk', 'easy', 'run'}
        for role, index in zip(roles, env.clips.tolist()):
            assert env.library.rows[index]['family'] in env.scene_transition_contract['families'][role]
        for _ in range(15):
            _, _, _, done, _ = env.step(torch.zeros(32, 22), auto_reset=False)
            assert not done.any()
        torch.testing.assert_close(env.scene_transitions.slots, (slots+1) % 8)
    finally:
        env.close()


def test_shuffled_ppo_export_resume_and_original_panel(tmp_path):
    from k1_motion.export import export_checkpoint
    from k1_motion.learning import ActorCritic
    from k1_motion.training_validation import replay_panel
    torch.set_num_threads(1)
    directory = mixed_library(tmp_path/'library')
    env = environment(directory)
    config = TrainConfig(stage='student', iterations=2, horizon=32, epochs=1, minibatch=64,
        hidden_sizes=(16, 8), bc_weight=0, evaluation_interval=0, checkpoint_interval=1,
        gamma=math.exp(-.02/60), gae_lambda=.99)
    try:
        result = train(env, tmp_path/'train', config)
        assert result['finite_updates'] and result['checkpoint_reload_max_error'] == 0
        assert result['last_metrics']['scene_transitions']['started'] > 0
        assert result['last_metrics']['episode_objective']['rollout_joint_error_auc_rad_s_per_world'] > 0
        assert 'ended_mean_joint_error_auc_rad_s' in result['last_metrics']['episode_objective']
        checkpoint = tmp_path/'train/checkpoint.pt'
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        assert saved['scene_transitions']['version'] == 'shuffled-scenes-v1'
        assert saved['config']['gamma'] == config.gamma
        export_checkpoint(checkpoint, tmp_path/'export/actor.pt')
        assert json.loads((tmp_path/'export/actor.json').read_text())['scene_transitions'] == saved['scene_transitions']
        resumed = train(env, tmp_path/'resume', replace(config, iterations=1), resume_checkpoint=checkpoint)
        assert resumed['transitions'] == result['transitions']+128
        with pytest.raises(ValueError, match='discount'):
            train(env, tmp_path/'bad-resume', replace(config, gamma=.99), resume_checkpoint=checkpoint)
        model = ActorCritic(saved['actor_size'], saved['critic_size'], (16, 8))
        model.load_state_dict(saved['model'])
        manager = env.scene_transitions
        replay = replay_panel(env, model, 'student', directory)
        assert replay['total'] == 8 and all(t['simulated_s'] <= .30001 for t in replay['trials'])
        assert env.scene_transitions is manager
    finally:
        env.close()


@pytest.mark.parametrize('settings', [dict(episode_seconds=float('nan')),
    dict(pause_seconds=[.3, .1]), dict(max_joint_jump_rad=-1), dict(candidate_count=0),
    dict(pattern=['walk', 'missing']), dict(unknown=True)])
def test_invalid_shuffle_contract(settings):
    from k1_motion.scene_transitions import transition_contract
    with pytest.raises(ValueError):
        transition_contract(dict(mode='shuffle', **settings), .02)
