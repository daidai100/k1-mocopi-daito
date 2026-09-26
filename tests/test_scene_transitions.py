"""Native-physics scene handoffs, followed by PPO/export/resume.

Failure modes: resetting the body/history at a handoff; jumping world origins or
headings; inconsistent FK/derivatives in bridges; splicing unsafe poses; looping
the same recording; masking falls or refreshing the failure grace period;
charging a whole episode to its final recording; contaminating fixed-panel
replay; mutating shared references; forgetting the transition resume contract.
"""
from dataclasses import replace
import json

import mujoco
import numpy as np
import pytest
import torch

from k1_motion.contracts import MotionClip
from k1_motion.learning import TrainConfig, train
from k1_motion.math3d import quaternion
from k1_motion.robot import K1Model
from k1_motion.tracking_env import TrackerEnv


SETTINGS = dict(blend_seconds=.2, episode_seconds=2., candidate_count=2, cache_size=16)


def scene_library(path, count=3, incompatible=False):
    """Short head-motion scenes with identical support and different world frames."""
    from scipy.spatial.transform import Rotation
    robot = K1Model()
    path.mkdir()
    rows = []
    for c in range(count):
        row = dict(id=f'scene-{c}', capture_group=f'synthetic/scene-{c}',
                   family='gesture', model_signature=robot.signature, split='train',
                   source_kind='synthetic_debug', kinematics_accepted=True,
                   reference_path=f'{c}.npz')
        clip = MotionClip.from_references([robot.neutral_reference(i*.02) for i in range(16)], row)
        v = {k: a.copy() for k, a in clip.values.items()}
        yaw = Rotation.from_euler('z', c*np.pi/2)
        v['root_position'][:, :2] += [3*c, -2*c]
        v['root_orientation'][:] = quaternion(yaw)
        v['joint_position'][:, 0] += .02*np.sin(np.linspace(c, c+np.pi, len(clip.times)))
        v['joint_velocity'][1:] = np.diff(v['joint_position'], axis=0)/.02
        if incompatible and c:
            v['root_position'][:, 2] += .5
        data = mujoco.MjData(robot.model)
        for i in range(len(clip.times)):
            data.qpos[:] = np.r_[v['root_position'][i], v['root_orientation'][i], v['joint_position'][i]]
            mujoco.mj_kinematics(robot.model, data)
            v['landmarks'][i] = robot.landmarks(data)
        MotionClip(clip.times, v, clip.metadata).save(path/row['reference_path'])
        rows.append(row)
    (path/'index.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    return path


def environment(path, enabled=True, backend='mujoco_cpp', **kwargs):
    return TrackerEnv(path, 2, 'cpu', backend, history=3, observation_profile='preview',
        reference_storage='packed', reward_profile='world-velocity-v1',
        **({'physics_options': {'workers': 2}} if backend == 'mujoco_cpp' else {}),
        scene_transitions=SETTINGS if enabled else None, **kwargs)


def start(env, frame=0):
    return env.reset(clips=torch.zeros(env.num_envs, dtype=torch.long),
                     frames=torch.full((env.num_envs,), frame, dtype=torch.long))


@pytest.mark.parametrize('backend', ['mujoco', 'mujoco_cpp'])
def test_handoff_preserves_body_history_and_world_reference(tmp_path, backend):
    torch.set_num_threads(1)
    directory = scene_library(tmp_path/'library')
    enabled, control = environment(directory, backend=backend), environment(directory, False, backend)
    try:
        for env in (enabled, control):
            start(env, 14)
        source = {k: v.clone() for k, v in enabled.library.values.items()}
        expected = enabled.reference_frames(enabled.frames+1)
        action = torch.full((2,22), .001)
        _, _, _, done, info = enabled.step(action, auto_reset=False)
        control.step(action, auto_reset=False)
        assert not done.any() and info['scene_transition_starts'] == 2
        assert (enabled.clips != 0).all() and (enabled.frames == -10).all()
        assert (enabled.episode_steps == 1).all()
        for key, value in enabled.physics.state().items():
            torch.testing.assert_close(value, control.physics.state()[key], rtol=0, atol=0)
        torch.testing.assert_close(enabled.previous_action, action)
        current = enabled.reference_frames(enabled.frames)
        for key in ('root_position', 'root_orientation', 'joint_position', 'landmarks', 'root_velocity'):
            torch.testing.assert_close(current[key], expected[key], rtol=1e-5, atol=2e-6)
        for key, value in enabled.library.values.items():
            torch.testing.assert_close(value, source[key], rtol=0, atol=0)
        # A partial episode reset discards only that world's transform/bridge.
        other = {k: v[1].clone() for k, v in current.items()}
        enabled.reset(torch.tensor([0]), clips=torch.tensor([0]), frames=torch.tensor([0]))
        for key, value in enabled.reference_frames(enabled.frames).items():
            torch.testing.assert_close(value[1], other[key], rtol=0, atol=0)
        assert enabled.episode_steps.tolist() == [0, 1]
    finally:
        enabled.close()
        control.close()


def test_bridge_fk_derivatives_and_next_scene_are_consistent(tmp_path):
    directory = scene_library(tmp_path/'library')
    env = environment(directory)
    try:
        start(env, 14)
        env.step(torch.zeros(2,22), auto_reset=False)
        frames = []
        for frame in range(-10, 2):
            frames.append(env.reference_frames(torch.full((2,), frame)))
        data = mujoco.MjData(env.spec.model)
        for i, ref in enumerate(frames):
            data.qpos[:] = np.r_[ref['root_position'][0], ref['root_orientation'][0], ref['joint_position'][0]]
            mujoco.mj_kinematics(env.spec.model, data)
            np.testing.assert_allclose(env.spec.landmarks(data), ref['landmarks'][0], atol=2e-6)
            if 0 < i <= 10:
                torch.testing.assert_close(ref['joint_velocity'],
                    (ref['joint_position']-frames[i-1]['joint_position'])/.02, atol=1e-5, rtol=1e-4)
                torch.testing.assert_close(ref['root_velocity'][:, :3],
                    (ref['root_position']-frames[i-1]['root_position'])/.02, atol=1e-5, rtol=1e-4)
            assert torch.isfinite(torch.cat([v.reshape(-1) for v in ref.values()])).all()
        assert torch.max((frames[10]['root_position']-frames[9]['root_position']).abs()) < .01
        assert torch.max((frames[10]['joint_position']-frames[9]['joint_position']).abs()) < .01
        for _ in range(10):
            _, _, _, done, info = env.step(torch.zeros(2,22), auto_reset=False)
            assert not done.any()
        assert info['scene_transition_completions'] == 2
        assert env.frames.tolist() == [0,0]
        assert env.episode_steps.tolist() == [11,11]
    finally:
        env.close()


def test_curriculum_counts_sources_and_continuous_episode_separately(tmp_path):
    from k1_motion.training_curriculum import FAILURE_OBSERVATION, TrainingCurriculum
    directory = scene_library(tmp_path/'library')
    manifest = tmp_path/'curriculum.json'
    manifest.write_text(json.dumps(dict(version='minimal-casual-curriculum-v2',
        train_ids=['scene-0', 'scene-1', 'scene-2'], locomotion_ids=['scene-0'],
        target_transition_weights={'scene-0': .5, 'scene-1': .25, 'scene-2': .25},
        reset_mix={'start': .5, 'failure_biased': .25, 'uniform': .25},
        sampling={'locomotion_transition_share': .5}, failure_observation=FAILURE_OBSERVATION)))
    env = environment(directory, safety_profile='casual-safe-v1')
    try:
        env.curriculum = TrainingCurriculum(env.library, manifest)
        start(env)
        bridge_steps = ended_steps = episode_ends = 0
        for _ in range(100):
            _, _, _, _, info = env.step(torch.zeros(2,22))
            bridge_steps += int(info['scene_bridge_steps'])
            ended_steps += int(info['ended_episode_steps'])
            episode_ends += int(info['ended_episodes'])
        assert bridge_steps > 0 and episode_ends == 2 and ended_steps == 200
        assert env.library.transition_count.sum() == 200-bridge_steps
        assert env.curriculum.group_steps.sum() == 200-bridge_steps
        assert env.curriculum.reset_counts.sum() == 2  # Physical end resets only.
        metrics = env.curriculum.update()
        assert metrics['schedule_transitions'] == 200-bridge_steps
        assert all(0 <= v <= 1 for v in metrics['fidelity_bad_tick_share'].values())
    finally:
        env.close()


def test_longer_physical_episodes_and_per_scene_accounting(tmp_path):
    directory = scene_library(tmp_path/'library')
    totals = []
    for enabled in (False, True):
        env = environment(directory, enabled)
        try:
            start(env)
            episodes = steps = handoffs = bridges = ended_bridges = 0
            for _ in range(150):
                _, _, _, _, info = env.step(torch.zeros(2,22))
                assert info['falls'] == 0 and info['tracking_failures'] == 0
                episodes += int(info['ended_episodes'])
                steps += int(info['ended_episode_steps'])
                handoffs += int(info.get('scene_transition_starts', 0))
                bridges += int(info.get('scene_bridge_steps', 0))
                ended_bridges += int(info.get('ended_episode_bridge_steps', 0))
            totals.append(steps/episodes*.02)
            if enabled:
                assert handoffs >= 8 and bridges > 0
                assert 0 < ended_bridges < steps
                assert (steps-ended_bridges)/episodes*.02 > 1.
                assert float(env.library.transition_count.sum()) == 300-bridges
                assert env.library.episode_duration_ema.max() <= 16
                # Source-segment duration estimates cannot become chain duration.
                env.library.update_sampling()
                assert env.library.episode_duration_ema.max() <= 16
        finally:
            env.close()
    assert totals[1] == pytest.approx(2.)
    assert totals[1] > 3*totals[0]


def test_falls_during_bridge_and_episode_cap_still_terminate(tmp_path):
    directory = scene_library(tmp_path/'library')
    env = environment(directory, backend='mujoco')
    try:
        start(env, 14)
        env.step(torch.zeros(2,22), auto_reset=False)
        robot = env.physics.robots[0]
        robot.data.qpos[2] = 1.
        robot.data.qpos[3:7] = [0, 1, 0, 0]
        mujoco.mj_forward(robot.model, robot.data)
        _, _, _, done, info = env.step(torch.zeros(2,22), auto_reset=False)
        assert done.tolist() == [True, False]
        assert info['scene_transition_failures'] == 1 and info['falls'] == 1
        start(env, 14)
        env.episode_steps[:] = 99
        _, _, _, done, info = env.step(torch.zeros(2,22), auto_reset=False)
        assert done.all() and info['scene_episode_limits'] == 2
        assert info['scene_transition_starts'] == 0
    finally:
        env.close()


@pytest.mark.parametrize('count,incompatible', [(1,False), (2,True)])
def test_unavailable_or_incompatible_successor_ends_without_looping(tmp_path, count, incompatible):
    directory = scene_library(tmp_path/'library', count, incompatible)
    env = environment(directory)
    try:
        start(env, 14)
        _, _, _, done, info = env.step(torch.zeros(2,22), auto_reset=False)
        assert done.all() and info['scene_transition_starts'] == 0
        assert info['scene_transition_unavailable'] == 2
        assert env.clips.tolist() == [0,0]
    finally:
        env.close()


def test_native_ppo_checkpoint_resume_export_and_fixed_panel(tmp_path):
    from k1_motion.export import export_checkpoint
    from k1_motion.learning import ActorCritic
    from k1_motion.training_validation import replay_panel
    torch.set_num_threads(1)
    directory = scene_library(tmp_path/'library')
    config = TrainConfig(stage='student', iterations=2, horizon=32, epochs=1, minibatch=64,
        hidden_sizes=(16,8), bc_weight=0, evaluation_interval=0, checkpoint_interval=1)
    env = environment(directory)
    try:
        result = train(env, tmp_path/'train', config)
        assert result['finite_updates'] and result['checkpoint_reload_max_error'] == 0
        contract = result['scene_transitions']
        assert contract['version'] == 'scene-transitions-v1'
        assert result['last_metrics']['scene_transitions']['started'] > 0
        checkpoint = tmp_path/'train/checkpoint.pt'
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        assert saved['scene_transitions'] == contract
        export_checkpoint(checkpoint, tmp_path/'export/actor.pt')
        assert json.loads((tmp_path/'export/actor.json').read_text())['scene_transitions'] == contract
        resumed = train(env, tmp_path/'resume', replace(config, iterations=1), resume_checkpoint=checkpoint)
        assert resumed['transitions'] == result['transitions']+64
        model = ActorCritic(saved['actor_size'], saved['critic_size'], (16,8))
        model.load_state_dict(saved['model'])
        manager = env.scene_transitions
        replay = replay_panel(env, model, 'student', directory)
        assert replay['total'] == 3
        assert all(t['simulated_s'] <= .30001 for t in replay['trials'])
        assert env.scene_transitions is manager
    finally:
        env.close()
    wrong = environment(directory, False)
    try:
        with pytest.raises(ValueError, match='scene transition'):
            train(wrong, tmp_path/'wrong-resume', config, resume_checkpoint=checkpoint)
    finally:
        wrong.close()


@pytest.mark.parametrize('settings', [dict(blend_seconds=0), dict(episode_seconds=float('nan')),
    dict(candidate_count=0), dict(cache_size=-1), dict(blend_seconds=.015), dict(typo=1)])
def test_invalid_settings_fail_explicitly(tmp_path, settings):
    directory = scene_library(tmp_path/'library')
    with pytest.raises(ValueError, match='[Ss]cene transition'):
        TrackerEnv(directory, 2, 'cpu', scene_transitions=settings)
