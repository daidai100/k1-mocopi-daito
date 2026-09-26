"""Failure cases: hidden heading/lag, wrong angular frame, reward drift,
invalid scales, lost reward identity, clip-vs-transition weighting, and queues
that exceed the requested budget or move a matched pair between GPUs.
"""
import json
from pathlib import Path
import sys

import pytest
import torch

from k1_motion.world_objective import WorldBodyTracking
from test_reward_series import sample


def test_decomposed_frames_huber_and_yaw_rate():
    state, ref, _ = sample()
    state['omega'] = torch.zeros(2, 3)
    reward = WorldBodyTracking('world-decomposed-v1')
    peak, _ = reward.step(state, ref, ref['landmark_velocity'])
    torch.testing.assert_close(peak, torch.full((2,), 9.5))
    state['position'][:, 0] += .5
    state['landmarks'][:, :, 0] += .5
    value, parts = reward.step(state, ref, ref['landmark_velocity'])
    torch.testing.assert_close(parts['body_relative'], torch.ones(2))
    torch.testing.assert_close(parts['root_xy_cost'], torch.full((2,), -.5))
    assert (value < peak).all()
    state['omega'][:, 2] = .5
    _, changed = reward.step(state, ref, ref['landmark_velocity'])
    assert (changed['yaw_rate'] < parts['yaw_rate']).all()
    torch.testing.assert_close(changed['root_velocity'], parts['root_velocity'])
    # Independent heading errors remain visible even with unchanged point data.
    state['orientation'][:] = torch.tensor([.70710678, 0, 0, .70710678])
    _, changed = reward.step(state, ref, ref['landmark_velocity'])
    assert (changed['heading'] < .01).all()


def test_decomposed_configuration_and_negative_tail_are_explicit():
    state, ref, _ = sample()
    state['omega'] = torch.zeros(2, 3)
    with pytest.raises(ValueError):
        WorldBodyTracking('world-decomposed-v1', settings={'scales': {'root_xy_m': 0}})
    with pytest.raises(ValueError):
        WorldBodyTracking('world-body-v1', settings={'weights': {'world_position': 1}})
    reward = WorldBodyTracking('world-decomposed-v1', settings={'weights': {'root_xy_cost': .5}})
    assert reward.contract['weights']['root_xy_cost'] == .5
    state['position'][:, 0] = 100
    state['landmarks'][:, :, 0] += 100
    value, parts = reward.step(state, ref, ref['landmark_velocity'])
    assert torch.isfinite(value).all() and (value < 0).all()
    assert 'early_termination_risk' in reward.contract


def test_real_native_training_export_reload_with_easy_mix(tmp_path):
    from test_training import make_library
    from k1_motion.tracking_env import TrackerEnv
    from k1_motion.learning import TrainConfig, train, Policy
    from k1_motion.export import export_checkpoint
    from k1_motion.contracts import MotionClip
    from k1_motion.control_validation import replay_clip
    robot, library = make_library(tmp_path)
    rows = [json.loads(s) for s in (library/'index.jsonl').read_text().splitlines()]
    rows = [{**rows[0], 'id': 'walk'}, {**rows[0], 'id': 'retention', 'capture_group': 'synthetic/retention'}]
    (library/'index.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    ids = [r['id'] for r in rows if r['split'] == 'train']
    assert len(ids) >= 2
    weights = {key: .8 if i == 0 else .2/(len(ids)-1) for i, key in enumerate(ids)}
    manifest = tmp_path/'mix.json'
    manifest.write_text(json.dumps(dict(version='easy-walk-curriculum-v1', train_ids=ids,
        locomotion_ids=ids[:1], target_transition_weights=weights,
        reset_mix={'start': .5, 'failure_biased': .25, 'uniform': .25},
        sampling={'locomotion_transition_share': .8})))
    torch.set_num_threads(1)
    env = TrackerEnv(library, 4, 'cpu', 'mujoco_cpp', history=3,
        observation_profile='preview', reward_profile='world-decomposed-v1',
        safety_profile='casual-safe-v1', physics_options={'workers': 2},
        action_settings={'actuator_profile': 'booster-train-k1-actuator-v1',
                         'command_velocity_limit': (.8*robot.velocity_limit).tolist()})
    try:
        result = train(env, tmp_path/'training', TrainConfig(stage='student', iterations=2,
            horizon=8, epochs=1, minibatch=16, hidden_sizes=(32, 16), evaluation_interval=0,
            checkpoint_interval=1, bc_weight=0, curriculum_manifest=str(manifest)))
        assert result['finite_updates'] and result['checkpoint_reload_max_error'] == 0
        assert result['reward_settings']['spatial_tracking']['version'] == 'world-decomposed-v1'
        # Expected transition mass, after undoing the episode-duration correction.
        sampler = env.library.weights * env.library.episode_duration_ema.clamp(min=1)
        sampler /= sampler.sum()
        torch.testing.assert_close(sampler[env.curriculum.locomotion].sum(), torch.tensor(.8))
        rows = [json.loads(s) for s in (tmp_path/'training/metrics.jsonl').read_text().splitlines()]
        assert rows[-1]['curriculum']['target_locomotion_transition_share'] == .8
        assert 'weighted/root_xy_cost' in rows[-1]['reward_components']
        assert all(torch.isfinite(torch.tensor(rows[-1][key])) for key in
                   ('policy_entropy', 'critic_explained_variance', 'fixed_observation_action_change_rms'))
        export_checkpoint(tmp_path/'training/checkpoint.pt', tmp_path/'actor.pt')
        result = replay_clip(robot, Policy(tmp_path/'actor.pt', robot.signature),
                             MotionClip.load(library/'standing.npz'))
        assert result['resets_during_trial'] == 0
        assert result['actuator_safety']['sample_period_s'] == .002
    finally:
        env.close()


def test_three_matched_pairs_and_finite_queue_budget(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
    from run_easy_walk_campaign import build_plans
    root = tmp_path/'bundle'
    plans = build_plans(root, tmp_path/'output', 'server', 2000)
    local = build_plans(root, tmp_path/'desktop', 'desktop', 2000)
    assert len(plans) == 4 and len(local) == 2
    all_plans = plans + local
    assert {p['seed'] for p in all_plans} == {42, 43, 44}
    for seed in (42, 43, 44):
        pair = [p for p in all_plans if p['seed'] == seed]
        assert len(pair) == 2
        assert pair[0]['slot'] == pair[1]['slot']
        assert pair[0]['environment'] == pair[1]['environment']
        assert {p['reward_profile'] for p in pair} == {'world-body-v1', 'world-decomposed-v1'}
        assert all(p['command'][p['command'].index('--iterations')+1] == '2000' for p in pair)
    with pytest.raises(ValueError):
        build_plans(root, tmp_path/'bad', 'server', 2001)
