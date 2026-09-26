"""Failure contracts first: labels are not motion, duplication is not diversity,
budgets must not change task mix, bad manifests must fail, and resumes must agree.
"""
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from k1_motion.training_curriculum import TrainingCurriculum

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


def rows_and_features():
    rows = [
        dict(id='w1', family='walk', take_name='walk_ff_loop_001', capture_group='bones_seed/walk_ff_loop_001', split='train'),
        dict(id='w2', family='walk', take_name='walk_ff_loop_002', capture_group='bones_seed/walk_ff_loop_002', split='train'),
        dict(id='w3', family='walk', take_name='walk_back_loop_001', capture_group='bones_seed/walk_back_loop_001', split='train'),
        dict(id='v', family='turn', take_name='high_big_valve_cw_001', capture_group='bones_seed/high_big_valve_cw_001', split='train'),
        dict(id='d', family='walk', take_name='dance_vogue_cat_walk_001', capture_group='bones_seed/dance_vogue_cat_walk_001', split='train'),
        dict(id='g', family='gesture', take_name='wave_001', capture_group='bones_seed/wave_001', split='train'),
        dict(id='s', family='walk', take_name='walk_stationary_001', capture_group='bones_seed/walk_stationary_001', split='train'),
    ]
    features = {r['id']: {'event_longest_s': {'forward_travel': 1., 'turning': 1.}} for r in rows}
    features['s'] = {'event_longest_s': {'forward_travel': 0., 'turning': 0.}}
    return rows, features


def test_manifest_uses_intent_and_measured_motion_preserves_all_ids_and_take_balance():
    from prepare_minimal_casual_curriculum import build_contract
    rows, features = rows_and_features()
    before = json.dumps(rows, sort_keys=True)
    contract = build_contract(rows, features)
    assert set(contract['train_ids']) == {r['id'] for r in rows}
    assert set(contract['locomotion_ids']) == {'w1', 'w2', 'w3'}
    weights = contract['target_transition_weights']
    assert sum(weights.values()) == pytest.approx(1.)
    assert weights['w1'] + weights['w2'] == pytest.approx(weights['w3'])
    assert sum(weights[x] for x in contract['locomotion_ids']) == pytest.approx(.5)
    assert all(v > 0 for v in weights.values())
    assert json.dumps(rows, sort_keys=True) == before
    assert contract == build_contract(list(reversed(rows)), features)
    for bad in [[*rows, rows[0]], [{**rows[0], 'split': 'validation'}, *rows[1:]],
                [{**rows[0], 'is_mirror': True}, *rows[1:]]]:
        with pytest.raises(ValueError):
            build_contract(bad, features)
    with pytest.raises(ValueError, match='feature'):
        build_contract(rows, {})


def test_native_family_labels_cannot_amplify_a_single_take():
    from prepare_minimal_casual_curriculum import build_contract
    rows, features = rows_and_features()
    rows[2]['family'] = 'jump'  # A historically bad native label must not get a family budget.
    rows.append(dict(rows[2], id='w4', family='walk', capture_group='bones_seed/walk_left_001'))
    features['w4'] = features['w3']
    rows.append(dict(rows[3], id='v2', family='kneel', capture_group=rows[3]['capture_group']))
    features['v2'] = features['v']
    contract = build_contract(rows, features)
    weights = contract['target_transition_weights']
    assert weights['w1'] + weights['w2'] == pytest.approx(1/6)
    assert weights['w3'] == pytest.approx(weights['w4'])
    assert weights['w3'] == pytest.approx(1/6)
    assert weights['v'] + weights['v2'] == pytest.approx(weights['g'])


def test_fixed_exposure_reset_mix_validation_and_resume(tmp_path):
    from prepare_minimal_casual_curriculum import build_contract
    rows, features = rows_and_features()
    contract = build_contract(rows, features)
    path = tmp_path / 'curriculum.json'
    path.write_text(json.dumps(contract))
    ordered = list(reversed(rows))
    library = SimpleNamespace(rows=ordered, device='cpu', lengths=torch.full((7,), 151),
                              take_weights=torch.ones(7), episode_duration_ema=torch.arange(1, 8)*13.)
    sampler = TrainingCurriculum(library, path)
    target = torch.tensor([contract['target_transition_weights'][r['id']] for r in ordered])
    for transitions in [0, 8_192_000, 65_536_000, 2_000_000_000]:
        sampler.transitions = transitions
        sampler.apply_weights()
        exposure = library.weights * library.episode_duration_ema
        torch.testing.assert_close(exposure / exposure.sum(), target)
    torch.manual_seed(20260922)
    clips = torch.arange(40000) % len(rows)
    frames = sampler.sample_frames(clips)
    assert int(frames.min()) == 0 and int(frames.max()) < 150
    sampler.record(clips, frames, torch.zeros_like(clips, dtype=torch.bool))
    metrics = sampler.update()
    starts, biased, uniform = [metrics[k] / len(clips) for k in ['reset_start', 'reset_failure_biased', 'reset_uniform']]
    assert .49 < starts < .51 and .24 < biased < .26 and .24 < uniform < .26
    assert metrics['target_locomotion_transition_share'] == .5
    assert sum(metrics['group_transition_share'].values()) == pytest.approx(1.)
    restored = TrainingCurriculum(library, path)
    restored.load_state_dict(sampler.state_dict())
    torch.testing.assert_close(restored.library.weights, sampler.library.weights)
    assert restored.transitions == sampler.transitions
    for bad in [dict(contract, target_transition_weights={'w1': 1.}),
                dict(contract, target_transition_weights={**contract['target_transition_weights'], 'w1': -1.}),
                dict(contract, reset_mix={'start': .25, 'failure_biased': .5, 'uniform': .25})]:
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError, match='casual'):
            TrainingCurriculum(library, path)


def test_minimal_casual_real_ppo_export_replay_resume(tmp_path):
    from prepare_minimal_casual_curriculum import build_contract
    from test_training import make_library
    from k1_motion.contracts import MotionClip
    from k1_motion.control_validation import replay_clip
    from k1_motion.export import export_checkpoint
    from k1_motion.learning import Policy, TrainConfig, train
    from k1_motion.tracking_env import TrackerEnv
    robot, directory = make_library(tmp_path)
    original = json.loads((directory / 'index.jsonl').read_text())
    rows = []
    for key, family, name in [('w', 'walk', 'walk_forward_001'), ('g', 'gesture', 'wave_001')]:
        row = {**original, 'id': key, 'family': family, 'take_name': name,
               'capture_group': 'synthetic/' + name, 'reference_path': key + '.npz'}
        clip = MotionClip.load(directory / 'standing.npz')
        clip.metadata.update(row)
        clip.save(directory / row['reference_path'])
        rows.append(row)
    (directory / 'index.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    manifest = tmp_path / 'casual.json'
    manifest.write_text(json.dumps(build_contract(rows, {'w': {'event_longest_s': {'forward_travel': 1.}},
                                                          'g': {'event_longest_s': {}}})))
    torch.set_num_threads(1)
    env = TrackerEnv(directory, 2, 'cpu', 'mujoco_cpp', history=3,
                     reference_storage='packed', physics_options={'workers': 2})
    env.curriculum = TrainingCurriculum(env.library, manifest)
    cfg = TrainConfig(stage='student', iterations=2, horizon=4, epochs=1, minibatch=8,
                      hidden_sizes=(32, 16), evaluation_interval=0, checkpoint_interval=1, bc_weight=0)
    try:
        report = train(env, tmp_path/'training', cfg)
        assert report['finite_updates'] and report['checkpoint_reload_max_error'] == 0
        export_checkpoint(tmp_path/'training/checkpoint.pt', tmp_path/'export/actor.pt')
        replay = replay_clip(robot, Policy(tmp_path/'export/actor.pt', robot.signature),
                             MotionClip.load(directory/'g.npz'))
        assert replay['resets_during_trial'] == 0
        assert np.isfinite(replay['relative_body_rmse_m'])
        resumed = train(env, tmp_path/'resumed', replace(cfg, iterations=1),
                        resume_checkpoint=tmp_path/'training/checkpoint.pt')
        assert resumed['optimizer_steps'] > report['optimizer_steps']
    finally:
        env.close()
