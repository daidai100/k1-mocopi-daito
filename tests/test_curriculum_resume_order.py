"""Resume first resets must sample restored state, once, after RNG restoration."""
import copy
from dataclasses import replace
import json

import pytest
import torch

from k1_motion.contracts import MotionClip
from k1_motion.learning import TrainConfig, train
from k1_motion.tracking_env import TrackerEnv
from k1_motion.training_curriculum import fidelity_curriculum_contract
from test_training import make_library
from test_minimal_casual_curriculum import rows_and_features  # noqa: F401 - scripts import setup
from prepare_minimal_casual_curriculum import build_contract


def make_case(tmp_path):
    robot, library = make_library(tmp_path)
    original = json.loads((library/'index.jsonl').read_text())
    rows = []
    for key, family, name in [('w','walk','walk_forward_001'),('g','gesture','wave_001')]:
        row = dict(original, id=key, family=family, take_name=name,
                   capture_group='synthetic/'+name, reference_path=key+'.npz')
        refs = [robot.neutral_reference(i*.02) for i in range(180)]
        MotionClip.from_references(refs, row).save(library/row['reference_path'])
        rows.append(row)
    (library/'index.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    contract = fidelity_curriculum_contract(build_contract(rows, {
        'w': {'event_longest_s': {'forward_travel': 1.}}, 'g': {'event_longest_s': {}}}))
    manifest = tmp_path/'curriculum.json'
    manifest.write_text(json.dumps(contract))
    config = TrainConfig(stage='student', iterations=1, horizon=1, epochs=1, minibatch=4,
                         hidden_sizes=(16,8), bc_weight=0, evaluation_interval=0,
                         checkpoint_interval=1, curriculum_manifest=str(manifest))
    def new_env():
        return TrackerEnv(library,4,'cpu','mujoco_cpp',history=3,observation_profile='preview',
                          reward_profile='world-body-v1',safety_profile='casual-safe-v1',
                          physics_options={'workers':2})
    torch.set_num_threads(1)
    env = new_env()
    try:
        train(env,tmp_path/'parent',config)
    finally:
        env.close()
    path = tmp_path/'parent/checkpoint.pt'
    saved = torch.load(path,map_location='cpu',weights_only=True)
    saved['curriculum_state']['failure_counts'][:] = torch.tensor([[1., 3., 9., 27.], [27., 9., 3., 1.]])
    saved['sampler_duration_ema'][:] = torch.tensor([12.,120.])
    saved['torch_rng_state'] = torch.Generator().manual_seed(2026092217).get_state()
    torch.save(saved,path)
    return new_env,config,path,saved


def test_first_resumed_physical_reset_uses_restored_sampler_rng_and_counts_once(tmp_path):
    new_env,config,path,saved = make_case(tmp_path)
    env = new_env()
    original_reset = env.reset
    seen = []
    def observe_reset(*args, **kwargs):
        seen.append(dict(failure_counts=env.curriculum.failure_counts.clone(),
                         duration=env.library.episode_duration_ema.clone(),
                         weights=env.library.weights.clone(),
                         rng=torch.get_rng_state().clone(),
                         window=env.curriculum.signal_window,
                         reset_counts=env.curriculum.reset_counts.clone()))
        result = original_reset(*args, **kwargs)
        seen[-1]['clips'] = env.clips.clone()
        seen[-1]['frames'] = env.frames.clone()
        seen[-1]['reset_counts_after'] = env.curriculum.reset_counts.clone()
        return result
    env.reset = observe_reset
    try:
        report = train(env,tmp_path/'resumed',config,resume_checkpoint=path)
        assert len(seen) == 1
        first = seen[0]
        torch.testing.assert_close(first['failure_counts'],saved['curriculum_state']['failure_counts'],rtol=0,atol=0)
        torch.testing.assert_close(first['duration'],saved['sampler_duration_ema'],rtol=0,atol=0)
        torch.testing.assert_close(first['rng'],saved['torch_rng_state'],rtol=0,atol=0)
        assert first['window'] is None
        assert first['reset_counts'].sum() == 0
        assert first['reset_counts_after'].sum() == env.num_envs
        target = torch.tensor([.5,.5])/saved['sampler_duration_ema']
        torch.testing.assert_close(first['weights'],target/target.sum(),rtol=0,atol=0)
        assert report['transitions'] == saved['transitions']+config.horizon*env.num_envs
        metrics = report['last_metrics']['curriculum']
        assert sum(metrics[key] for key in ('reset_start','reset_failure_biased','reset_uniform')) == env.num_envs
        persisted = torch.load(tmp_path/'resumed/checkpoint.pt',map_location='cpu',weights_only=True)
        assert persisted['curriculum_state']['transitions'] == saved['curriculum_state']['transitions']+env.num_envs
        # Rebuild just the reset from the checkpoint RNG and sampler state; this
        # reproduces clips/frames exactly without an intervening random draw.
        expected = new_env()
        try:
            from k1_motion.training_curriculum import TrainingCurriculum
            expected.library.episode_duration_ema.copy_(saved['sampler_duration_ema'])
            expected.curriculum = TrainingCurriculum(expected.library,config.curriculum_manifest)
            expected.curriculum.load_state_dict(saved['curriculum_state'])
            torch.set_rng_state(saved['torch_rng_state'])
            expected.reset()
            torch.testing.assert_close(first['clips'],expected.clips,rtol=0,atol=0)
            torch.testing.assert_close(first['frames'],expected.frames,rtol=0,atol=0)
        finally:
            expected.close()
    finally:
        env.close()


def test_resume_checks_actual_observation_shape_after_deferred_reset(tmp_path):
    new_env,config,path,_ = make_case(tmp_path)
    env = new_env()
    original_reset = env.reset
    def wrong_shape(*args, **kwargs):
        causal, privileged = original_reset(*args, **kwargs)
        return causal[:, :-1], privileged
    env.reset = wrong_shape
    try:
        with pytest.raises(ValueError,match='observation dimensions'):
            train(env,tmp_path/'bad-shape',config,resume_checkpoint=path)
    finally:
        env.close()


def test_incompatible_resume_rejects_before_any_physical_reset(tmp_path):
    new_env,config,path,saved = make_case(tmp_path)
    broken = copy.deepcopy(saved)
    broken['reward_settings']['self_collision_weight'] = 1.
    torch.save(broken,path)
    env = new_env()
    calls = []
    original_reset = env.reset
    def count_reset(*args, **kwargs):
        calls.append(True)
        return original_reset(*args, **kwargs)
    env.reset = count_reset
    try:
        with pytest.raises(ValueError,match='reward settings'):
            train(env,tmp_path/'bad-contract',replace(config,iterations=1),resume_checkpoint=path)
        assert calls == []
    finally:
        env.close()
