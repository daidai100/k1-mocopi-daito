"""Failure cases: lost overrides on reload, changed legacy rewards, silent
invalid settings, or optimizer resume under a different translation objective.
"""
from dataclasses import replace
import json

import pytest
import torch

from k1_motion.world_objective import WorldBodyTracking


@pytest.mark.parametrize('settings', [
    {'weights': {'root_velocity': -1}}, {'weights': {'root_velocity': True}},
    {'weights': {'root_velocity': float('nan')}}, {'weights': {'unknown': 1}},
    {'scales': {'root_xy_m': 0}}, {'fall_penalty': 0},
])
def test_reject_invalid_settings(settings):
    with pytest.raises(ValueError):
        WorldBodyTracking('survival-position-v2', settings)


def test_translation_variants_and_default_are_independent():
    from test_causal_balanced_reward import inputs
    base = WorldBodyTracking('survival-position-v2')
    velocity = WorldBodyTracking('survival-position-v2', {'weights': {'root_xy_position': 5., 'root_velocity': 6.}})
    catchup = WorldBodyTracking('survival-position-v2', {'weights': {'root_xy_error': 1.}})
    state, reference, _ = inputs()
    state['position'][:, 0] += 2.
    _, a = base.step(state, reference, reference['landmark_velocity'])
    _, b = velocity.step(state, reference, reference['landmark_velocity'])
    _, c = catchup.step(state, reference, reference['landmark_velocity'])
    torch.testing.assert_close(b['weighted/root_velocity'], 2*a['weighted/root_velocity'])
    torch.testing.assert_close(c['weighted/root_xy_error'], 2*a['weighted/root_xy_error'])
    assert base.contract == WorldBodyTracking('survival-position-v2').contract
    assert base.contract['maximum_tracking_per_second'] == velocity.contract['maximum_tracking_per_second']
    with pytest.raises(ValueError):
        WorldBodyTracking('survival-position-v1', {'weights': {'root_velocity': 6.}})


def test_native_training_export_reconstruct_and_resume(tmp_path):
    from test_shuffled_survival import mixed_library
    from k1_motion.tracking_env import TrackerEnv
    from k1_motion.learning import TrainConfig, train
    from k1_motion.training_validation import checkpoint_environment_settings
    from k1_motion.export import export_checkpoint
    torch.set_num_threads(1)
    library = mixed_library(tmp_path/'library')
    settings = {'weights': {'root_xy_position': 5., 'root_velocity': 6.}}
    env = TrackerEnv(library, 2, 'cpu', 'mujoco_cpp', history=3,
        observation_profile='preview', reward_profile='survival-position-v2',
        safety_profile='casual-safe-v1', world_reward_settings=settings,
        physics_options={'workers': 2})
    cfg = TrainConfig(stage='student', iterations=2, horizon=32, epochs=1, minibatch=64,
        hidden_sizes=(16, 8), bc_weight=0, evaluation_interval=0, checkpoint_interval=1)
    try:
        report = train(env, tmp_path/'training', cfg)
        assert report['finite_updates'] and report['checkpoint_reload_max_error'] == 0
        checkpoint = tmp_path/'training/checkpoint.pt'
        saved = torch.load(checkpoint, weights_only=True)
        options = checkpoint_environment_settings(saved)
        restored = TrackerEnv(library, 2, 'cpu', 'mujoco_cpp', **options, physics_options={'workers': 2})
        try:
            assert restored.reward_settings == env.reward_settings
            train(restored, tmp_path/'resumed', replace(cfg, iterations=1), resume_checkpoint=checkpoint)
            restored.reward_settings['spatial_tracking']['weights']['root_velocity'] = 3.
            with pytest.raises(ValueError, match='reward settings'):
                train(restored, tmp_path/'wrong', cfg, resume_checkpoint=checkpoint)
        finally:
            restored.close()
        export_checkpoint(checkpoint, tmp_path/'export/actor.pt')
        metadata = json.loads((tmp_path/'export/actor.json').read_text())
        assert metadata['observation'] == saved['observation']
        assert metadata['checkpoint_iteration'] == 2
    finally:
        env.close()
