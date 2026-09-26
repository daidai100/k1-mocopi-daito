"""The expanded zero-update diagnostic must not impersonate new training."""
import importlib.util
from pathlib import Path

import torch

from k1_motion.actuation import action_settings
from k1_motion.actuators import ACTUATOR_PROFILE
from k1_motion.learning import ActorCritic
from k1_motion.observations import observation_contract
from k1_motion.robot import K1Model


def test_expansion_preserves_actions_provenance_and_zero_new_exposure():
    spec = importlib.util.spec_from_file_location('prepare_preview_baseline',
           Path(__file__).resolve().parents[1]/'scripts/prepare_preview_baseline.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.set_num_threads(1)
    observation = observation_contract(10,'causal')
    model = ActorCritic(observation['size'],observation['size']+7,(8,4))
    inherited = dict(model=model.state_dict(),optimizer={'not':'resumable'},stage='student',
                     actor_size=observation['size'],critic_size=observation['size']+7,
                     observation=observation,hidden_sizes=[8,4],iteration=2500,
                     transitions=655360000,train_parents=['original_training_take'],
                     model_signature=K1Model().signature,reference_fingerprint='original-fingerprint',
                     source_revision='old-source')
    robot = K1Model()
    settings = action_settings(robot,dict(actuator_profile=ACTUATOR_PROFILE,
                                command_velocity_limit=(.8*robot.velocity_limit).tolist()))
    checkpoints,proof = module.prepare_checkpoints(inherited,robot,settings,'new-source','sha-initializer')
    assert proof['new_actor_size']==1680
    assert proof['zero_added_actor_columns']==320
    assert proof['future_feature_action_max_error']==0
    assert proof['inherited_actor_action_max_error']<2e-6
    for horizon,checkpoint in checkpoints.items():
        assert horizon in (0.,.3)
        assert checkpoint['train_parents']==inherited['train_parents']
        assert checkpoint['reference_fingerprint']=='original-fingerprint'
        assert checkpoint['iteration']==checkpoint['transitions']==checkpoint['optimizer_steps']==0
        assert checkpoint['initializer_provenance']['transitions']==655360000
        assert checkpoint['initializer_provenance']['optimizer_steps'] is None
        assert checkpoint['optimizer_resume_supported'] is False
        assert 'optimizer' not in checkpoint
        assert checkpoint['physics_contract']['actuator']['profile']==ACTUATOR_PROFILE
        assert checkpoint['observation']['preview_horizon_s']==horizon
