"""New actuator dynamics may not drift under exact resume or deployment.

Uses native PPO/save/export/runtime. One changed effective torque constant must
fail even though the named action profile and nominal model signature match.
"""
import copy
import json

import pytest
import torch

from k1_motion.actuators import ACTUATOR_PROFILE
from k1_motion.export import export_checkpoint
from k1_motion.learning import Policy, TrainConfig, train
from k1_motion.runtime import Controller
from k1_motion.tracking_env import TrackerEnv
from test_training import make_library


def test_new_actuator_exact_resume_and_export_reject_effective_parameter_drift(tmp_path):
    robot, library = make_library(tmp_path)
    torch.set_num_threads(1)
    settings = {'actuator_profile': ACTUATOR_PROFILE,
                'command_velocity_limit': (.8*robot.velocity_limit).tolist()}
    env = TrackerEnv(library, 2, 'cpu', 'mujoco_cpp', action_settings=settings,
                     physics_options={'workers': 2})
    config = TrainConfig(stage='student', iterations=1, horizon=4, epochs=1, minibatch=8,
                         hidden_sizes=(32, 16), evaluation_interval=0, bc_weight=0)
    try:
        initial = train(env, tmp_path/'initial', config)
        path = tmp_path/'initial/checkpoint.pt'
        exact = train(env, tmp_path/'exact', config, resume_checkpoint=path)
        assert exact['optimizer_steps'] > initial['optimizer_steps']
        export_checkpoint(path, tmp_path/'export/actor.pt')
        policy = Policy(tmp_path/'export/actor.pt', robot.signature)
        assert policy.metadata['actuator_contract'] == env.physics.contract['actuator']
        Controller(robot, policy)
        env.physics.contract = copy.deepcopy(env.physics.contract)
        env.physics.contract['actuator']['peak_effort_nm'][0] += .01
        with pytest.raises(ValueError, match='actuator'):
            train(env, tmp_path/'drifted', config, resume_checkpoint=path)
        metadata_path = tmp_path/'export/actor.json'
        metadata = json.loads(metadata_path.read_text())
        metadata['actuator_contract']['peak_effort_nm'][0] += .01
        metadata_path.write_text(json.dumps(metadata))
        policy = Policy(tmp_path/'export/actor.pt', robot.signature)
        with pytest.raises(ValueError, match='actuator'):
            Controller(robot, policy)
        saved = torch.load(path, weights_only=True)
        saved.pop('physics_contract')
        torch.save(saved, tmp_path/'missing-contract.pt')
        with pytest.raises(ValueError, match='actuator'):
            export_checkpoint(tmp_path/'missing-contract.pt', tmp_path/'invalid/actor.pt')
    finally:
        env.close()
