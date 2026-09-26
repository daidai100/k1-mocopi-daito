"""World lag, orientation nullspaces and substep safety must survive PPO/export.

Failure modes: independent root centering hides lag; distant marker extinguishes
whole-body learning; posture is unobserved by point positions; command clipping
is mistaken for measured speed safety; reward changes are silently resumed.
"""
from dataclasses import replace

import numpy as np
import pytest
import torch

from k1_motion.world_objective import WorldBodyTracking, SafetyObjective
from test_reward_series import sample
from test_training import make_library


def test_world_tracking_penalizes_root_lag_and_preserves_shared_translation():
    state, reference, _ = sample()
    reward = WorldBodyTracking()
    maximum, _ = reward.step(state, reference, reference['landmark_velocity'])
    torch.testing.assert_close(maximum, torch.full((2,), 9.5))
    state['landmarks'][:, :, 0] += .15
    state['position'][:, 0] += .15
    lagged, parts = reward.step(state, reference, reference['landmark_velocity'])
    torch.testing.assert_close(lagged, torch.full((2,), 5.))
    torch.testing.assert_close(parts['world_body_rmse_m'], torch.full((2,), .15))
    for obj, key in ((state,'position'), (reference,'root_position')):
        obj[key] += torch.tensor([3.,-5.,.2])
        obj['landmarks'] += torch.tensor([3.,-5.,.2])
    translated, _ = reward.step(state, reference, reference['landmark_velocity'])
    torch.testing.assert_close(translated, lagged)


def test_one_bad_marker_does_not_extinguish_other_markers_or_posture_signal():
    state, reference, _ = sample()
    reward = WorldBodyTracking()
    state['landmarks'][:, 5, 0] += 2.
    before, parts = reward.step(state, reference, reference['landmark_velocity'])
    assert (parts['world_position'] > 16/17).all()
    # Head yaw at neutral is a verified positional nullspace in the K1 model.
    state['q'][:, 0] += .3
    after, changed = reward.step(state, reference, reference['landmark_velocity'])
    assert (after < before).all()
    torch.testing.assert_close(changed['world_position'], parts['world_position'])


def test_safety_uses_measured_substeps_without_penalizing_legal_speed():
    objective = SafetyObjective()
    metrics = {k:torch.zeros(2) for k in ('operating_speed_fraction','joint_limit_fraction')}
    cost, _ = objective.step(torch.zeros(2), metrics)
    torch.testing.assert_close(cost, torch.zeros(2))
    # End-state speed can be legal although a substep exceeded its limit.
    metrics['operating_speed_fraction'][1] = .1
    cost, parts = objective.step(torch.tensor([1.,0.]), metrics)
    torch.testing.assert_close(cost, torch.tensor([4.,.4]))
    torch.testing.assert_close(parts['safety/operating_speed_fraction'], torch.tensor([0.,.1]))
    with pytest.raises(ValueError, match='substep'):
        objective.step(torch.zeros(2), {})


def test_native_world_safety_ppo_export_and_reward_resume_contract(tmp_path):
    from k1_motion.tracking_env import TrackerEnv
    from k1_motion.learning import TrainConfig, train, Policy
    from k1_motion.export import export_checkpoint
    from k1_motion.contracts import MotionClip
    from k1_motion.control_validation import replay_clip
    robot, library = make_library(tmp_path)
    torch.set_num_threads(1)
    config = TrainConfig(stage='student',iterations=2,horizon=8,epochs=1,minibatch=16,
        hidden_sizes=(32,16),evaluation_interval=0,checkpoint_interval=1,bc_weight=0)
    settings = dict(history=3,observation_profile='planar',reward_profile='world-body-v1',
        safety_profile='casual-safe-v1',physics_options={'workers':2},
        action_settings={'actuator_profile':'booster-train-k1-actuator-v1',
                         'command_velocity_limit':(.8*robot.velocity_limit).tolist()})
    env=TrackerEnv(library,2,'cpu','mujoco_cpp',**settings)
    try:
        result=train(env,tmp_path/'training',config)
        assert result['finite_updates'] and result['checkpoint_reload_max_error']==0
        assert result['reward_settings']['safety']['version']=='casual-safe-v1'
        export_checkpoint(tmp_path/'training/checkpoint.pt',tmp_path/'export/actor.pt')
        replay=replay_clip(robot,Policy(tmp_path/'export/actor.pt',robot.signature),
                           MotionClip.load(library/'standing.npz'))
        assert np.isfinite(replay['world_body_rmse_m'])
        assert replay['actuator_safety']['sample_period_s']==.002
        assert replay['resets_during_trial']==0
        other=TrackerEnv(library,2,'cpu','mujoco_cpp',**{**settings,'reward_profile':'simple-track-v2'})
        try:
            with pytest.raises(ValueError,match='reward settings'):
                train(other,tmp_path/'invalid',replace(config,iterations=1),
                      resume_checkpoint=tmp_path/'training/checkpoint.pt')
        finally:
            other.close()
    finally:
        env.close()
