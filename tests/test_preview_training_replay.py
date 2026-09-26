"""Native PPO -> exported buffered runtime with matching observation clocks.

Failures: unbounded preview leakage, mismatched input width, missing tail drain,
fake timestamp freshness, and horizon-changing optimizer resume.
"""
from dataclasses import replace

import numpy as np
import pytest
import torch

from k1_motion.tracking_env import TrackerEnv
from k1_motion.learning import TrainConfig, train, Policy
from k1_motion.export import export_checkpoint
from k1_motion.contracts import MotionClip
from k1_motion.control_validation import replay_clip
from test_training import make_library


@pytest.mark.parametrize('horizon',[0.,.3])
def test_preview_native_training_export_buffered_replay_and_resume(tmp_path,horizon):
    robot, library=make_library(tmp_path)
    torch.set_num_threads(1)
    settings=dict(history=3,observation_profile='preview',preview_horizon_s=horizon,
                  reward_profile='world-body-v1',physics_options={'workers':2})
    env=TrackerEnv(library,2,'cpu','mujoco_cpp',**settings)
    try:
        zeros=torch.zeros(2,dtype=torch.long)
        actor,_=env.reset(clips=zeros,frames=zeros)
        assert actor.shape==(2,env.observation['size'])
        assert actor.shape[1]==3*(env.observation['frame_size']+1)+120
        masks=actor[:,-120:].reshape(2,3,40)[:,:,-1]
        torch.testing.assert_close(masks,torch.full((2,3),float(horizon>0)))
        env.frames[:]=48
        env.command_frames[:]=48
        actor,_=env.observe()
        torch.testing.assert_close(actor[:,-120:],torch.zeros(2,120))
        config=TrainConfig(stage='student',iterations=2,horizon=8,epochs=1,minibatch=16,
            hidden_sizes=(32,16),evaluation_interval=0,checkpoint_interval=1,bc_weight=0)
        report=train(env,tmp_path/'training',config)
        assert report['finite_updates'] and report['checkpoint_reload_max_error']==0
        export_checkpoint(tmp_path/'training/checkpoint.pt',tmp_path/'export/actor.pt')
        policy=Policy(tmp_path/'export/actor.pt',robot.signature)
        result=replay_clip(robot,policy,MotionClip.load(library/'standing.npz'))
        assert result['resets_during_trial']==0
        assert result['preview']['playback_delay_s']==.3
        assert result['preview']['preview_horizon_s']==horizon
        assert not result['controller_stopped']
        assert np.isfinite(result['world_body_rmse_m'])
        other=TrackerEnv(library,2,'cpu','mujoco_cpp',**{**settings,'preview_horizon_s':.3-horizon})
        try:
            with pytest.raises(ValueError,match='observation'):
                train(other,tmp_path/'invalid_resume',replace(config,iterations=1),
                      resume_checkpoint=tmp_path/'training/checkpoint.pt')
        finally:
            other.close()
    finally:
        env.close()
