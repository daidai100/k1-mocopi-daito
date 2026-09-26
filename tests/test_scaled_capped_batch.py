"""Risks: scaled standing height, stale derivatives, crossed clip boundaries,
mutated cache, uncapped outliers, weak global XY tracking, lost checkpoint contract.
Reproduce: PYTHONPATH=src .venv/bin/python -m pytest tests/test_scaled_capped_batch.py
"""
import torch
from test_training import make_library
from test_reward_series import sample


def test_scale_ground_jump_derivatives_and_source_preservation(tmp_path):
    from k1_motion.learning import MotionLibrary
    from k1_motion.reference_scale import scale_library, foot_bottom
    robot, directory = make_library(tmp_path)
    lib = MotionLibrary(directory, robot, 'cpu', storage='packed')
    lib.rows = [dict(row, family='jump') for row in lib.rows]
    original = {k: v.clone() for k,v in lib.values.items()}
    # Translate complete K1 poses, including a genuine airborne interval.
    for offset, length in zip(lib.offsets.tolist(), lib.lengths.tolist()):
        travel = torch.arange(length)*.01
        lift = torch.zeros(length); lift[length//3:2*length//3] = .2
        lib.values['root_position'][offset:offset+length,0] += travel
        lib.values['landmarks'][offset:offset+length,:,0] += travel[:,None]
        lib.values['root_position'][offset:offset+length,2] += lift
        lib.values['landmarks'][offset:offset+length,:,2] += lift[:,None]
    before = {k:v.clone() for k,v in lib.values.items()}
    bottom = foot_bottom(before, robot)
    scale_library(lib, robot, .9)
    torch.testing.assert_close(foot_bottom(lib.values, robot), bottom-.1*bottom.clamp(min=0), atol=2e-7, rtol=1e-5)
    torch.testing.assert_close(lib.values['joint_position'],original['joint_position'])
    torch.testing.assert_close(lib.values['landmarks']-lib.values['root_position'][:,None], before['landmarks']-before['root_position'][:,None])
    for offset,length in zip(lib.offsets.tolist(),lib.lengths.tolist()):
        p=lib.values['root_position'][offset:offset+length]
        torch.testing.assert_close(p[0,:2],before['root_position'][offset,:2])
        torch.testing.assert_close(lib.values['root_velocity'][offset+1:offset+length,:3],torch.diff(p,dim=0)/robot.control_dt)
        assert not lib.values['root_velocity'][offset,:3].any()
    assert lib.reference_scale_contract['scale']==.9


def test_capped_body_and_independent_root_xy():
    from k1_motion.world_objective import WorldBodyTracking
    s,r,_=sample(); s['omega']=torch.zeros(2,3)
    reward=WorldBodyTracking('world-capped-root-v1')
    _, a=reward.step(s,r,None)
    s['landmarks'][:,0,0]+=1
    _, b=reward.step(s,r,None)
    s['landmarks'][:,0,0]+=100
    _, c=reward.step(s,r,None)
    torch.testing.assert_close(b['body_relative'], c['body_relative'])
    assert reward.contract['weights']['root_xy_position']>reward.contract['weights']['body_relative']
    s['position'][:,0]+=.5
    _,d=reward.step(s,r,None)
    assert (d['root_xy_position']<a['root_xy_position']).all()


def test_native_scaled_capped_training_checkpoint(tmp_path):
    from k1_motion.tracking_env import TrackerEnv
    from k1_motion.learning import TrainConfig,train
    robot, directory=make_library(tmp_path)
    torch.set_num_threads(1)
    env=TrackerEnv(directory,2,'cpu','mujoco_cpp',history=3,observation_profile='preview',
        reward_profile='world-capped-root-v1',reference_scale=.9, reference_storage='packed', physics_options={'workers':2})
    try:
        report=train(env,tmp_path/'training',TrainConfig(stage='student',iterations=2,horizon=8,
            epochs=1,minibatch=16,hidden_sizes=(32,16),evaluation_interval=0,checkpoint_interval=1,bc_weight=0))
        assert report['finite_updates'] and report['checkpoint_reload_max_error']==0
        assert report['reward_settings']['reference_scale']['scale']==.9
    finally:
        env.close()
