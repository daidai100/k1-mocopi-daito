"""Task reward must prefer moving with a moving target and quiet static holds.

Failure cases: global drift hidden by local pose, velocity reward extinguished
by drift, blanket stillness bonus, stance slip mistaken for swing motion,
floating feet earning stance reward, unbounded negative survival incentives,
silent changes to old checkpoint rewards, and missing support diagnostics.
"""
import pytest
import torch

from test_reward_series import sample


def inputs():
    state,reference,_=sample()
    state['velocity']=torch.zeros(2,3)
    state['omega']=torch.zeros(2,3)
    state['dq']=torch.zeros(2,22)
    reference['root_velocity']=torch.zeros(2,6)
    reference['joint_velocity']=torch.zeros(2,22)
    reference['contacts']=torch.ones(2,2)
    reference['contact_confidence']=torch.ones(2,2)
    reference['landmark_velocity']=torch.zeros(2,17,3)
    support=dict(contact=torch.ones(2,2,dtype=torch.bool),slip_speed=torch.zeros(2,2))
    return state,reference,support


def test_tracking_hold_slip_and_floating_counterexamples():
    from k1_motion.world_objective import WorldBodyTracking
    reward=WorldBodyTracking('causal-balanced-v1')
    state,reference,support=inputs()
    maximum,parts=reward.step(state,reference,reference['landmark_velocity'],support)
    assert torch.allclose(maximum,torch.full((2,),reward.contract['maximum_tracking_per_second']))
    state['dq'][1]=2
    state['velocity'][1,0]=.3
    noisy,p=reward.step(state,reference,reference['landmark_velocity'],support)
    assert noisy[0]>noisy[1] and p['hold_active_fraction'].min()==1
    state,reference,support=inputs()
    reference['root_velocity'][:,0]=.5
    reference['landmark_velocity'][:,:,0]=.5
    state['velocity'][0,0]=.5
    speed=reference['landmark_velocity'].clone()
    speed[1]=0
    moving,p=reward.step(state,reference,speed,support)
    assert moving[0]>moving[1] and not p['hold_active_fraction'].any()
    state,reference,support=inputs()
    support['slip_speed'][1]=.3
    slip,p=reward.step(state,reference,reference['landmark_velocity'],support)
    assert slip[0]>slip[1]
    support['contact'][1]=False
    floating,p=reward.step(state,reference,reference['landmark_velocity'],support)
    assert p['stance_still'][1]==0 and floating[0]>floating[1]
    with pytest.raises(ValueError,match='support'):
        reward.step(state,reference,reference['landmark_velocity'])


def test_world_and_velocity_remain_independent_bounded_and_old_profile_unchanged():
    from k1_motion.world_objective import WorldBodyTracking
    reward=WorldBodyTracking('causal-balanced-v1')
    state,reference,support=inputs()
    _,before=reward.step(state,reference,reference['landmark_velocity'],support)
    state['position'][1,0]+=1
    state['landmarks'][1,:,0]+=1
    value,after=reward.step(state,reference,reference['landmark_velocity'],support)
    assert value[0]>value[1] and after['world_position'][1]<.05
    torch.testing.assert_close(after['root_velocity'],before['root_velocity'])
    assert value.min()>=0 and value.max()<=reward.contract['maximum_tracking_per_second']
    assert WorldBodyTracking('world-velocity-v1').contract['maximum_tracking_per_second']==11.5
