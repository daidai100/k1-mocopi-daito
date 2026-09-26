"""Padding failure modes, followed by a real saved-library/train/export replay.

Airborne feet, a COM outside actual ground support, degenerate toe support,
nonfoot support and collisions must never become standing holds. A zero first
causal derivative must not hide an immediately moving recording. Padding must
not change original poses, source time or duration-based membership, leak a
held-out recording, make stale derivative/contact labels, or omit long clips
whose endpoints cannot be padded. Repeated preparation must not overwrite data.
"""
from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from k1_motion.contracts import MotionClip
from k1_motion.robot import K1Model


def moving_clip(robot, key='motion', frames=511, airborne=False):
    refs=[]
    for i in range(frames):
        ref=robot.neutral_reference(i*.02)
        q=ref.joint_position.copy()
        q[0]=.08*np.sin(i*.02*2)
        position=ref.root_position.copy()
        position[2]+=.1 if airborne else 0
        ref=replace(ref,joint_position=q,root_position=position)
        robot.reset(ref)
        refs.append(replace(ref,landmarks=robot.landmarks()))
    meta=dict(id=key,capture_group='synthetic/'+key,family='gesture',dataset='synthetic',
        split='train',is_mirror=False,training_eligible=True,kinematics_accepted=True,
        model_signature=robot.signature,frames=frames,rejected_ticks=0,
        recovery_audit=dict(accepted=True,geometry_audited=True,audit_hz=500,rejection_reasons=[]))
    clip=MotionClip.from_references(refs,meta,sample_times=np.arange(frames)*.02)
    from k1_motion.reference_velocity_clock import playback_derivatives
    joint,root=playback_derivatives(clip)
    return MotionClip(clip.times,{**clip.values,'joint_velocity':joint,'root_velocity':root},
        clip.metadata,clip.source_times,clip.received_times)


def test_actual_ground_support_and_com_are_required():
    from k1_motion.standing_padding import endpoint_support, support_margin
    robot=K1Model()
    clip=moving_clip(robot)
    result=endpoint_support(clip,0,robot)
    assert result['feasible'] and result['com_margin_m']>0
    assert result['supporting_feet']==[True,True]
    air=endpoint_support(moving_clip(robot,airborne=True),0,robot)
    assert not air['feasible'] and 'no_ground_support_polygon' in air['reasons']
    square=np.array([[-1,-1],[1,-1],[1,1],[-1,1]])
    assert support_margin([0,0],square)==pytest.approx(1.)
    assert support_margin([2,0],square)<0
    assert support_margin([1,0],square)==pytest.approx(0.)
    assert support_margin([0,0],square[:2]) is None
    # Moving the root and feet together cannot fake a COM/support failure.
    shifted={k:v.copy() for k,v in clip.values.items()}
    shifted['root_position'][:,:2]+=[10,-5]
    shifted['landmarks'][:,:,:2]+=[10,-5]
    translated=MotionClip(clip.times,shifted,clip.metadata,clip.source_times,clip.received_times)
    assert endpoint_support(translated,0,robot)['com_margin_m']==pytest.approx(result['com_margin_m'])
    # An actual articulated forward lean puts mass ahead of the foot polygon.
    lean={k:v.copy() for k,v in clip.values.items()}
    from scipy.spatial.transform import Rotation
    quat=Rotation.from_euler('y',.55).as_quat()[[3,0,1,2]]
    lean['root_orientation'][:]=quat
    tilted=MotionClip(clip.times,lean,clip.metadata,clip.source_times,clip.received_times)
    assert not endpoint_support(tilted,0,robot)['feasible']


def test_saved_padding_preserves_source_and_has_zero_hold_velocities(tmp_path):
    from k1_motion.standing_padding import pad_clip
    from k1_motion.reference_velocity_clock import velocity_errors
    robot=K1Model()
    clip=moving_clip(robot)
    before={k:v.copy() for k,v in clip.values.items()}
    padded,audit=pad_clip(clip,robot)
    assert audit['leading_frames']==audit['trailing_frames']==15
    assert audit['original_duration_s']==pytest.approx(10.2)
    assert len(padded.times)==len(clip.times)+30
    assert padded.times[-1]-clip.times[-1]==pytest.approx(.6)
    for key,value in before.items():
        np.testing.assert_array_equal(clip.values[key],value)
        np.testing.assert_allclose(padded.values[key][15:-15],value,atol=1e-11,rtol=0)
    for key in ('joint_velocity','root_velocity'):
        assert not padded.values[key][:16].any()
        assert not padded.values[key][-15:].any()
    np.testing.assert_array_equal(padded.source_times[15:-15],clip.source_times)
    np.testing.assert_allclose(padded.received_times[15:-15],clip.received_times+.3)
    assert (padded.received_times<=padded.times+1e-12).all()
    assert max(velocity_errors(padded).values())<1e-8
    padded.save(tmp_path/'clip.npz')
    loaded=MotionClip.load(tmp_path/'clip.npz')
    assert loaded.metadata['standing_padding']==audit
    with pytest.raises(ValueError,match='already'):
        pad_clip(loaded,robot)
    still=MotionClip.from_references([robot.neutral_reference(i*.02) for i in range(30)],clip.metadata)
    _,quiet=pad_clip(still,robot)
    assert quiet['leading_frames']==quiet['trailing_frames']==0


def test_library_retains_all_long_originals_and_excludes_padding_from_cutoff(tmp_path):
    from k1_motion.standing_padding import prepare_library
    robot=K1Model()
    source=tmp_path/'source'
    source.mkdir()
    rows=[]
    for key,frames,air in [('long',511,False),('air',512,True),('exact10',501,False),('short',491,False)]:
        clip=moving_clip(robot,key,frames,air)
        clip.save(source/(key+'.npz'))
        rows.append({**clip.metadata,'reference_path':key+'.npz'})
    (source/'index.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    result=prepare_library(source,tmp_path/'prepared',workers=1)
    assert result['all_originals']==4 and result['training_originals']==2
    selected=[json.loads(line) for line in (tmp_path/'prepared/library/index.jsonl').read_text().splitlines()]
    assert {r['id'] for r in selected}=={'long','air'}
    air=next(r for r in selected if r['id']=='air')
    assert air['standing_padding']['leading_frames']==air['standing_padding']['trailing_frames']==0
    assert all(r['standing_padding']['original_duration_s']>10 for r in selected)
    with pytest.raises(FileExistsError):
        prepare_library(source,tmp_path/'prepared',workers=1)
    rows[0]['split']='test'
    (source/'index.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError,match='training'):
        prepare_library(source,tmp_path/'leaked',workers=1)


def test_native_padded_causal_reward_training_export_replay(tmp_path):
    from k1_motion.standing_padding import pad_clip
    from k1_motion.learning import MotionLibrary, TrainConfig, train, Policy
    from k1_motion.tracking_env import TrackerEnv
    from k1_motion.export import export_checkpoint
    from k1_motion.control_validation import replay_clip
    robot=K1Model()
    clip,audit=pad_clip(moving_clip(robot,frames=40),robot)
    directory=tmp_path/'library'
    clip.save(directory/'motion.npz')
    row={**clip.metadata,'reference_path':'motion.npz'}
    (directory/'index.jsonl').write_text(json.dumps(row)+'\n')
    torch.set_num_threads(1)
    library=MotionLibrary(directory,robot,'cpu',storage='packed')
    assert library.lengths.tolist()==[70]
    assert not library.values['landmark_velocity'][:16].any()
    env=TrackerEnv(directory,2,'cpu','mujoco_cpp',history=3,reference_storage='packed',
        library=library,reward_profile='causal-balanced-v1',safety_profile='casual-safe-v1',
        observation_profile='preview',physics_options={'workers':2})
    cfg=TrainConfig(stage='student',iterations=2,horizon=8,epochs=1,minibatch=16,
        hidden_sizes=(32,16),evaluation_interval=0,checkpoint_interval=1,bc_weight=0)
    try:
        result=train(env,tmp_path/'training',cfg)
        assert result['finite_updates'] and result['checkpoint_reload_max_error']==0
        assert result['last_metrics']['reference_exposure']['seen_originals']==1
        assert result['last_metrics']['reference_exposure']['total_originals']==1
        assert result['last_metrics']['reference_exposure']['total_transitions']==32
        assert 'hold_active_fraction' in result['last_metrics']['reward_components']
        export_checkpoint(tmp_path/'training/checkpoint.pt',tmp_path/'export/actor.pt')
        replay=replay_clip(robot,Policy(tmp_path/'export/actor.pt',robot.signature),clip)
        assert replay['resets_during_trial']==0
        continued=train(env,tmp_path/'resume',replace(cfg,iterations=1),
                        resume_checkpoint=tmp_path/'training/checkpoint.pt')
        assert continued['optimizer_steps']>result['optimizer_steps']
        assert continued['last_metrics']['reference_exposure']['total_transitions']==48
    finally:
        env.close()


def test_training_and_live_input_cannot_see_after_300ms(tmp_path):
    from k1_motion.preview import PreviewBuffer
    from k1_motion.observations import ObservationBuilder
    from k1_motion.tracking_env import TrackerEnv
    from test_training import make_library
    robot,directory=make_library(tmp_path)
    buffers=[PreviewBuffer(),PreviewBuffer()]
    for i in range(40):
        ref=robot.neutral_reference(i*.02)
        # Even preloaded future packets cannot enter a beyond-horizon slot.
        ref=replace(ref,received_time=min(i*.02,.3))
        buffers[0].push(ref)
        buffers[1].push(replace(ref,joint_position=ref.joint_position+(1. if i>15 else 0.)))
    observations=[]
    for buffer in buffers:
        window=buffer.sample(.3)
        observations.append(ObservationBuilder(robot.neutral,history=3,profile='preview').build(
            robot.state(.3),window.current,np.zeros(22),.3,preview=window))
    np.testing.assert_array_equal(*observations)
    env=TrackerEnv(directory,1,'cpu',history=3,observation_profile='preview')
    try:
        ids=torch.zeros(1,dtype=torch.long)
        before,_=env.reset(clips=ids,frames=ids)
        env.library.values['joint_position'][:,16:]+=1.
        after,_=env.reset(clips=ids,frames=ids)
        torch.testing.assert_close(before,after,atol=0,rtol=0)
        env.library.values['joint_position'][:,15]+=1.
        at_boundary,_=env.reset(clips=ids,frames=ids)
        assert not torch.equal(before,at_boundary)
    finally:
        env.close()
