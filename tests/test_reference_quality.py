"""Contact audit must see sliding even when pose and derivative clocks agree."""
import numpy as np
import pytest


def test_near_floor_contact_audit_detects_consistent_but_sliding_translation():
    from k1_motion.robot import K1Model
    from k1_motion.contracts import MotionClip
    from k1_motion.reference_quality import audit_reference_consistency
    from k1_motion.reference_velocity_clock import playback_derivatives
    robot=K1Model()
    clip=MotionClip.from_references([robot.neutral_reference(i*.02) for i in range(11)],
        {'model_signature':robot.signature})
    quiet=audit_reference_consistency(clip,robot)
    assert quiet['near_floor_samples']>0
    assert quiet['near_floor_tangent_mean_m_s'] == pytest.approx(0.,abs=1e-9)
    clip.values.update({key:value.copy() for key,value in clip.values.items()})
    offset=clip.times
    clip.values['root_position'][:,0]+=offset
    clip.values['landmarks'][:,:,0]+=offset[:,None]
    joint,root=playback_derivatives(clip)
    clip.values.update(joint_velocity=joint,root_velocity=root)
    sliding=audit_reference_consistency(clip,robot)
    assert sliding['playback_velocity_max_error']<1e-8
    assert sliding['landmark_fk_max_error_m']<1e-8
    assert sliding['near_floor_tangent_mean_m_s']>.8
    assert not sliding['contact_consistent']
    for field in ('root_position','landmarks'):
        clip.values[field][...,2]+=.0005
    shifted=audit_reference_consistency(clip,robot)
    assert shifted['near_floor_samples']==sliding['near_floor_samples']
    assert np.isfinite(shifted['near_floor_tangent_p95_m_s'])
