"""A 30 Hz source hold must not permit a two-tick jump in a 50 Hz command."""
import numpy as np

from k1_motion.recovery_validation import retarget_recovery
from k1_motion.reference_velocity_clock import velocity_errors
from k1_motion.robot import K1Model


def test_versioned_slow_source_recovery_obeys_each_command_tick_and_is_causal(tmp_path):
    from k1_motion.contracts import MotionClip
    robot=K1Model()
    times=np.arange(12)/30
    positions=np.tile(robot.neutral_landmarks*1.6,(len(times),1,1))
    positions[:,5,0]+=np.tile([0.,.3,-.3],4)
    human=dict(times=times,positions=positions,
        orientations=np.tile([1.,0.,0.,0.],(len(times),17,1)))
    legacy,_=retarget_recovery(robot,human,{'source_motion_id':'slow_source'})
    old_speed=np.abs(np.diff(legacy.values['joint_position'],axis=0)/.02).max()
    assert old_speed>6.01  # This fixture reproduces the actual source-hold defect.
    fixed,_=retarget_recovery(robot,human,{'source_motion_id':'slow_source'},control_tick_hold=True)
    fixed.save(tmp_path/'motion.npz')
    reloaded=MotionClip.load(tmp_path/'motion.npz')
    assert np.abs(np.diff(reloaded.values['joint_position'],axis=0)/.02).max()<=6.+1e-6
    assert max(velocity_errors(reloaded).values())<=1e-8
    held=np.r_[False,np.diff(reloaded.source_times)==0]
    assert held.any() and np.all(reloaded.values['joint_velocity'][held]==0)
    assert np.all(reloaded.source_times<=reloaded.times+1e-12)
    assert 'control-tick-hold' in reloaded.metadata['retarget_version']
    prefix={k:v[:6] for k,v in human.items()}
    short,_=retarget_recovery(K1Model(),prefix,{'source_motion_id':'slow_source'},control_tick_hold=True)
    np.testing.assert_array_equal(short.values['joint_position'],fixed.values['joint_position'][:len(short.times)])
