"""Versioned trajectory screening must expose misses hidden by endpoint progress.

Risks: lateral endpoint error, closed-loop/out-back omission, prefix-only reward
for early falls, off-by-one trace/reference alignment, changing historical gates,
and a failed old clean gate being promoted by the new diagnostics.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from k1_motion.control_validation import motion_fidelity
from k1_motion.trajectory_metrics import trajectory_fidelity


def recording(target, actual=None, dt=.02):
    target = np.asarray(target, dtype=float)
    actual = target if actual is None else np.asarray(actual, dtype=float)
    orientation = np.tile([1., 0, 0, 0], (len(target), 1))
    velocity = np.vstack((np.zeros((1, 3)), np.diff(target, axis=0)/dt))
    clip = SimpleNamespace(times=np.arange(len(target))*dt,
                           values={'root_position': target, 'root_orientation': orientation,
                                   'root_velocity': velocity})
    qpos = np.column_stack((actual[1:], orientation[1:len(actual)], np.zeros((len(actual)-1, 22))))
    qvel = np.column_stack((np.diff(actual, axis=0)/dt, np.zeros((len(actual)-1, 25))))
    return clip, qpos, qvel


def test_lateral_miss_rejected_without_changing_old_fidelity():
    time = np.arange(0, 20.0001, .02)
    target = np.column_stack((.1*time, np.zeros_like(time), np.full_like(time, .5)))
    actual = target.copy()
    actual[:, 1] = .1*time
    clip, qpos, qvel = recording(target, actual)
    old = motion_fidelity(clip, qpos, qvel)
    assert old['motion_fidelity_passed'] and old['root_progress_ratio'] == pytest.approx(1.)
    result = trajectory_fidelity(clip, qpos, completed=True, old_clean_success=True)
    assert result['screening_only'] and not result['clean_v3']
    assert result['root_xy_p95_m'] > 1.8
    assert result['reference_path_length_m'] == pytest.approx(2.)
    assert result['actual_path_length_m'] == pytest.approx(np.sqrt(8))
    assert motion_fidelity(clip, qpos, qvel) == old


@pytest.mark.parametrize('shape', ['loop', 'out_back'])
def test_no_net_displacement_does_not_hide_missing_travel(shape):
    time = np.arange(0, 20.0001, .02)
    if shape == 'loop':
        target = np.column_stack((.3*np.cos(2*np.pi*time/20), .3*np.sin(2*np.pi*time/20), np.full_like(time, .5)))
    else:
        target = np.column_stack((.1*np.minimum(time, 20-time), np.zeros_like(time), np.full_like(time, .5)))
    clip, qpos, qvel = recording(target, np.repeat(target[:1], len(time), axis=0))
    old = motion_fidelity(clip, qpos, qvel)
    assert old['motion_fidelity_passed'] and old['root_progress_ratio'] is None
    result = trajectory_fidelity(clip, qpos, completed=True, old_clean_success=True)
    assert not result['clean_v3']
    assert result['reference_path_length_m'] > 1.8
    assert result['actual_path_length_m'] == 0
    assert result['path_length_ratio'] == 0


def test_early_fall_uses_full_intended_duration_and_aligned_reference_ticks():
    target = np.column_stack((np.arange(101)*.02, np.zeros(101), np.full(101, .5)))
    clip, full, _ = recording(target)
    prefix = trajectory_fidelity(clip, full[:10], completed=False, old_clean_success=False)
    complete = trajectory_fidelity(clip, full, completed=True, old_clean_success=True)
    assert prefix['root_xy_rmse_m'] == 0
    assert prefix['root_xy_p95_m'] == 0
    assert prefix['survived_fraction'] == pytest.approx(.1)
    assert prefix['full_reference_duration_xy_score'] == pytest.approx(.1)
    assert prefix['executed_duration_s'] == pytest.approx(.2)
    assert prefix['intended_duration_s'] == pytest.approx(2.)
    assert prefix['reference_path_length_m'] == pytest.approx(2.)
    assert prefix['executed_reference_path_length_m'] == pytest.approx(.2)
    assert not prefix['clean_v3']
    assert complete['full_reference_duration_xy_score'] == pytest.approx(1.)
    assert complete['clean_v3']
    # A caller cannot certify a short prefix by accidentally marking it completed.
    assert not trajectory_fidelity(clip, full[:10], completed=True, old_clean_success=True)['clean_v3']
    assert not trajectory_fidelity(clip, full, completed=True, old_clean_success=False)['clean_v3']


def test_native_replay_adds_nested_screening_and_preserves_old_clean_definition(tmp_path):
    from test_training import make_library
    from k1_motion.contracts import MotionClip
    from k1_motion.control_validation import replay_clip
    from k1_motion.runtime import ZeroResidual
    robot, library = make_library(tmp_path)
    clip = MotionClip.load(library/'standing.npz')
    result = replay_clip(robot, ZeroResidual(), clip)
    assert result['clean_success'] == (result['pose_balance_collision_passed'] and result['motion_fidelity_passed'])
    diagnostics = result['trajectory_v3']
    assert diagnostics['version'] == 'k1-trajectory-fidelity-v3-screening'
    assert diagnostics['screening_only']
    assert diagnostics['executed_intervals'] == len(clip.times)-1
    assert diagnostics['clean_v3'] <= result['clean_success']
