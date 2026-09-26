"""Opt-in retarget speed contract must not change the pinned physical model."""
from dataclasses import replace

import numpy as np
import pytest

from k1_motion.contracts import HumanFrame, MotionClip
from k1_motion.low_pose import retarget_low_pose
from k1_motion.recovery_validation import RECOVERY_GATES, audit_recovery, retarget_recovery
from k1_motion.retarget_speed import retarget_speed_contract
from k1_motion.robot import K1Model
from k1_motion.streaming import retarget_at_control_rate


def moving_fixture(robot):
    times = np.arange(8) * .02
    positions = np.tile(robot.neutral_landmarks * 1.6, (len(times), 1, 1))
    positions[:, 5, 0] += np.array([0, .15, -.15, .15, -.15, .15, -.15, .15])
    return dict(times=times, positions=positions,
                orientations=np.tile([1., 0., 0., 0.], (len(times), 17, 1)))


@pytest.mark.parametrize("convert", [retarget_at_control_rate, retarget_recovery])
def test_official80_retarget_roundtrip_causality_and_legacy_default(tmp_path, convert):
    robot = K1Model()
    signature, nominal, config = robot.signature, robot.velocity_limit.copy(), dict(robot.config)
    human = moving_fixture(robot)
    metadata = {"source_motion_id": "speed-fixture"}
    default, _ = convert(robot, human, metadata)
    explicit, _ = convert(robot, human, metadata, speed_profile="legacy-command-v1")
    candidate, _ = convert(robot, human, metadata, speed_profile="official-80-v1")
    prefix, _ = convert(robot, {k: v[:4] for k, v in human.items()}, metadata,
                        speed_profile="official-80-v1")
    candidate.save(tmp_path / "official80.npz")
    loaded = MotionClip.load(tmp_path / "official80.npz")
    for key in candidate.values:
        np.testing.assert_array_equal(default.values[key], explicit.values[key])
        np.testing.assert_array_equal(candidate.values[key], loaded.values[key])
        np.testing.assert_array_equal(candidate.values[key][:len(prefix.times)], prefix.values[key])
    np.testing.assert_array_equal(candidate.times, default.times)
    np.testing.assert_array_equal(candidate.source_times, default.source_times)
    np.testing.assert_array_equal(candidate.received_times, default.received_times)
    speed = np.abs(np.diff(candidate.values["joint_position"], axis=0) / .02)
    assert np.all(speed <= .8*nominal+1e-6)
    assert speed.max() > 6.1  # Exercise the new bound, not only a stationary fixture.
    assert candidate.metadata['retarget_speed_contract']['profile'] == 'official-80-v1'
    assert candidate.metadata['retarget_version'] != default.metadata['retarget_version']
    assert 'retarget_speed_contract' not in default.metadata
    assert robot.signature == signature and robot.config == config
    np.testing.assert_array_equal(robot.velocity_limit, nominal)
    assert loaded.metadata['model_signature'] == signature


def test_speed_contract_and_low_pose_are_explicit_without_changing_defaults():
    robot = K1Model()
    contract = retarget_speed_contract(robot, "official-80-v1")
    np.testing.assert_allclose(contract['joint_velocity_limits_rad_s'], .8*robot.velocity_limit)
    with pytest.raises(ValueError, match="speed profile"):
        retarget_speed_contract(robot, "unregistered-faster")
    human = moving_fixture(robot)
    frames = [HumanFrame(float(t), float(t), i, human['positions'][i], human['orientations'][i], "fixture")
              for i, t in enumerate(human['times'][:3])]
    clip, _ = retarget_low_pose(robot, frames, {}, speed_profile="official-80-v1")
    assert clip.metadata['retarget_speed_contract'] == contract
    speed = np.abs(np.diff(clip.values['joint_position'], axis=0)/.02)
    assert np.all(speed <= .8*robot.velocity_limit+1e-6)


def test_new_speed_audit_keeps_geometry_gates_and_legacy_speed_audit():
    robot = K1Model()
    refs = [replace(robot.neutral_reference(i*.02), root_position=np.array([0.,0.,1.]),
                    joint_position=robot.neutral+np.eye(22)[2]*.16*i) for i in range(2)]
    clip = MotionClip.from_references(refs, {"model_signature": robot.signature})
    args = (clip, clip, [{"rms_landmark_error_m": 0.}], {"rms_landmark_error_m": 0.})
    legacy = audit_recovery(*args)
    candidate = audit_recovery(*args, speed_profile="official-80-v1")
    assert 'control_clock_velocity_limit' in legacy['rejection_reasons']
    assert 'control_clock_velocity_limit' not in candidate['rejection_reasons']
    assert candidate['gates'] == legacy['gates'] == RECOVERY_GATES
    assert candidate['samples'] == legacy['samples']
    assert candidate['audit_hz'] == 500.0
    # A faster speed profile never admits a geometrically invalid reference.
    bad = MotionClip.from_references([replace(r, root_position=np.array([0.,0.,.1])) for r in refs],
                                    {"model_signature": robot.signature})
    result = audit_recovery(bad, bad, args[2], args[3], speed_profile="official-80-v1")
    assert not result['accepted']
    assert 'ground_penetration_on_command_path' in result['rejection_reasons']
