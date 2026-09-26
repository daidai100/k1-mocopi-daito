"""An offline scaled target must retain K1 geometry and clock contracts."""

import numpy as np
import pytest

from k1_motion.calibration import Calibration
from k1_motion.contracts import HumanFrame, MotionClip
from k1_motion.robot import K1Model
from k1_motion.streaming import retarget_at_control_rate
from k1_motion.recovery_validation import retarget_recovery


def motion(robot):
    times = np.arange(6) * robot.control_dt
    positions = np.tile(robot.neutral_landmarks * 1.8, (len(times), 1, 1))
    positions[:, :, 0] += np.arange(len(times))[:, None] * 0.035
    positions[:, :, 2] += np.arange(len(times))[:, None] * 0.012
    orientations = np.tile([1., 0., 0., 0.], (len(times), 17, 1))
    return {"times": times, "positions": positions, "orientations": orientations}


def test_robot_fit_scales_travel_and_rise_after_neutral_body_calibration():
    robot = K1Model()
    human = motion(robot)
    frames = [HumanFrame(float(t), float(t), i, human["positions"][i],
                         human["orientations"][i], "fixture") for i, t in enumerate(human["times"])]
    base = Calibration.from_neutral(frames[0], robot)
    fit = Calibration.from_neutral(frames[0], robot, motion_profile="robot-fit-70-v1")
    first = fit.apply(frames[0]).positions
    last = fit.apply(frames[-1]).positions
    original = base.apply(frames[-1]).positions
    np.testing.assert_allclose(first, base.apply(frames[0]).positions)
    np.testing.assert_allclose(last[0, :2] - first[0, :2],
                               .7 * (original[0, :2] - first[0, :2]))
    np.testing.assert_allclose(last[0, 2] - first[0, 2], .7 * (original[0, 2] - first[0, 2]))
    np.testing.assert_allclose(last - last[0], original - original[0])
    assert fit.metadata()["motion_scale"]["profile"] == "robot-fit-70-v1"
    with pytest.raises(ValueError, match="motion scale profile"):
        Calibration.from_neutral(frames[0], robot, motion_profile="unknown")


@pytest.mark.parametrize("convert", [retarget_at_control_rate, retarget_recovery])
def test_scaled_reference_roundtrips_and_preserves_control_clock(tmp_path, convert):
    robot = K1Model()
    human = motion(robot)
    metadata = {"source_motion_id": "fixture/scale"}
    default, _ = convert(robot, human, metadata)
    scaled, _ = convert(robot, human, metadata, motion_profile="robot-fit-70-v1")
    prefix, _ = convert(robot, {k: v[:4] for k, v in human.items()}, metadata,
                        motion_profile="robot-fit-70-v1")
    scaled.save(tmp_path / "scaled.npz")
    loaded = MotionClip.load(tmp_path / "scaled.npz")
    assert scaled.metadata["retarget_version"] != default.metadata["retarget_version"]
    assert scaled.metadata["motion_scale"]["profile"] == "robot-fit-70-v1"
    assert scaled.metadata["experimental_reference"] is True
    assert scaled.metadata["training_eligible"] is False
    for key in scaled.values:
        np.testing.assert_array_equal(scaled.values[key], loaded.values[key])
        np.testing.assert_array_equal(scaled.values[key][:len(prefix.times)], prefix.values[key])
    assert scaled.times[-1] > default.times[-1]
    assert scaled.metadata["motion_scale"]["clock_scale"] == 1.25
    np.testing.assert_allclose(scaled.source_times, scaled.times / 1.25)
    assert np.all(scaled.received_times <= scaled.times + 1e-12)
    np.testing.assert_allclose(scaled.values["root_velocity"][1:, :3],
                               np.diff(scaled.values["root_position"], axis=0) / robot.control_dt)
    joint_speed = np.abs(np.diff(scaled.values["joint_position"], axis=0) / robot.control_dt)
    assert joint_speed.max() <= robot.config["command_velocity_limit"] + 1e-6
