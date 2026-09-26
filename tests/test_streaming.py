import json

import numpy as np
import pytest
import torch

from k1_motion.contracts import MotionClip
from k1_motion.corpus import family_of
from k1_motion.learning import MotionLibrary
from k1_motion.robot import K1Model
from k1_motion.streaming import control_schedule, retarget_at_control_rate


def human_motion(robot, times):
    positions = np.tile(robot.neutral_landmarks * 1.8, (len(times), 1, 1))
    positions[:, 5, 0] += 0.02 * np.sin(np.arange(len(times)) * 0.2)
    return {
        "times": times,
        "positions": positions,
        "orientations": np.tile([1.0, 0, 0, 0], (len(times), 17, 1)),
    }


def test_floor_getups_are_not_selected_as_flat_ground_stances():
    for annotation in (
        "A person kneeling on the floor and then standing up using the right hand",
        "The lays down on the floor and stands up again.",
        "A person makes a handstand.",
        "A person performs a hand stand.",
        "A person performs a headstand and a backflip.",
    ):
        assert family_of({"annotations": [annotation]}) == "external_support_or_recovery"
    assert family_of({"annotations": ["A person stands still."]}) == "stance"
    assert family_of({"annotations": ["A person stands upright and raises the arms."]}) == "reach"
    assert family_of({"annotations": ["A person with bent knees stands up and reaches forward."]}) == "reach"


def test_latest_frame_clock_handles_reordering_loss_and_future_roundoff():
    times = np.arange(6) * 0.01
    arrivals = np.array([0, 0.05, 0.02 + 1e-12, 0.02, 0.04, 0.05])
    dropped = np.array([False, False, False, False, True, False])
    ticks, indices = control_schedule(times, 0.01, arrivals, dropped)
    np.testing.assert_array_equal(indices, [0, 0, 3, 3, 3, 5])
    assert np.all(arrivals[indices] <= ticks)
    # A source frame that arrives even slightly after a tick is not fresh yet.
    _, indices = control_schedule([0, 0.02, 0.04], 0.02, [0, 0.02 + 1e-12, 0.04])
    np.testing.assert_array_equal(indices, [0, 0, 2])


def test_skipped_native_frames_cannot_influence_control_references():
    robot = K1Model()
    human = human_motion(robot, 7 + np.arange(61) / 120)
    _, indices = control_schedule(human["times"], robot.control_dt)
    metadata = {"source_motion_id": "synthetic/high-rate"}
    original, _ = retarget_at_control_rate(robot, human, metadata)
    unseen = np.ones(len(human["times"]), bool)
    unseen[indices] = False
    human["positions"][unseen, 5, 0] += 0.2
    changed, _ = retarget_at_control_rate(robot, human, metadata)
    for key in original.values:
        np.testing.assert_array_equal(original.values[key], changed.values[key])
    np.testing.assert_array_equal(original.source_times, human["times"][indices])
    assert np.all(original.received_times <= original.times)


def test_control_reference_roundtrip_and_loader_preserve_held_frame_age(tmp_path):
    torch.set_num_threads(1)
    robot = K1Model()
    human = human_motion(robot, 5 + np.arange(7) * 0.07)
    metadata = {
        "source_motion_id": "synthetic/low-rate",
        "capture_group": "synthetic/low-rate",
        "family": "reach",
        "split": "train",
        "model_signature": robot.signature,
        "kinematics_accepted": True,
        "reference_path": "clip.npz",
    }
    clip, _ = retarget_at_control_rate(robot, human, metadata)
    clip.save(tmp_path / "clip.npz")
    loaded = MotionClip.load(tmp_path / "clip.npz")
    np.testing.assert_array_equal(loaded.source_times, clip.source_times)
    np.testing.assert_array_equal(loaded.received_times, clip.received_times)
    assert loaded.frame(2).source_time == 5 and loaded.frame(2).received_time == 0
    assert loaded.frame(2, received_time=10).received_time == 10
    (tmp_path / "index.jsonl").write_text(json.dumps(metadata) + "\n")
    library = MotionLibrary(tmp_path, robot, "cpu")
    assert library.lengths[0] == len(clip.times)
    np.testing.assert_allclose(library.values["age"][0], clip.times - clip.received_times, atol=1e-8)
    assert library.values["age"][0, 2].item() == pytest.approx(0.04)
    with pytest.raises(ValueError, match="future"):
        MotionClip(clip.times, clip.values, clip.metadata, clip.source_times, clip.times + 0.001)
    # The review artifact must display the exact held/new native-rate human
    # frames that produced the control-rate reference, including a nonzero clock.
    from k1_motion.viewer import write_viewer

    np.savez_compressed(tmp_path / "human.npz", **human)
    write_viewer(tmp_path / "clip.npz", tmp_path / "review.html", tmp_path / "human.npz")
    text = (tmp_path / "review.html").read_text()
    payload = json.loads(text.split('<script id="motion" type="application/json">')[1].split("</script>")[0])
    assert len(payload["human"]) == len(payload["robot"]) == len(clip.times)
    assert payload["human"][0] == payload["human"][2]
