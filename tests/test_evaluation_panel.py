import json

import numpy as np
import pytest
import torch

from k1_motion.evaluation import THRESHOLDS
from k1_motion.evaluation_panel import _trial, build_panel, evaluate_panel
from k1_motion.robot import K1Model


@pytest.fixture
def streaming_data(tmp_path):
    torch.set_num_threads(1)
    robot = K1Model()
    human = {
        "times": np.arange(31) * 0.01,
        "positions": np.tile(robot.neutral_landmarks * 1.8, (31, 1, 1)),
        "orientations": np.tile([1.0, 0, 0, 0], (31, 17, 1)),
    }
    np.savez(tmp_path / "human.npz", **human)
    row = {
        "id": "a",
        "split": "test",
        "family": "stance",
        "capture_group": "synthetic/a",
        "source_motion_id": "synthetic/a",
        "human_path": "human.npz",
        "kinematics_accepted": True,
        "model_signature": robot.signature,
    }
    rejected = {
        **row,
        "id": "b",
        "capture_group": "synthetic/b",
        "source_motion_id": "synthetic/b",
        "kinematics_accepted": False,
        "rejection_counts": {"self_collision": 1},
    }
    (tmp_path / "index.jsonl").write_text("\n".join(map(json.dumps, [row, rejected])) + "\n")
    return robot, human, row


def test_frozen_panel_retains_rejections_and_verifies_human_bytes(tmp_path, streaming_data):
    panel_path = tmp_path / "panel.json"
    panel = build_panel(tmp_path, panel_path)
    assert panel["independent_recordings"] == 2 and panel["reference_rejections"] == 1
    assert len(panel["trials"]) == 20
    assert sum(t["recording_id"] == "b" for t in panel["trials"]) == 10
    with pytest.raises(FileExistsError):
        build_panel(tmp_path, panel_path)
    result = evaluate_panel(tmp_path, panel_path, tmp_path / "replay", limit=2)
    trials = [json.loads(s) for s in (tmp_path / "replay/trials.jsonl").read_text().splitlines()]
    assert trials[0]["completed"] and trials[0]["resets_during_trial"] == 0
    assert trials[1]["reason"] == "offline_reference_rejected"
    assert result["partial_panel"] and not result["behaviorally_accepted"]
    with np.load(tmp_path / "replay" / (panel["trials"][0]["trial_id"] + ".npz")) as trace:
        np.testing.assert_array_equal(trace["source_indices"], np.arange(0, 31, 2))
        assert np.isfinite(trace["qpos"]).all()
        for name, width in (
            ("qvel", 28), ("actions", 22), ("torques", 22), ("reference_joints", 22),
            ("reference_roots", 3), ("reference_orientations", 4), ("reference_contacts", 2),
        ):
            assert trace[name].shape == (len(trace["qpos"]), width)
            assert np.isfinite(trace[name]).all()
        assert np.max(np.abs(trace["actions"])) <= 1.0
    # Changing the canonical recording without changing the index is still caught.
    with (tmp_path / "human.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="canonical human data changed"):
        evaluate_panel(tmp_path, panel_path, tmp_path / "changed-data", limit=1)


def test_named_diagnostics_preserve_frozen_order_and_rejection(tmp_path, streaming_data):
    panel_path = tmp_path / "panel.json"
    panel = build_panel(tmp_path, panel_path)
    selected = [panel["trials"][i]["trial_id"] for i in (3, 2)]
    result = evaluate_panel(tmp_path, panel_path, tmp_path / "selected", trial_ids=selected)
    trials = [json.loads(s) for s in (tmp_path / "selected/trials.jsonl").read_text().splitlines()]
    assert [t["trial_id"] for t in trials] == list(reversed(selected))
    assert trials[1]["reason"] == "offline_reference_rejected"
    assert result["partial_panel"] and result["trials"] == 2
    assert not result["behaviorally_accepted"]
    parallel = evaluate_panel(tmp_path, panel_path, tmp_path / "parallel", trial_ids=selected, workers=2)
    assert parallel["families"] == result["families"]
    for path in (tmp_path / "selected").glob("*.npz"):
        with np.load(path) as serial, np.load(tmp_path / "parallel" / path.name) as worker:
            for name in serial.files:
                np.testing.assert_array_equal(serial[name], worker[name])
    for names, limit in ((["absent"], None), ([selected[0]] * 2, None), (selected, 1)):
        with pytest.raises(ValueError):
            evaluate_panel(tmp_path, panel_path, tmp_path / "invalid", limit=limit, trial_ids=names)
        assert not (tmp_path / "invalid").exists()


def test_end_to_end_replay_commands_do_not_depend_on_future(tmp_path, streaming_data):
    robot, original, row = streaming_data
    panel = build_panel(tmp_path, tmp_path / "panel.json")
    traces = []
    for direction in (-1, 1):
        human = {k: v.copy() for k, v in original.items()}
        human["positions"][24:, 5, 0] += direction * 0.05
        case = {**panel["trials"][0], "trial_id": f"future-{direction}", "scenario": "clean"}
        result = _trial(robot, None, row, case, human, panel["assumptions"], THRESHOLDS, tmp_path)
        assert result["completed"]
        with np.load(tmp_path / (case["trial_id"] + ".npz")) as trace:
            traces.append(trace["targets"].copy())
    np.testing.assert_array_equal(traces[0][:12], traces[1][:12])
    assert not np.allclose(traces[0][12:], traces[1][12:])
