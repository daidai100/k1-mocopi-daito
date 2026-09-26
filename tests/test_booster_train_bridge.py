"""End-to-end preparation contract for the native Booster Train handoff."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from k1_motion.contracts import MotionClip
from k1_motion.robot import K1Model


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/train_booster.py"


def library(tmp_path, *, eligible=True):
    robot = K1Model()
    directory = tmp_path / "library"
    directory.mkdir()
    row = {
        "id": "standing",
        "capture_group": "synthetic/standing",
        "family": "stance",
        "split": "train",
        "kinematics_accepted": True,
        "training_eligible": eligible,
        "model_signature": robot.signature,
        "reference_path": "standing.npz",
    }
    refs = [robot.neutral_reference(i * 0.02) for i in range(50)]
    MotionClip.from_references(refs, row).save(directory / "standing.npz")
    (directory / "index.jsonl").write_text(json.dumps(row) + "\n")
    return directory


def invoke(directory, output):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--library", str(directory), "--motion-id", "standing",
         "--output", str(output), "--prepare-only"],
        capture_output=True, text=True,
    )


def test_prepare_booster_motion_and_run_receipt(tmp_path):
    directory = library(tmp_path)
    output = tmp_path / "bridge"
    result = invoke(directory, output)
    assert result.returncode == 0, result.stderr
    csv = np.loadtxt(output / "motion.csv", delimiter=",")
    assert csv.shape == (50, 29)
    np.testing.assert_allclose(csv[:, 3:6], 0)
    np.testing.assert_allclose(csv[:, 6], 1)
    np.testing.assert_allclose(csv[:, 7:], np.broadcast_to(K1Model().neutral, (50, 22)), atol=1e-6)
    receipt = json.loads((output / "prepare.json").read_text())
    assert receipt["motion_id"] == "standing"
    assert receipt["frames"] == 50 and receipt["fps"] == 50
    assert receipt["native_task"] == "Booster-K1-MJ_Dance_004-v0"
    assert receipt["status"] == "prepared_only"
    with np.load(output / "motion.npz") as motion:
        assert motion["joint_pos"].shape == (50, 22)
        assert motion["body_pos_w"].shape[0] == 50
        assert "trunk" in motion["body_names"].tolist()
        np.testing.assert_allclose(motion["body_quat_w"][:, 0, 0], 1, atol=1e-6)


def test_reject_ineligible_clip_before_export(tmp_path):
    directory = library(tmp_path, eligible=False)
    output = tmp_path / "bridge"
    result = invoke(directory, output)
    assert result.returncode != 0
    assert "training_eligible" in result.stderr
    assert not output.exists()


def test_prepare_accepts_library_symlinked_clip_storage(tmp_path):
    directory = library(tmp_path)
    shared = tmp_path / "shared"
    shared.mkdir()
    (directory / "standing.npz").rename(shared / "standing.npz")
    (directory / "standing.npz").symlink_to(shared / "standing.npz")
    result = invoke(directory, tmp_path / "bridge")
    assert result.returncode == 0, result.stderr


def test_native_reset_tracks_neutral_reference(tmp_path):
    isaac_python = os.environ.get("K1_ISAAC_PYTHON")
    if not isaac_python:
        import pytest
        pytest.skip("Set K1_ISAAC_PYTHON for the native Isaac integration test")
    directory = library(tmp_path)
    prepared = tmp_path / "bridge"
    assert invoke(directory, prepared).returncode == 0
    output = tmp_path / "native"
    native = subprocess.run(
        [isaac_python, str(SCRIPT.parent / "booster_native_train.py"),
         "--motion", str(prepared / "motion.npz"), "--output", str(output),
         "--num-envs", "128", "--iterations", "1", "--seed", "42", "--device", "cpu",
         "--diagnose-reset"], capture_output=True, text=True,
    )
    assert native.returncode == 0, native.stderr[-1500:]
    report = json.loads((output / "reset_diagnostic.json").read_text())
    assert max(item["max_z_error_m"] for item in report.values()) < 0.25
