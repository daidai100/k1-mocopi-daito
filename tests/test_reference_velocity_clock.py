"""Failure modes: wrong clock/frame, future use, changed poses, wrong cohort/version,
stale receipts, row/payload identity mismatch, overwrites and incomplete releases.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from k1_motion.contracts import CONVENTIONS, JOINT_NAMES, LANDMARKS, MotionClip
from k1_motion.reference_velocity_clock import (
    REPAIR_VERSION,
    repair_training_library,
    repair_v7_clip,
    velocity_errors,
)


def fixture_clip(key="fixture", version="causal-ik-v7-contact-root-projection"):
    times = np.arange(8) * 0.02
    source = np.array([0.0, 0.016666, 0.033332, 0.058331, 0.074997, 0.099996, 0.116662, 0.133328])
    q = np.arange(8)[:, None] * np.linspace(0.001, 0.015, 22)
    pos = np.c_[times * 0.4, times**2, np.full(8, 0.6)]
    r = Rotation.from_euler("xyz", np.c_[times * 0.2, times * 0.3, times * 0.8])
    quaternion = r.as_quat()[:, [3, 0, 1, 2]]
    dt = np.diff(source if version == "causal-ik-v7-contact-root-projection" else times)
    joint = np.zeros_like(q)
    joint[1:] = np.diff(q, axis=0) / dt[:, None]
    root = np.zeros((8, 6))
    root[1:, :3] = np.diff(pos, axis=0) / dt[:, None]
    root[1:, 3:] = (r[1:] * r[:-1].inv()).as_rotvec() / dt[:, None]
    values = dict(
        root_position=pos,
        root_orientation=quaternion,
        root_velocity=root,
        joint_position=q,
        joint_velocity=joint,
        landmarks=np.tile(pos[:, None, :], (1, len(LANDMARKS), 1)),
        contacts=np.ones((8, 2)),
        contact_confidence=np.ones((8, 2)),
        valid=np.ones(8),
    )
    metadata = dict(
        id=key,
        split="train",
        is_mirror=False,
        capture_group="fixture/" + key,
        model_signature="fixture-model",
        retarget_version=version,
        joint_names=list(JOINT_NAMES),
        conventions=CONVENTIONS,
        family="walk",
        dataset="fixture",
        training_eligible=True,
        kinematics_accepted=True,
        recovery_audit=dict(accepted=True, physics_qualified=False),
    )
    return MotionClip(times, values, metadata, source, source.copy())


def fixture_library(
    path,
    versions=(
        "causal-ik-v7-contact-root-projection",
        "causal-ik-v8-command-path-recovery",
        "causal-knee-support-ik-v1",
    ),
):
    path.mkdir()
    rows = []
    for i, version in enumerate(versions):
        clip = fixture_clip(str(i), version)
        clip.save(path / "clips" / f"{i}.npz")
        rows.append({**clip.metadata, "reference_path": f"clips/{i}.npz"})
    (path / "index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rows


def test_repair_derivatives_world_angular_and_zero_initial_preserve_everything_else():
    clip = fixture_clip()
    result = repair_v7_clip(clip)
    assert max(velocity_errors(result).values()) < 1e-12
    assert not result.values["root_velocity"][0].any()
    assert not result.values["joint_velocity"][0].any()
    for key in set(clip.values) - {"root_velocity", "joint_velocity"}:
        np.testing.assert_array_equal(result.values[key], clip.values[key])
    for field in ["times", "source_times", "received_times"]:
        np.testing.assert_array_equal(getattr(result, field), getattr(clip, field))
    assert {k: v for k, v in result.metadata.items() if k != "velocity_clock_repair"} == clip.metadata
    assert result.metadata["velocity_clock_repair"]["version"] == REPAIR_VERSION
    assert not np.array_equal(result.values["root_velocity"], clip.values["root_velocity"])


def test_repair_is_causal_at_every_prefix_and_does_not_alias_source():
    clip = fixture_clip()
    result = repair_v7_clip(clip)
    for stop in range(2, len(clip.times)):
        prefix = MotionClip(
            clip.times[:stop],
            {k: v[:stop] for k, v in clip.values.items()},
            dict(clip.metadata),
            clip.source_times[:stop],
            clip.received_times[:stop],
        )
        corrected = repair_v7_clip(prefix)
        for key in ["joint_velocity", "root_velocity"]:
            np.testing.assert_array_equal(corrected.values[key], result.values[key][:stop])
    assert not np.shares_memory(result.values["joint_position"], clip.values["joint_position"])


@pytest.mark.parametrize(
    "fault",
    [
        "unknown",
        "heldout",
        "mirror",
        "invalid",
        "wrong_source_velocity",
        "already_repaired",
        "nonzero_initial",
    ],
)
def test_repair_rejects_unexplained_or_out_of_scope_inputs(fault):
    clip = fixture_clip()
    clip.values = {key: value.copy() for key, value in clip.values.items()}
    if fault == "unknown":
        clip.metadata["retarget_version"] = "unknown-v99"
    elif fault == "heldout":
        clip.metadata["split"] = "test"
    elif fault == "mirror":
        clip.metadata["is_mirror"] = True
    elif fault == "invalid":
        clip.values["valid"][2] = 0
    elif fault == "wrong_source_velocity":
        clip.values["joint_velocity"][2, 3] += 1
    elif fault == "already_repaired":
        clip.metadata["velocity_clock_repair"] = {"version": REPAIR_VERSION}
    elif fault == "nonzero_initial":
        clip.values["root_velocity"][0, 0] = 1
    with pytest.raises(ValueError):
        repair_v7_clip(clip)


def test_library_cli_e2e_retains_count_originals_admission_and_link_targets(tmp_path):
    source = tmp_path / "source"
    rows = fixture_library(source)
    identities = {p.name: p.read_bytes() for p in (source / "clips").glob("*.npz")}
    selection = tmp_path / "ids.json"
    selection.write_text(json.dumps({"train_ids": [r["id"] for r in rows]}))
    output = tmp_path / "repaired"
    script = Path(__file__).parents[1] / "scripts/repair_reference_velocity_clock.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--library",
            str(source),
            "--output",
            str(output),
            "--train-ids",
            str(selection),
            "--workers",
            "2",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads((output / "report.json").read_text())
    assert report["clips"] == 3 and report["repaired_clips"] == 1 and report["unchanged_clips"] == 2
    assert report["invalid_ticks"] == 0 and report["geometry_arrays_unchanged"]
    assert report["before"]["all"]["joint_velocity_max_rad_s"] > 0.01
    assert report["after"]["all"]["joint_velocity_max_rad_s"] < 1e-10
    new_rows = [json.loads(x) for x in (output / "index.jsonl").read_text().splitlines()]
    assert [r["id"] for r in new_rows] == [r["id"] for r in rows]
    for old, new in zip(rows, new_rows):
        assert {k: v for k, v in new.items() if k not in {"reference_path", "velocity_clock_repair"}} == {
            k: v for k, v in old.items() if k != "reference_path"
        }
        assert (output / new["reference_path"]).exists()
        assert (source / "clips" / f"{old['id']}.npz").read_bytes() == identities[f"{old['id']}.npz"]
    with pytest.raises(FileExistsError):
        repair_training_library(source, output, [r["id"] for r in rows])


def test_repair_preserves_source_order_even_when_selected_ids_are_reversed(tmp_path):
    source = tmp_path / "source"
    rows = fixture_library(source)
    output = tmp_path / "out"
    repair_training_library(source, output, [r["id"] for r in reversed(rows)])
    actual = [json.loads(line)["id"] for line in (output / "index.jsonl").read_text().splitlines()]
    assert actual == [r["id"] for r in rows]


def test_source_order_variant_reuses_immutable_payloads_and_preserves_source_order(tmp_path):
    from k1_motion.reference_velocity_clock import order_repaired_library

    source = tmp_path / "source"
    rows = fixture_library(source)
    repaired = tmp_path / "repaired"
    ids = [r["id"] for r in rows]
    repair_training_library(source, repaired, ids)
    # Recreate the historical sorted/selected-order artifact in this fixture.
    for filename in ["index.jsonl", "audit.jsonl"]:
        path = repaired / filename
        path.write_text("\n".join(reversed(path.read_text().splitlines())) + "\n")
    old_index = (repaired / "index.jsonl").read_bytes()
    old_files = {p.name: p.read_bytes() for p in (repaired / "clips").glob("*.npz")}
    output = tmp_path / "ordered"
    report = order_repaired_library(source, repaired, output, list(reversed(ids)))
    assert report["ordering"]["source_manifest_order_preserved"]
    assert report["clips"] == 3 and report["repaired_clips"] == 1
    actual = [json.loads(line)["id"] for line in (output / "index.jsonl").read_text().splitlines()]
    assert actual == ids
    assert (output / "clips").is_symlink()
    assert (repaired / "index.jsonl").read_bytes() == old_index
    for key in ids:
        assert (output / "clips" / f"{key}.npz").read_bytes() == old_files[f"{key}.npz"]
    with pytest.raises(FileExistsError):
        order_repaired_library(source, repaired, output, ids)


@pytest.mark.parametrize(
    "fault", ["unknown_version", "row_payload_identity", "duplicate", "nontraining", "missing_id"]
)
def test_library_rejects_before_publishing(tmp_path, fault):
    source = tmp_path / "source"
    rows = fixture_library(source)
    selection = [r["id"] for r in rows]
    if fault == "unknown_version":
        rows[1]["retarget_version"] = "unknown-v99"
    elif fault == "row_payload_identity":
        rows[1]["capture_group"] = "wrong"
    elif fault == "duplicate":
        selection.append(selection[0])
    elif fault == "nontraining":
        rows[1]["split"] = "validation"
    elif fault == "missing_id":
        selection.append("absent")
    (source / "index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    output = tmp_path / "out"
    with pytest.raises(ValueError):
        repair_training_library(source, output, selection)
    assert not output.exists()


@pytest.mark.parametrize("source_hz", [120, 20])
def test_control_rate_producer_versions_playback_derivatives_with_skipped_and_held_frames(source_hz):
    from k1_motion.robot import K1Model
    from k1_motion.streaming import retarget_at_control_rate

    robot = K1Model()
    times = 3 + np.arange(9) / source_hz
    positions = np.tile(robot.neutral_landmarks * 1.8, (len(times), 1, 1))
    positions[:, 5, 0] += np.arange(len(times)) * 0.002
    human = dict(times=times, positions=positions, orientations=np.tile([1.0, 0, 0, 0], (len(times), 17, 1)))
    clip, reports = retarget_at_control_rate(robot, human, {"source_motion_id": "fixture/clock"})
    assert clip.metadata["retarget_version"] == "causal-ik-v7-playback-velocity-v1"
    assert all(r["retarget_version"] == clip.metadata["retarget_version"] for r in reports)
    assert max(velocity_errors(clip).values()) < 1e-10
    assert clip.metadata["sampling"]["velocity_clock"] == "sample_clock"
    assert np.all(clip.received_times <= clip.times)
    if source_hz == 20:
        held = np.r_[False, np.diff(clip.source_times) == 0]
        assert held.any()
        assert not clip.values["joint_velocity"][held].any()
        assert not clip.values["root_velocity"][held].any()
