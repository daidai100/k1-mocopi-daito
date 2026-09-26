"""Versioned repair of legacy source-clock velocities in 50 Hz training goals.

This changes derivative channels only. Saved poses, clocks and prior geometric
admission remain exact; neither the old nor repaired goal is physics-qualified.
"""

from concurrent.futures import ProcessPoolExecutor
import copy
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from .contracts import MotionClip
from .math3d import rotation

REPAIR_VERSION = "reference-playback-velocity-v1"
LEGACY_VERSION = "causal-ik-v7-contact-root-projection"
PLAYBACK_RETARGET_VERSION = "causal-ik-v7-playback-velocity-v1"
UNCHANGED_VERSIONS = {
    "causal-ik-v8-command-path-recovery",
    "causal-ik-v8-command-path-recovery+control-tick-hold-v1",
    "causal-knee-support-ik-v1",
    PLAYBACK_RETARGET_VERSION,
}
VELOCITY_KEYS = {"joint_velocity", "root_velocity"}


def _derivatives(clip, times):
    dt = np.diff(times)
    if np.any(dt <= 0) or not np.isfinite(dt).all():
        raise ValueError("Derivative clock must strictly increase")
    joint = np.zeros_like(clip.values["joint_position"])
    root = np.zeros_like(clip.values["root_velocity"])
    joint[1:] = np.diff(clip.values["joint_position"], axis=0) / dt[:, None]
    root[1:, :3] = np.diff(clip.values["root_position"], axis=0) / dt[:, None]
    orientations = rotation(clip.values["root_orientation"])
    root[1:, 3:] = (orientations[1:] * orientations[:-1].inv()).as_rotvec() / dt[:, None]
    return joint, root


def playback_derivatives(clip):
    """Causal world-frame derivatives of saved playback poses, first frame zero."""
    return _derivatives(clip, clip.times)


def velocity_errors(clip, *, source_clock=False):
    """Max component error including the zero initial causal derivative."""
    times = clip.source_times if source_clock else clip.times
    if times is None:
        raise ValueError("Missing original source clock")
    joint, root = _derivatives(clip, times)
    return {
        "joint_velocity_max_rad_s": float(np.max(abs(joint - clip.values["joint_velocity"]))),
        "root_linear_velocity_max_m_s": float(np.max(abs(root[:, :3] - clip.values["root_velocity"][:, :3]))),
        "root_angular_velocity_max_rad_s": float(
            np.max(abs(root[:, 3:] - clip.values["root_velocity"][:, 3:]))
        ),
    }


def repair_v7_clip(clip):
    if (
        clip.metadata.get("retarget_version") != LEGACY_VERSION
        or clip.metadata.get("split") != "train"
        or clip.metadata.get("is_mirror") is not False
    ):
        raise ValueError("Clock repair requires an original v7 training reference")
    if "velocity_clock_repair" in clip.metadata:
        raise ValueError("Reference already declares a velocity clock repair")
    if not clip.values["valid"].all():
        raise ValueError("Clock repair cannot admit invalid ticks")
    if not np.allclose(np.diff(clip.times), 0.02, rtol=0, atol=1e-12):
        raise ValueError("Clock repair requires the declared 50 Hz playback clock")
    source_errors = velocity_errors(clip, source_clock=True)
    if max(source_errors.values()) > 1e-8:
        raise ValueError("Legacy velocity fields do not match their known source-clock producer")
    joint, root = _derivatives(clip, clip.times)
    values = {key: value.copy() for key, value in clip.values.items()}
    values.update(joint_velocity=joint, root_velocity=root)
    receipt = dict(
        version=REPAIR_VERSION,
        source_retarget_version=LEGACY_VERSION,
        derivative_clock="saved_motion_playback_times",
        method="causal_backward_difference",
        root_linear_frame="world",
        root_angular_frame="world",
        first_frame="zero; no past sample is available",
        changed_arrays=sorted(VELOCITY_KEYS),
        geometry_and_source_clocks_unchanged=True,
        prior_admission_unchanged=True,
        physics_qualified=False,
    )
    metadata = copy.deepcopy(clip.metadata)
    metadata["velocity_clock_repair"] = receipt
    return MotionClip(
        clip.times.copy(), values, metadata, clip.source_times.copy(), clip.received_times.copy()
    )


def _process(job):
    row, source, staging = job
    original_path = (Path(source) / row["reference_path"]).resolve(strict=True)
    clip = MotionClip.load(original_path)
    for key in ("id", "split", "is_mirror", "capture_group", "model_signature", "retarget_version"):
        if row.get(key) != clip.metadata.get(key):
            raise ValueError(f"Reference row/payload differs: {row['id']} {key}")
    if not clip.values["valid"].all():
        raise ValueError("Invalid training reference: " + row["id"])
    before = velocity_errors(clip)
    repaired = row["retarget_version"] == LEGACY_VERSION
    result = repair_v7_clip(clip) if repaired else clip
    after = velocity_errors(result)
    if max(after.values()) > 1e-8:
        raise ValueError("Playback derivative mismatch in " + row["id"])
    unchanged = all(
        np.array_equal(result.values[k], clip.values[k]) for k in clip.values if k not in VELOCITY_KEYS
    )
    unchanged &= all(
        np.array_equal(getattr(result, k), getattr(clip, k))
        for k in ("times", "source_times", "received_times")
    )
    if not unchanged:
        raise ValueError("Clock repair changed pose, validity, contact or clock arrays")
    relative = Path("clips") / (row["id"] + ".npz")
    destination = Path(staging) / relative
    updated = {**row, "reference_path": str(relative)}
    if repaired:
        result.save(destination)
        reloaded = MotionClip.load(destination)
        if any(not np.array_equal(result.values[k], reloaded.values[k]) for k in result.values):
            raise ValueError("Saved repaired derivative payload differs")
        updated["velocity_clock_repair"] = result.metadata["velocity_clock_repair"]
    else:
        destination.symlink_to(original_path)
    evidence = dict(
        id=row["id"],
        family=row["family"],
        dataset=row["dataset"],
        retarget_version=row["retarget_version"],
        frames=len(clip.times),
        repaired=repaired,
        invalid_ticks=int((~result.values["valid"].astype(bool)).sum()),
        geometry_arrays_unchanged=bool(unchanged),
        before=before,
        after=after,
    )
    return updated, evidence


def repair_training_library(source, output, train_ids, *, workers=1):
    source, output = Path(source).resolve(), Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError("Refusing to overwrite output library")
    if workers < 1:
        raise ValueError("Worker count must be positive")
    payload = (source / "index.jsonl").read_bytes()
    rows = [json.loads(line) for line in payload.splitlines()]
    by_id = {row["id"]: row for row in rows}
    if len(by_id) != len(rows) or len(set(train_ids)) != len(train_ids):
        raise ValueError("Duplicate source or selected motion IDs")
    if not train_ids or not set(train_ids) <= set(by_id):
        raise ValueError("Selected training IDs are empty or missing")
    selected_ids = set(train_ids)
    # Keep stable environment/sample indices across the original and repaired
    # treatments; a membership manifest must not reorder the shared corpus.
    selected = [row for row in rows if row["id"] in selected_ids]
    for row in selected:
        if (
            row.get("split") != "train"
            or row.get("is_mirror") is not False
            or not row.get("training_eligible")
            or not re.fullmatch(r"[A-Za-z0-9_-]+", row["id"])
            or row.get("retarget_version") not in UNCHANGED_VERSIONS | {LEGACY_VERSION}
            or "velocity_clock_repair" in row
        ):
            raise ValueError("Unsupported or ineligible training reference: " + row["id"])
    staging = output.with_name(output.name + ".partial")
    staging.mkdir(parents=True, exist_ok=False)
    (staging / "clips").mkdir()
    jobs = [(row, str(source), str(staging)) for row in selected]
    if workers == 1:
        results = list(map(_process, jobs))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_process, jobs, chunksize=16))
    updated, evidence = zip(*results)
    summary = {"before": {}, "after": {}}
    for row in evidence:
        for label in ("all", "family/" + row["family"], "retarget_version/" + row["retarget_version"]):
            for phase in summary:
                stats = summary[phase].setdefault(label, {"clips": 0, "frames": 0})
                stats["clips"] += 1
                stats["frames"] += row["frames"]
                for key, value in row[phase].items():
                    stats[key] = max(stats.get(key, 0.0), value)
    report = dict(
        version=REPAIR_VERSION,
        source_library=str(source),
        source_manifest_sha256=hashlib.sha256(payload).hexdigest(),
        clips=len(evidence),
        repaired_clips=sum(row["repaired"] for row in evidence),
        unchanged_clips=sum(not row["repaired"] for row in evidence),
        invalid_ticks=sum(row["invalid_ticks"] for row in evidence),
        geometry_arrays_unchanged=all(row["geometry_arrays_unchanged"] for row in evidence),
        source_payloads_modified=False,
        training_only=True,
        mirrors=0,
        admission="Original acceptance records retained; positions and command path unchanged",
        first_frame="zero causal derivative",
        physics_qualified=False,
        heldout_payloads_read_or_modified=False,
        **summary,
    )
    (staging / "index.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in updated))
    (staging / "audit.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in evidence))
    (staging / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not all((staging / row["reference_path"]).exists() for row in updated):
        raise ValueError("Derived library contains a broken payload link")
    staging.rename(output)
    return report


def order_repaired_library(source, repaired, output, train_ids):
    """Publish an immutable source-order view, sharing previously audited payloads."""
    source, repaired, output = Path(source).resolve(), Path(repaired).resolve(), Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError("Refusing to overwrite ordered library")
    source_payload = (source / "index.jsonl").read_bytes()
    original = [json.loads(line) for line in source_payload.splitlines()]
    selected_ids = set(train_ids)
    if not selected_ids or len(selected_ids) != len(train_ids):
        raise ValueError("Empty or duplicate training IDs")
    selected = [row for row in original if row["id"] in selected_ids]
    if (
        len(selected) != len(selected_ids)
        or len({r["id"] for r in selected}) != len(selected)
        or any(row.get("split") != "train" or row.get("is_mirror") is not False for row in selected)
    ):
        raise ValueError("Selected source training identities differ")
    rows = [json.loads(line) for line in (repaired / "index.jsonl").read_text().splitlines()]
    audits = [json.loads(line) for line in (repaired / "audit.jsonl").read_text().splitlines()]
    by_id, by_audit = {r["id"]: r for r in rows}, {r["id"]: r for r in audits}
    if (
        set(by_id) != selected_ids
        or set(by_audit) != selected_ids
        or len(rows) != len(by_id)
        or len(audits) != len(by_audit)
    ):
        raise ValueError("Audited repaired library has different motion identities")
    def stripped(value):
        return {
            k: v for k, v in value.items() if k not in ("reference_path", "velocity_clock_repair")
        }

    for row in selected:
        corrected = by_id[row["id"]]
        if stripped(row) != stripped(corrected):
            raise ValueError("Repaired library changed source metadata: " + row["id"])
        expected_path = str(Path("clips") / (row["id"] + ".npz"))
        if corrected["reference_path"] != expected_path or not (repaired / expected_path).exists():
            raise ValueError("Repaired payload path differs or is broken")
    report = json.loads((repaired / "report.json").read_text())
    if (
        report.get("version") != REPAIR_VERSION
        or report.get("clips") != len(selected)
        or report.get("invalid_ticks") != 0
        or not report.get("geometry_arrays_unchanged")
        or report.get("source_manifest_sha256") != hashlib.sha256(source_payload).hexdigest()
    ):
        raise ValueError("Repaired library lacks matching source audit receipt")
    source_ids = [row["id"] for row in selected]
    report["ordering"] = dict(
        version="source-manifest-order-v1",
        source_manifest_order_preserved=True,
        ordered_ids_sha256=hashlib.sha256(json.dumps(source_ids, separators=(",", ":")).encode()).hexdigest(),
        immutable_payload_library=str(repaired),
        payloads_rewritten=False,
        reason="Preserve sample identity under matched seeds and environment indices",
    )
    staging = output.with_name(output.name + ".partial")
    staging.mkdir(parents=True, exist_ok=False)
    (staging / "clips").symlink_to(repaired / "clips", target_is_directory=True)
    (staging / "index.jsonl").write_text(
        "".join(json.dumps(by_id[key], sort_keys=True) + "\n" for key in source_ids)
    )
    (staging / "audit.jsonl").write_text(
        "".join(json.dumps(by_audit[key], sort_keys=True) + "\n" for key in source_ids)
    )
    (staging / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    staging.rename(output)
    return report
