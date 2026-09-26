#!/usr/bin/env python3
"""Recover full boxing/avoidance recordings from the other local source corpora."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
VERSION = "complementary-whole-body-references-v1"


def gap_family(row):
    """Do not classify 'striking a pose' as actual boxing."""
    text = " ".join(row.get("annotations", [])).lower()
    if re.search(r"punch|box(?:ing|er)|\bjab\b|uppercut|fight", text):
        return "punch"
    if re.search(r"step(?:s|ping)? over|walk(?:s|ing)? over.*obstacle|hurdle", text):
        return "step_over"
    if re.search(r"obstacle|dodg|duck", text):
        return "avoidance"
    return None


def unconfigured_support(row):
    text = " ".join(row.get("annotations", [])).lower()
    return bool(re.search(r"stepping stones|step stones|\bbeam\b|\bplatform\b|stairs|ladder|climb"
                          r"|(?:on|along) (?:a |the )?wall|placing.*feet on objects", text))


def convert(job):
    from k1_motion.adapters import bvh_frames, mmm_frames
    from k1_motion.contracts import MotionClip
    from k1_motion.ground_reference import correct_walking_ground
    from k1_motion.math3d import rotation
    from k1_motion.motion_coverage import measure_motion, movement_tags
    from k1_motion.recovery_geometry import recovery_model
    from k1_motion.recovery_validation import audit_recovery, retarget_recovery
    from k1_motion.streaming import retarget_at_control_rate
    record, output, revision = job
    output = Path(output)
    started = time.monotonic()
    key = hashlib.sha256(record["source_motion_id"].encode()).hexdigest()[:16]
    row = {**record, "id": key, "source_revision": revision, "source_path": record["path"],
           "source_duration_seconds": record["duration_seconds"], "is_mirror": False,
           "physics_qualified": False, "training_eligible": False, "human_reviewed": False,
           "kinematics_accepted": False, "complementary_reference_version": VERSION,
           "license_directory": str(ROOT / "licenses" / record["dataset"])}
    try:
        if record["format"] == "BVH":
            frames = list(bvh_frames(ROOT / record["path"], record["dataset"], record["source_motion_id"]))
        elif record["format"] == "MMM":
            frames = list(mmm_frames(ROOT / record["path"],
                ROOT / "data/reference/mmmpy_lite/mmmpy_lite/data/models/mmm/mmm.urdf",
                record.get("track_index", 0), record["source_motion_id"]))
        else:
            raise ValueError("Unsupported source format")
    except ValueError as error:
        return {**row, "source_error": str(error), "elapsed_seconds": time.monotonic() - started}
    if len(frames) < 2:
        return {**row, "source_error": "Too few canonical human frames"}
    human = {"times": np.array([f.source_time for f in frames]),
             "positions": np.stack([f.positions for f in frames]),
             "orientations": np.stack([f.orientations for f in frames])}
    row["source_clock_origin_s"] = float(human["times"][0])
    human["times"] -= human["times"][0]
    robot = recovery_model("retarget")
    row["model_signature"] = robot.signature
    baseline, reports = retarget_at_control_rate(robot, human, row)
    baseline_error = float(np.mean([r["rms_landmark_error_m"] for r in reports]))
    baseline_row = {"rms_landmark_error_m": baseline_error}
    audit = audit_recovery(baseline, baseline, reports, baseline_row, ground_profile=True)
    row["baseline_recovery_audit"] = audit
    candidate = baseline
    if not audit["accepted"]:
        candidate, reports = retarget_recovery(robot, human, row,
            control_tick_hold=record.get('recovery_profile') == 'control-tick-hold-v1')
    error = float(np.mean([r["rms_landmark_error_m"] for r in reports]))
    candidate, correction = correct_walking_ground(candidate)
    error += correction["mean_landmark_error_increase_bound_m"]
    # Persist before auditing; no trimming, warping, invalid-tick dropping or
    # previous-controller success filter is used.
    attempt = output / "attempts" / f"{key}.npz"
    candidate.save(attempt)
    candidate = MotionClip.load(attempt)
    audit = audit_recovery(candidate, baseline, [{"rms_landmark_error_m": error}],
                           baseline_row, ground_profile=True)
    reasons = list(audit["rejection_reasons"])
    if (audit["ground_profile"]["max_nonfoot_penetration_m"] > .0001
            or correction["nonfoot_ground_contact_samples"]):
        reasons.append("nonfoot_support_task_requires_separate_audit")
    upright = rotation(candidate.values["root_orientation"]).apply([0, 0, 1])[:, 2]
    if candidate.values["root_position"][:, 2].min() < .22 or upright.min() < .2:
        reasons.append("low_support_task_requires_separate_audit")
    features = measure_motion(candidate, robot.limits)
    if row["family"] == "step_over" and features["event_longest_s"]["high_foot_lift_proxy"] < .1:
        reasons.append("step_over_foot_lift_not_preserved")
    if np.any(candidate.values["joint_position"] < robot.limits[:, 0] - 1e-8) or np.any(
            candidate.values["joint_position"] > robot.limits[:, 1] + 1e-8):
        reasons.append("joint_limits")
    duration = float(human["times"][-1] - human["times"][0])
    if duration - candidate.times[-1] >= robot.control_dt + 1e-9:
        raise ValueError("Source duration was truncated")
    if not np.array_equal(candidate.values["joint_position"],
                          MotionClip.load(attempt).values["joint_position"]):
        raise ValueError("Persisted reference changed")
    row.update(kinematics_accepted=not reasons, recovery_audit=audit, ground_correction=correction,
               rms_landmark_error_m=error, frames=len(candidate.times),
               retargeted_seconds=features["duration_s"], source_complete_duration_s=duration,
               rejected_ticks=int((~candidate.values["valid"].astype(bool)).sum()),
               retarget_version=candidate.metadata["retarget_version"],
               sampling=candidate.metadata["sampling"], calibration=candidate.metadata["calibration"],
               movement_tags=movement_tags(row), motion_features=features,
               span_reference_audit={"version": VERSION, "accepted": not reasons,
                    "rejection_reasons": sorted(set(reasons)), "full_source_duration": True,
                    "source_joint_motion_refit_within_original_audit_gates": True,
                    "controller_success_used": False, "obstacle_clearance_validated": False},
               attempt_reference_path=str(attempt.resolve()), elapsed_seconds=time.monotonic() - started)
    destination = output / "clips" / f"{key}.npz" if not reasons else attempt
    if not reasons:
        row["reference_path"] = str(destination.resolve())
    candidate.metadata.update(row)
    candidate.save(destination)
    loaded = MotionClip.load(destination)
    for field in candidate.values:
        if not np.array_equal(candidate.values[field], loaded.values[field]):
            raise ValueError("Complementary reference reload mismatch")
    return row


def main():
    from freeze_source import freeze_source
    from k1_motion.corpus import recording_split
    from k1_motion.reference_admission import admit_reference, reference_rejections
    from prepare_broad_references import atomic_json, rows, worker_init
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    for part in (args.output, args.output / "attempts", args.output / "clips", args.output / "pool"):
        part.mkdir(parents=True, exist_ok=True)
    base = list(rows(args.base_library / "index.jsonl"))
    existing = {r["source_motion_id"] for r in base}
    inventories = [ROOT / "manifests" / f"{d}.motions.jsonl"
                   for d in ("bandai_namco", "kit_motion_language", "lafan1")]
    previous = list(rows(ROOT / "artifacts/references-v9-expanded-train/index.jsonl"))
    old_splits = {r["capture_group"]: r["split"] for r in previous}
    selected, excluded, seen = [], Counter(), set()
    for inventory in inventories:
        for record in rows(inventory):
            if record["source_motion_id"] in seen:
                continue
            seen.add(record["source_motion_id"])
            family = gap_family(record)
            if family is None or record["source_motion_id"] in existing:
                continue
            if unconfigured_support(record):
                excluded["terrain_support_task_not_configured"] += 1
                continue
            split = recording_split(record["capture_group"])
            if record["capture_group"] in old_splits and old_splits[record["capture_group"]] != split:
                raise ValueError("Existing source split differs")
            # Apply only contextual/task checks; the actual audit follows conversion.
            check = {**record, "split": split, "family": family, "kinematics_accepted": True,
                     "recovery_audit": {"accepted": True}, "rejected_ticks": 0,
                     "reference_path": "pending.npz"}
            reasons = reference_rejections(check)
            if reasons:
                excluded.update(reasons)
                continue
            selected.append({**record, "family": family, "split": split})
    selected.sort(key=lambda r: r["source_motion_id"])
    snapshot, revision = freeze_source(ROOT)
    contract = {"version": VERSION, "source_revision": revision, "source_snapshot": str(snapshot.resolve()),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "inventory_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in inventories},
                "base_manifest_sha256": hashlib.sha256((args.base_library / "index.jsonl").read_bytes()).hexdigest(),
                "base_library": str(args.base_library.resolve()), "selected": selected,
                "exclusions": dict(excluded), "workers": args.workers, "full_source_duration": True,
                "controller_success_used_for_selection": False}
    path = args.output / "campaign.json"
    if path.exists() and json.loads(path.read_text()) != contract:
        raise ValueError("Complementary campaign changed")
    atomic_json(path, contract)
    if args.plan_only:
        print(json.dumps({"selected": len(selected), "families": dict(Counter(r["family"] for r in selected)),
                          "exclusions": dict(excluded)}, indent=2))
        return
    ledger = args.output / "index.jsonl"
    done = list(rows(ledger)) if ledger.exists() else []
    done_ids = {r["source_motion_id"] for r in done}
    expected = {r["source_motion_id"] for r in selected}
    if len(done_ids) != len(done) or not done_ids <= expected:
        raise ValueError("Invalid complementary resume ledger")
    started = time.monotonic()
    with ledger.open("a") as sink, ProcessPoolExecutor(max_workers=args.workers, initializer=worker_init,
                                                      initargs=(snapshot,)) as executor:
        jobs = [executor.submit(convert, (r, str(args.output), revision)) for r in selected
                if r["source_motion_id"] not in done_ids]
        for future in as_completed(jobs):
            row = future.result()
            done.append(row)
            sink.write(json.dumps(row, allow_nan=False) + "\n")
            sink.flush()
            status = {"running": True, "processed": len(done), "expected": len(selected),
                      "accepted": sum(r["kinematics_accepted"] for r in done),
                      "elapsed_seconds": time.monotonic() - started}
            atomic_json(args.output / "status.json", status)
            print(json.dumps(status), flush=True)
    new = []
    for row in sorted(done, key=lambda r: r["id"]):
        if row["kinematics_accepted"]:
            reasons = reference_rejections(row)
            if reasons:
                raise ValueError(f"Complementary admission differs: {reasons}")
            new.append(admit_reference(row))
    pool = base + new
    if len({r["id"] for r in pool}) != len(pool) or len({r["source_motion_id"] for r in pool}) != len(pool):
        raise ValueError("Duplicate complementary source")
    training_groups = {r["capture_group"] for r in pool if r["split"] == "train"}
    if training_groups & {r["capture_group"] for r in pool if r["split"] != "train"}:
        raise ValueError("Complementary split leakage")
    text = "".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in pool)
    index = args.output / "pool/index.jsonl"
    if index.exists() and index.read_text() != text:
        raise ValueError("Frozen complementary pool changed")
    index.write_text(text)
    summary = {"complete": True, "processed": len(done), "base_originals_preserved": len(base),
               "new_admitted_originals": len(new), "pool_splits": dict(Counter(r["split"] for r in pool)),
               "new_splits": dict(Counter(r["split"] for r in new)),
               "new_train_families": dict(Counter(r["family"] for r in new if r["split"] == "train")),
               "pool_train_families": dict(Counter(r["family"] for r in pool if r["split"] == "train")),
               "rejection_reasons": dict(Counter(reason for r in done if not r["kinematics_accepted"]
                    for reason in r.get("span_reference_audit", {}).get("rejection_reasons", ["source_error"]))),
               "elapsed_seconds": time.monotonic() - started,
               "pool_manifest_sha256": hashlib.sha256(text.encode()).hexdigest(),
               "full_source_duration": True, "physics_qualified": False, "obstacle_clearance_validated": False}
    atomic_json(args.output / "summary.json", summary)
    atomic_json(args.output / "status.json", {"running": False, "complete": True, "processed": len(done)})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
