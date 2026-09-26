#!/usr/bin/env python3
"""Append missing whole-body families from complete, audited conversion shards."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
FAMILIES = {"gesture", "bow", "punch", "kick", "jump", "other"}
VERSION = "whole-body-reference-expansion-v1"


def rows(path):
    with Path(path).open() as stream:
        for line in stream:
            if not line.endswith("\n"):
                raise ValueError(f"Incomplete ledger: {path}")
            yield json.loads(line)


def atomic_json(path, value):
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def worker_init(snapshot):
    os.environ["K1_MOTION_ROOT"] = str(ROOT)
    sys.path.insert(0, str(snapshot))
    for name in list(sys.modules):
        if name == "k1_motion" or name.startswith("k1_motion."):
            del sys.modules[name]


def candidate_rejections(row, held_out):
    from k1_motion.reference_admission import reference_rejections
    reasons = reference_rejections(row, held_out)
    ground_only = row.get("recovery_audit", {}).get("rejection_reasons") == [
        "ground_penetration_on_command_path"]
    if ground_only and row.get("attempt_reference_path"):
        reasons = [r for r in reasons if r not in {"retargeting_audit_failed", "missing_reference"}]
    return reasons


def prepare_one(job):
    from k1_motion.contracts import MotionClip
    from k1_motion.ground_reference import correct_walking_ground
    from k1_motion.math3d import rotation
    from k1_motion.motion_coverage import measure_motion, movement_tags
    from k1_motion.recovery_geometry import recovery_model
    from k1_motion.recovery_validation import audit_recovery
    source, original, output = job
    started = time.monotonic()
    key = "reference_path" if original["kinematics_accepted"] else "attempt_reference_path"
    path = Path(source) / original[key]
    clip = MotionClip.load(path)
    robot = recovery_model("audit")
    for field in ("id", "capture_group", "split", "family", "model_signature", "source_motion_id", "is_mirror"):
        if clip.metadata[field] != original[field]:
            raise ValueError(f"Payload/ledger mismatch: {original['id']} {field}")
    if original["kinematics_accepted"] and clip.metadata.get("recovery_audit") != original["recovery_audit"]:
        raise ValueError("Payload audit differs from completed source ledger")
    if (clip.metadata["model_signature"] != robot.signature or not clip.values["valid"].all()
            or original["rejected_ticks"] or clip.source_times is None
            or np.any(clip.source_times > clip.times + 1e-12)
            or not np.allclose(np.diff(clip.times), robot.control_dt, rtol=0, atol=1e-12)):
        raise ValueError("Invalid payload/model/causal clocks")
    if original["frames"] != len(clip.times):
        raise ValueError("Ledger frame count mismatch")
    candidate = clip
    correction = None
    error = original["rms_landmark_error_m"]
    baseline_error = error
    if not original["kinematics_accepted"]:
        candidate, correction = correct_walking_ground(clip)
        error += correction["mean_landmark_error_increase_bound_m"]
        # The attempt payload retains the original pre-recovery human error.
        baseline_error = clip.metadata["rms_landmark_error_m"]
    audit = audit_recovery(candidate, clip, [{"rms_landmark_error_m": error}],
                           {"rms_landmark_error_m": baseline_error}, ground_profile=True)
    features = measure_motion(candidate, robot.limits)
    reasons = list(audit["rejection_reasons"])
    if audit["ground_profile"]["max_nonfoot_penetration_m"] > .0001:
        reasons.append("nonfoot_support_task_requires_separate_audit")
    if correction and correction["nonfoot_ground_contact_samples"]:
        reasons.append("nonfoot_support_task_requires_separate_audit")
    # Do not silently admit a new floor task under an incidental semantic label.
    upright = rotation(candidate.values["root_orientation"]).apply([0, 0, 1])[:, 2]
    if candidate.values["root_position"][:, 2].min() < .22 or upright.min() < .2:
        reasons.append("low_support_task_requires_separate_audit")
    if np.any(candidate.values["joint_position"] < robot.limits[:, 0] - 1e-8) or np.any(
            candidate.values["joint_position"] > robot.limits[:, 1] + 1e-8):
        reasons.append("joint_limits")
    row = {**original, "source_reference_path": str(path.resolve()),
           "source_kinematics_accepted": original["kinematics_accepted"],
           "source_recovery_audit": original["recovery_audit"], "recovery_audit": audit,
           "kinematics_accepted": not reasons, "training_eligible": False, "physics_qualified": False,
           "rms_landmark_error_m": error, "retargeted_seconds": features["duration_s"],
           "movement_tags": movement_tags(original), "motion_features": features,
           "span_reference_audit": {"version": VERSION, "accepted": not reasons,
                "rejection_reasons": sorted(set(reasons)), "joint_motion_unchanged": True,
                "horizontal_motion_unchanged": True, "clocks_unchanged": True,
                "controller_success_used": False, "obstacle_clearance_validated": False},
           "elapsed_seconds": time.monotonic() - started}
    row.pop("reference_path", None)
    row.pop("rl_reference_audit", None)
    if correction:
        row["ground_correction"] = correction
    destination = Path(output) / ("clips" if not reasons else "attempts") / (row["id"] + ".npz")
    row["reference_path" if not reasons else "attempt_reference_path"] = str(destination.resolve())
    candidate.metadata.update(row)
    candidate.save(destination)
    loaded = MotionClip.load(destination)
    for name in candidate.values:
        if not np.array_equal(candidate.values[name], loaded.values[name]):
            raise ValueError("Saved reference changed")
    return row


def plan(args):
    from convert_bones_seed import records
    from freeze_source import freeze_source
    from k1_motion.reference_admission import take_family
    registry = list(records(args.metadata))
    source_registry = {r["id"]: r for r in registry}
    held_out = {take_family(r["capture_group"]) for r in registry if r["split"] != "train"}
    base_rows = list(rows(args.base_library / "index.jsonl"))
    existing = {r["id"] for r in base_rows}
    if len(existing) != len(base_rows):
        raise ValueError("Duplicate base original")
    if {take_family(r["capture_group"]) for r in base_rows if r["split"] == "train"} & held_out:
        raise ValueError("Base training leaks held-out take families")
    selected, seen, sources, exclusions = [], set(), [], Counter()
    for source in args.source:
        source = source.resolve()
        sources.append({"root": str(source), "campaign": json.loads((source / "campaign.json").read_text())})
        for row in rows(source / "index.jsonl"):
            if row["id"] in seen:
                raise ValueError("Duplicate source row")
            seen.add(row["id"])
            metadata = source_registry[row["id"]]
            for field in ("family", "capture_group", "split", "is_mirror", "source_motion_id"):
                if row[field] != metadata[field]:
                    raise ValueError(f"Source registry mismatch: {field}")
            if row["is_mirror"] or row["id"] in existing or row["family"] not in FAMILIES:
                continue
            reasons = candidate_rejections(row, held_out)
            if reasons:
                exclusions.update(f"{row['family']}:{r}" for r in reasons)
                continue
            key = "reference_path" if row["kinematics_accepted"] else "attempt_reference_path"
            relative = Path(row[key])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Expected an in-source relative payload path")
            selected.append({"source": str(source), "row": row})
    if seen != set(source_registry):
        raise ValueError(f"Incomplete source registry: {len(seen)} / {len(source_registry)}")
    snapshot, revision = freeze_source(ROOT)
    selected.sort(key=lambda r: r["row"]["id"])
    return {"version": VERSION, "source_revision": revision, "source_snapshot": str(snapshot.resolve()),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "base_library": str(args.base_library.resolve()),
            "base_manifest_sha256": hashlib.sha256((args.base_library / "index.jsonl").read_bytes()).hexdigest(),
            "metadata_sha256": hashlib.sha256(args.metadata.read_bytes()).hexdigest(),
            "sources": sources, "selected": selected, "exclusions": dict(exclusions),
            "source_rows": len(seen), "expansion_families": sorted(FAMILIES),
            "held_out_take_families": sorted(held_out), "workers": args.workers,
            "append_only": True, "controller_success_used_for_selection": False,
            "contact_policy": "New additions: strict 500 Hz foot-support/flight geometry; prior audited knees retained"}


def main():
    from k1_motion.reference_admission import admit_reference, reference_rejections, take_family
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--base-library", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("Invalid worker count")
    args.output = args.output.resolve()
    for part in (args.output, args.output / "clips", args.output / "attempts", args.output / "pool"):
        part.mkdir(parents=True, exist_ok=True)
    contract = plan(args)
    path = args.output / "campaign.json"
    if path.exists() and json.loads(path.read_text()) != contract:
        raise ValueError("Expansion contract changed; use a new output version")
    atomic_json(path, contract)
    for i, source in enumerate(contract["sources"]):
        paths = [item["row"]["reference_path" if item["row"]["kinematics_accepted"] else "attempt_reference_path"]
                 for item in contract["selected"] if item["source"] == source["root"]]
        (args.output / f"source-{i}-files.txt").write_text("".join(p + "\n" for p in paths))
    if args.plan_only:
        print(json.dumps({"selected": len(contract["selected"]), "by_family": dict(Counter(
            x["row"]["family"] for x in contract["selected"])), "exclusions": contract["exclusions"]}, indent=2))
        return
    ledger = args.output / "index.jsonl"
    done = list(rows(ledger)) if ledger.exists() else []
    done_ids = {r["id"] for r in done}
    expected = {x["row"]["id"] for x in contract["selected"]}
    if len(done_ids) != len(done) or not done_ids <= expected:
        raise ValueError("Invalid resumed ledger")
    started, last = time.monotonic(), 0.
    with ledger.open("a") as sink, ProcessPoolExecutor(max_workers=args.workers, initializer=worker_init,
            initargs=(contract["source_snapshot"],)) as workers:
        jobs = [workers.submit(prepare_one, (x["source"], x["row"], str(args.output)))
                for x in contract["selected"] if x["row"]["id"] not in done_ids]
        for future in as_completed(jobs):
            row = future.result()
            sink.write(json.dumps(row, allow_nan=False) + "\n")
            sink.flush()
            done.append(row)
            if time.monotonic() - last >= 15:
                status = {"running": True, "processed": len(done), "expected": len(expected),
                          "accepted": sum(r["kinematics_accepted"] for r in done),
                          "elapsed_seconds": time.monotonic() - started}
                atomic_json(args.output / "status.json", status)
                print(json.dumps(status), flush=True)
                last = time.monotonic()
    held_out = set(contract["held_out_take_families"])
    new_rows = []
    for row in sorted(done, key=lambda r: r["id"]):
        if row["kinematics_accepted"]:
            reasons = reference_rejections(row, held_out)
            if reasons:
                raise ValueError(f"Prepared reference admission changed: {reasons}")
            new_rows.append(admit_reference(row))
    base = list(rows(args.base_library / "index.jsonl"))
    pool = base + new_rows
    if len({r["id"] for r in pool}) != len(pool):
        raise ValueError("Duplicate pool original")
    if {take_family(r["capture_group"]) for r in pool if r["split"] == "train"} & held_out:
        raise ValueError("Held-out training leakage")
    index_text = "".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in pool)
    index = args.output / "pool/index.jsonl"
    if index.exists() and index.read_text() != index_text:
        raise ValueError("Frozen pool changed")
    index.write_text(index_text)
    summary = {"complete": True, "base_originals_preserved": len(base), "processed": len(done),
               "new_admitted_originals": len(new_rows), "pool_originals": len(pool),
               "pool_splits": dict(Counter(r["split"] for r in pool)),
               "pool_train_families": dict(Counter(r["family"] for r in pool if r["split"] == "train")),
               "added": {f: {"processed": sum(r["family"] == f for r in done),
                    "splits": dict(Counter(r["split"] for r in new_rows if r["family"] == f)),
                    "train_take_families": len({take_family(r["capture_group"]) for r in new_rows
                                                if r["family"] == f and r["split"] == "train"})}
                    for f in sorted(FAMILIES)},
               "rejection_reasons": dict(Counter(reason for r in done for reason in
                    r["span_reference_audit"]["rejection_reasons"])),
               "pool_manifest_sha256": hashlib.sha256(index_text.encode()).hexdigest(),
               "elapsed_seconds": time.monotonic() - started,
               "controller_success_used_for_selection": False, "physics_qualified": False,
               "obstacle_clearance_validated": False}
    atomic_json(args.output / "summary.json", summary)
    atomic_json(args.output / "status.json", {"running": False, "complete": True, "processed": len(done)})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
