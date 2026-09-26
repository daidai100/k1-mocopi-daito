#!/usr/bin/env python3
"""Repair complete foot-supported references and freeze an expanded RL library.

Defaults to the original walking workflow. Other reviewed families require an
explicit --families selection and use a separately versioned admission policy.
"""
# ruff: noqa: E402
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from convert_bones_seed import records
from k1_motion.contracts import MotionClip
from k1_motion.ground_reference import (GROUND_REFERENCE_SETTINGS, GROUND_REFERENCE_VERSION,
                                        correct_walking_ground)
from k1_motion.recovery_validation import audit_recovery
from k1_motion.reference_admission import (GROUNDED_RL_FAMILIES, admit_reference,
                                            ground_rl_accepted, grounded_reference_audit,
                                            reference_rejections, take_family, walking_reference_audit)


def read_rows(path):
    with Path(path).open() as stream:
        for line in stream:
            if not line.endswith("\n"):
                raise ValueError(f"Incomplete input ledger: {path}")
            yield json.loads(line)


def atomic_json(path, value):
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def repair_one(job):
    source, original, output = job
    started = time.monotonic()
    output = Path(output)
    row = {**original, "v4_kinematics_accepted": original["kinematics_accepted"],
           "v4_recovery_audit": original.get("recovery_audit"),
           "physics_qualified": False, "training_eligible": False,
           "walking_reference_version": GROUND_REFERENCE_VERSION,
           "ground_reference_version": GROUND_REFERENCE_VERSION}
    audit = original.get("recovery_audit", {})
    repairable = audit.get("rejection_reasons") == ["ground_penetration_on_command_path"]
    if not original["kinematics_accepted"] and not repairable:
        row.update(walking_status="other_failures_retained", kinematics_accepted=False,
                   ground_reference_status="other_failures_retained")
        row.pop("reference_path", None)
        return row
    key = "reference_path" if original["kinematics_accepted"] else "attempt_reference_path"
    source_path = Path(source) / original[key]
    clip = MotionClip.load(source_path)
    for field in ("id", "family", "capture_group", "split", "source_motion_id", "model_signature", "is_mirror"):
        if clip.metadata[field] != original[field]:
            raise ValueError(f"Ground reference payload/ledger mismatch: {field}")
    if not clip.values["valid"].all() or original["rejected_ticks"]:
        raise ValueError("Ground candidate has invalid reference ticks")
    row["source_reference_path"] = str(source_path.resolve())
    if original["kinematics_accepted"]:
        if not audit.get("accepted"):
            raise ValueError("Original acceptance lacks independent audit")
        row["walking_status"] = "v4_pass_preserved"
    else:
        candidate, correction = correct_walking_ground(clip)
        # Triangle inequality bounds the increase of each landmark RMS error;
        # no source frame is refit, removed, slowed, or read from the future.
        error = original["rms_landmark_error_m"] + correction["mean_landmark_error_increase_bound_m"]
        attempt_path = output / "attempts" / f"{row['id']}.npz"
        candidate.metadata.update(kinematics_accepted=False, training_eligible=False)
        candidate.save(attempt_path)
        candidate = MotionClip.load(attempt_path)
        audit = audit_recovery(candidate, clip, [{"rms_landmark_error_m": error}], original,
                               ground_profile=True)
        if original["family"] != "walk":
            # Rejected V4 attempts retain the original V3 error in their payload
            # metadata. Preserve the cumulative human-tracking budget, rather
            # than grant another 15 mm on top of a V4 regression.
            baseline_error = clip.metadata["rms_landmark_error_m"]
            if not np.isfinite(baseline_error):
                raise ValueError("Missing finite original human-tracking error")
            audit["original_human_tracking_baseline_m"] = baseline_error
            if error > baseline_error + audit["gates"]["max_mean_landmark_error_increase_m"]:
                if "human_tracking_regression" not in audit["rejection_reasons"]:
                    audit["rejection_reasons"].append("human_tracking_regression")
                audit["accepted"] = False
        if correction["nonfoot_ground_contact_samples"]:
            audit["rejection_reasons"].append("nonfoot_ground_contact")
            audit["accepted"] = False
        row.update(ground_correction=correction, recovery_audit=audit,
                   attempt_reference_path=str(attempt_path.resolve()),
                   kinematics_accepted=audit["accepted"], rms_landmark_error_m=error,
                   human_tracking_measurement="conservative_upper_bound_after_vertical_translation",
                   walking_status="ground_corrected" if audit["accepted"] else "ground_repair_rejected")
        row["rl_reference_audit"] = (walking_reference_audit(audit) if row["family"] == "walk"
                                     else grounded_reference_audit(audit, row["family"]))
        if not audit["accepted"] and ground_rl_accepted(row):
            row["walking_status"] = "bounded_ground_rl_reference"
        clip = candidate
    row.pop("reference_path", None)
    row["ground_reference_status"] = row["walking_status"]
    row["elapsed_seconds"] = time.monotonic() - started
    if row["kinematics_accepted"] or ground_rl_accepted(row):
        destination = output / "clips" / f"{row['id']}.npz"
        row["reference_path"] = str(destination.resolve())
        clip.metadata.update(row)
        clip.save(destination)
        loaded = MotionClip.load(destination)
        for key in clip.values:
            if not np.array_equal(loaded.values[key], clip.values[key]):
                raise ValueError("Ground reference artifact reload mismatch")
    return row


def pilot_selection(items, count):
    result, used = [], set()
    counts = Counter()
    for source, row in sorted(items, key=lambda item: item[1]["id"]):
        group = take_family(row["capture_group"])
        audit = row.get("recovery_audit", {})
        status = ("accepted" if row["kinematics_accepted"] else "ground_only"
                  if audit.get("rejection_reasons") == ["ground_penetration_on_command_path"] else "other")
        bucket = (row["family"], "train" if row["split"] == "train" else "held_out", status)
        if group in used or counts[bucket] >= count:
            continue
        used.add(group)
        counts[bucket] += 1
        result.append((source, row))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--base-library", type=Path)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--families", nargs="+", default=["walk"],
                        choices=sorted(GROUNDED_RL_FAMILIES | {"walk"}))
    parser.add_argument("--pilot-per-stratum", type=int, help="Bounded, take-disjoint development panel")
    args = parser.parse_args()
    if args.workers < 1 or args.pilot_per_stratum is not None and args.pilot_per_stratum < 1:
        raise ValueError("Invalid worker or pilot count")
    if args.base_library and args.pilot_per_stratum:
        raise ValueError("A partial pilot cannot replace family coverage in a full library")
    families = set(args.families)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name in ("clips", "attempts", "pool"):
        (output / name).mkdir(exist_ok=True)
    items, seen, contracts = [], set(), []
    for source in args.source:
        source = source.resolve()
        contracts.append({"root": str(source),
                          "campaign": json.loads((source / "campaign.json").read_text())})
        for row in read_rows(source / "index.jsonl"):
            if row.get("is_mirror") or row["family"] not in families:
                continue
            if row["id"] in seen:
                raise ValueError("Duplicate original source motion")
            seen.add(row["id"])
            items.append((str(source), row))
    registry = list(records(args.metadata))
    expected = {r["id"] for r in registry if r["family"] in families and not r["is_mirror"]}
    if not args.pilot_per_stratum and seen != expected:
        raise ValueError(f"Family source coverage mismatch: {len(seen)} versus {len(expected)}")
    if args.pilot_per_stratum:
        items = pilot_selection(items, args.pilot_per_stratum)
    sources = [ROOT / "scripts/prepare_walking_references.py", ROOT / "src/k1_motion/ground_reference.py",
               ROOT / "src/k1_motion/recovery_validation.py", ROOT / "src/k1_motion/reference_admission.py",
               ROOT / "src/k1_motion/robot.py", ROOT / "src/k1_motion/recovery_geometry.py"]
    contract = {"version": GROUND_REFERENCE_VERSION, "settings": GROUND_REFERENCE_SETTINGS,
                "families": sorted(families),
                "sources": contracts, "originals": len(items), "workers": args.workers,
                "pilot_per_stratum": args.pilot_per_stratum,
                "selection": sorted(row["id"] for _, row in items),
                "selection_sha256": hashlib.sha256(json.dumps(
                    sorted(items, key=lambda item: item[1]["id"]), sort_keys=True).encode()).hexdigest(),
                "source_hashes": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sources},
                "metadata_sha256": hashlib.sha256(args.metadata.read_bytes()).hexdigest(),
                "base_library": str(args.base_library.resolve()) if args.base_library else None,
                "base_manifest_sha256": hashlib.sha256((args.base_library / "index.jsonl").read_bytes()).hexdigest()
                if args.base_library else None,
                "controller_success_used_for_selection": False}
    campaign = output / "campaign.json"
    if campaign.exists() and json.loads(campaign.read_text()) != contract:
        raise ValueError("Ground preparation contract changed; use a new output version")
    atomic_json(campaign, contract)
    ledger = output / "index.jsonl"
    done = list(read_rows(ledger)) if ledger.exists() else []
    done_ids = {r["id"] for r in done}
    if len(done_ids) != len(done) or not done_ids <= set(contract["selection"]):
        raise ValueError("Invalid resumed ground-reference ledger")
    with ledger.open("a") as sink, ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(repair_one, (source, row, str(output))): row["id"]
                   for source, row in items if row["id"] not in done_ids}
        last = 0
        for future in as_completed(futures):
            row = future.result()  # Operational failures never become data rejections.
            sink.write(json.dumps(row, allow_nan=False) + "\n")
            sink.flush()
            done.append(row)
            if time.monotonic() - last > 15:
                progress = {"processed": len(done), "expected": len(items),
                            "accepted": sum(r["kinematics_accepted"] or ground_rl_accepted(r) for r in done),
                            "running": True}
                atomic_json(output / "status.json", progress)
                print(json.dumps(progress), flush=True)
                last = time.monotonic()
    held_out = {take_family(r["capture_group"]) for r in registry if r["split"] != "train"}
    pool_rows, rejected = [], Counter()
    for row in sorted(done, key=lambda r: r["id"]):
        reasons = reference_rejections(row, held_out)
        if reasons:
            rejected.update(reasons)
        else:
            pool_rows.append(admit_reference(row))
    selected_pool_rows = pool_rows.copy()
    walk_rows = [r for r in pool_rows if r["family"] == "walk"]
    if args.base_library:
        base_clips = output / "base_clips"
        base_clips.mkdir(exist_ok=True)
        for row in read_rows(args.base_library / "index.jsonl"):
            if row["family"] in families:
                continue
            if reference_rejections(row, held_out):
                raise ValueError("Existing RL base violates preserved admission/split contract")
            source = args.base_library / row["reference_path"]
            target = base_clips / f"{row['id']}.npz"
            if not target.exists():
                shutil.copy2(source, target)
            pool_rows.append({**row, "reference_path": str(target),
                              "frozen_base_reference_path": str(source.resolve())})
    pool_rows.sort(key=lambda r: r["id"])
    if len({r["id"] for r in pool_rows}) != len(pool_rows):
        raise ValueError("Duplicate expanded pool original")
    train_groups = {take_family(r["capture_group"]) for r in pool_rows if r["split"] == "train"}
    if train_groups & held_out:
        raise ValueError("Expanded pool leaks held-out take families")
    index_text = "".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in pool_rows)
    index = output / "pool/index.jsonl"
    if index.exists() and index.read_text() != index_text:
        raise ValueError("Expanded frozen manifest changed")
    index.write_text(index_text)
    corrected = [r for r in done if r["walking_status"] in ("ground_corrected", "bounded_ground_rl_reference")]
    summary = {
        "version": GROUND_REFERENCE_VERSION, "complete": len(done) == len(items),
        "originals": len(done), "walking_originals": sum(r["family"] == "walk" for r in done),
        "v4_accepted": sum(r["v4_kinematics_accepted"] for r in done),
        "strict_geometry_accepted": sum(r["kinematics_accepted"] for r in done),
        "accepted": sum(r["kinematics_accepted"] or ground_rl_accepted(r) for r in done),
        "newly_recovered": len(corrected), "status_counts": dict(Counter(r["walking_status"] for r in done)),
        "remaining_reasons": dict(Counter(reason for r in done
                                           if not r["kinematics_accepted"] and not ground_rl_accepted(r)
                                           for reason in r.get("rl_reference_audit", r.get("recovery_audit", {}))
                                           .get("rejection_reasons", ["source_error"]))),
        "walk_pool_splits": dict(Counter(r["split"] for r in walk_rows)),
        "walk_train_hours": sum(r["retargeted_seconds"] for r in walk_rows if r["split"] == "train") / 3600,
        "walk_train_capture_groups": len({r["capture_group"] for r in walk_rows if r["split"] == "train"}),
        "walk_train_take_families": len({take_family(r["capture_group"]) for r in walk_rows if r["split"] == "train"}),
        "walk_admission_rejections": dict(rejected),
        "admission_rejections": dict(rejected),
        "families": {family: {
            "originals": sum(r["family"] == family for r in done),
            "v4_accepted": sum(r["v4_kinematics_accepted"] for r in done if r["family"] == family),
            "strict_geometry_accepted": sum(r["kinematics_accepted"] for r in done if r["family"] == family),
            "rl_usable": sum(r["kinematics_accepted"] or ground_rl_accepted(r)
                             for r in done if r["family"] == family),
            "status_counts": dict(Counter(r["walking_status"] for r in done if r["family"] == family)),
            "pool_splits": dict(Counter(r["split"] for r in selected_pool_rows if r["family"] == family)),
            "train_hours": sum(r["retargeted_seconds"] for r in selected_pool_rows
                               if r["family"] == family and r["split"] == "train") / 3600,
            "train_take_families": len({take_family(r["capture_group"]) for r in selected_pool_rows
                                        if r["family"] == family and r["split"] == "train"}),
        } for family in sorted(families)},
        "pool_split_counts": dict(Counter(r["split"] for r in pool_rows)),
        "pool_train_families": dict(Counter(r["family"] for r in pool_rows if r["split"] == "train")),
        "pool_manifest_sha256": hashlib.sha256(index_text.encode()).hexdigest(),
        "controller_success_used_for_selection": False,
        "physics_qualified_hours": 0, "hardware_verified": False,
        "max_corrected_lift_m": max((r["ground_correction"]["max_lift_m"] for r in corrected), default=0),
    }
    atomic_json(output / "summary.json", summary)
    atomic_json(output / "status.json", {"processed": len(done), "expected": len(items), "running": False})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
