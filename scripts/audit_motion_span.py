#!/usr/bin/env python3
"""Compare measured before/after span and freeze representative development panels."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]


def rows(path):
    with Path(path).open() as stream:
        for line in stream:
            if not line.endswith("\n"):
                raise ValueError(f"Incomplete ledger: {path}")
            yield json.loads(line)


def feature_key(row):
    path = Path(row["reference_path"])
    stat = path.stat()
    return {"id": row["id"], "path": str(path.resolve()), "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "model_signature": row["model_signature"]}


def measure_one(row):
    from k1_motion.contracts import MotionClip
    from k1_motion.motion_coverage import measure_motion
    from k1_motion.recovery_geometry import recovery_model
    from k1_motion.reference_admission import ground_rl_accepted
    from k1_motion.low_pose_contract import low_pose_accepted
    clip = MotionClip.load(row["reference_path"])
    spec = recovery_model("audit")
    for field in ("id", "family", "capture_group", "split", "model_signature", "source_motion_id"):
        if clip.metadata[field] != row[field]:
            raise ValueError(f"Payload/manifest mismatch: {row['id']} {field}")
    if row["model_signature"] != spec.signature or not clip.values["valid"].all():
        raise ValueError("Invalid model or reference ticks")
    if clip.source_times is None or np.any(clip.source_times > clip.times + 1e-12):
        raise ValueError("Missing/noncausal source clock")
    if not np.allclose(np.diff(clip.times), spec.control_dt, rtol=0, atol=1e-12):
        raise ValueError("Changed control cadence")
    if row.get("low_pose_reference_audit"):
        if not low_pose_accepted(row) or not low_pose_accepted(clip.metadata):
            raise ValueError("Low-pose contract mismatch")
    elif not row["kinematics_accepted"]:
        if not ground_rl_accepted(row) or row["rl_reference_audit"] != clip.metadata.get("rl_reference_audit"):
            raise ValueError("Bounded-ground contract mismatch")
    if row.get("span_reference_audit"):
        if (not row["span_reference_audit"]["accepted"] or row["span_reference_audit"]
                != clip.metadata.get("span_reference_audit") or row["recovery_audit"]
                != clip.metadata.get("recovery_audit")):
            raise ValueError("Broad motion audit/payload mismatch")
    features = measure_motion(clip, spec.limits)
    if "motion_features" in row and features != row["motion_features"]:
        raise ValueError("Independent saved-payload feature recomputation differs")
    return {"key": feature_key(row), "features": features,
            "maximum_source_age_s": float(np.max(clip.times - clip.source_times))}


def select_panel(rows_, features, split, count, excluded_groups=()):
    from k1_motion.reference_admission import take_family
    eligible = [r for r in rows_ if r["split"] == split
                and take_family(r["capture_group"]) not in excluded_groups]
    # Choose distinct takes first; never use policy outcomes to choose easy clips.
    eligible.sort(key=lambda r: r["id"])
    selected, counts, groups = {}, Counter(), set()
    for row in eligible:
        group = take_family(row["capture_group"])
        if group not in groups and counts[row["family"]] < count:
            selected[row["id"]] = row
            counts[row["family"]] += 1
            groups.add(group)
    if split == "train":
        # Add event extremes to exercise combinations, not only semantic labels.
        for event in ("fast_arm_motion", "arms_and_legs_active", "arms_while_travelling",
                      "backward_travel", "leftward_travel", "rightward_travel", "airborne_proxy",
                      "high_foot_lift_proxy", "low_pelvis"):
            candidates = sorted(eligible, key=lambda r: (-features[r["id"]]["features"]
                                                         ["event_longest_s"][event], r["id"]))
            for row in candidates[:2]:
                if features[row["id"]]["features"]["event_longest_s"][event] >= .1:
                    selected[row["id"]] = row
    return list(selected.values())


def main():
    from convert_bones_seed import records
    from k1_motion.motion_coverage import COVERAGE_VERSION, CoverageAccumulator
    from k1_motion.corpus import recording_split
    from k1_motion.reference_admission import reference_rejections, take_family
    from k1_motion.robot import K1Model
    from prepare_broad_references import atomic_json
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--base-library", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    base = list(rows(args.base_library / "index.jsonl"))
    pool = list(rows(args.library / "index.jsonl"))
    if pool[:len(base)] != base:
        raise ValueError("Existing base rows changed")
    if len({r["id"] for r in pool}) != len(pool) or any(r.get("is_mirror") for r in pool):
        raise ValueError("Duplicate original or augmentation in pool")
    registry = list(records(args.metadata))
    for dataset in ("bandai_namco", "kit_motion_language", "lafan1"):
        registry.extend({"capture_group": r["capture_group"], "split": recording_split(r["capture_group"])}
                        for r in rows(ROOT / "manifests" / f"{dataset}.motions.jsonl"))
    held_out = {take_family(r["capture_group"]) for r in registry if r["split"] != "train"}
    test_groups = {take_family(r["capture_group"]) for r in registry if r["split"] == "test"}
    for row in pool:
        reasons = reference_rejections(row, held_out)
        if reasons or row["training_eligible"] != (row["split"] == "train"):
            raise ValueError(f"Invalid admission: {row['id']} {reasons}")
    train = [r for r in pool if r["split"] == "train"]
    cache = {item["key"]["id"]: item for item in rows(args.feature_cache)} if args.feature_cache else {}
    features = {r["id"]: cache[r["id"]] for r in train if r["id"] in cache
                and cache[r["id"]]["key"] == feature_key(r)
                and cache[r["id"]]["features"]["version"] == COVERAGE_VERSION}
    cached_count = len(features)
    with (args.output / "features.jsonl").open("w") as sink:
        for value in features.values():
            sink.write(json.dumps(value, allow_nan=False) + "\n")
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for value in executor.map(measure_one, [r for r in train if r["id"] not in features], chunksize=8):
                features[value["key"]["id"]] = value
                sink.write(json.dumps(value, allow_nan=False) + "\n")
                if len(features) % 1000 == 0:
                    print(json.dumps({"measured_train": len(features), "expected": len(train)}), flush=True)
    robot = K1Model()
    before, after = CoverageAccumulator(robot.limits), CoverageAccumulator(robot.limits)
    base_ids = {r["id"] for r in base}
    for row in train:
        f = features[row["id"]]["features"]
        after.add(row, f)
        if row["id"] in base_ids:
            before.add(row, f)
    old, new = before.report(), after.report()
    report = {"version": COVERAGE_VERSION, "complete": True,
              "library": str(args.library.resolve()), "base_library": str(args.base_library.resolve()),
              "base_manifest_sha256": hashlib.sha256((args.base_library / "index.jsonl").read_bytes()).hexdigest(),
              "manifest_sha256": hashlib.sha256((args.library / "index.jsonl").read_bytes()).hexdigest(),
              "model_signature": robot.signature, "before": old, "after": new,
              "preserved_base_originals": len(base), "added_originals": len(pool) - len(base),
              "splits": dict(Counter(r["split"] for r in pool)),
              "feature_payloads_reused_from_exact_stat_bound_cache": cached_count,
              "maximum_source_age_s": max(f["maximum_source_age_s"] for f in features.values()),
              "new_joint_bins_with_at_least_three_take_families": sum(
                  sum(a >= 3 and b < 3 for a, b in zip(new["joints"][j]["bin_take_families"],
                                                       old["joints"][j]["bin_take_families"]))
                  for j in new["joints"]),
              "missing_source_intents": [tag for tag in ("boxing_striking", "shadowboxing_explicit", "dance", "arm_reach_gesture",
                    "jump_hop", "kick", "kneel", "crawl", "step_over")
                    if tag not in new["source_intent_tags"]],
              "training_families_below_ten_related_takes_diagnostic_only": [f for f, v in new["families"].items()
                                                                          if v["take_families"] < 10],
              "split_leakage": 0, "duplicate_originals": 0, "invalid_retained_ticks": 0,
              "elapsed_seconds": time.monotonic() - started, "physics_qualified": False,
              "controller_success_used_for_selection": False, "obstacle_clearance_validated": False,
              "limitations": ["Marginal joint bins do not prove coverage of all coordinated motion combinations",
                              "Metadata intent and kinematic proxies are not dynamic task success",
                              "No manipulation, climbing, obstacle perception or obstacle-contact task was added",
                              "Crawling and other unaudited hand/floor support remain open coverage gaps"]}
    atomic_json(args.output / "report.json", report)
    train_panel = select_panel(pool, features, "train", 6)
    validation = select_panel(pool, features, "validation", 4, test_groups)
    (args.output / "canary").mkdir()
    (args.output / "canary/index.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n"
                                                           for r in train_panel + validation))
    atomic_json(args.output / "validation-panel.json", [dict(r, cohort="broad_" + r["family"]) for r in validation])
    atomic_json(args.output / "panels.json", {"train_originals": len(train_panel),
                "validation_originals": len(validation),
                "train_families": dict(Counter(r["family"] for r in train_panel)),
                "validation_families": dict(Counter(r["family"] for r in validation)),
                "held_out_test_used": False, "controller_outcome_used": False})
    print(json.dumps({k: report[k] for k in ("complete", "splits", "added_originals", "missing_source_intents",
        "new_joint_bins_with_at_least_three_take_families", "elapsed_seconds")}, indent=2))


if __name__ == "__main__":
    main()
