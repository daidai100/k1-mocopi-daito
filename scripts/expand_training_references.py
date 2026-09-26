#!/usr/bin/env python3
"""Add recording-grouped training diversity while preserving every held-out row."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

from freeze_source import freeze_source


@lru_cache(maxsize=1)
def robot_model():
    from k1_motion.robot import K1Model

    return K1Model()


def additions(parent_rows, inventories, target_per_family, max_per_group):
    from k1_motion.corpus import family_of, record_duration_ok, recording_split

    existing = {r["source_motion_id"] for r in parent_rows}
    counts = Counter(
        r["family"] for r in parent_rows if r["split"] == "train" and r.get("training_eligible", True)
    )
    groups = Counter(
        (r["family"], r["capture_group"])
        for r in parent_rows
        if r["split"] == "train" and r.get("training_eligible", True)
    )
    buckets = defaultdict(list)
    for record in inventories:
        family = family_of(record)
        if (
            family in ("external_support_or_recovery", "unclassified")
            or not record_duration_ok(record)
            or recording_split(record["capture_group"]) != "train"
            or record["source_motion_id"] in existing
        ):
            continue
        buckets[(family, record["capture_group"])].append({**record, "family": family})
    for records in buckets.values():
        records.sort(key=lambda r: (r["duration_seconds"], r["source_motion_id"]))
    selected = []
    for family in sorted({f for f, _ in buckets}):
        while counts[family] < target_per_family:
            eligible = [k for k, v in buckets.items() if k[0] == family and v and groups[k] < max_per_group]
            if not eligible:
                break
            key = min(eligible, key=lambda k: (groups[k], buckets[k][0]["duration_seconds"], k[1]))
            row = buckets[key].pop(0)
            if row["source_motion_id"] in existing:
                continue
            existing.add(row["source_motion_id"])
            selected.append(row)
            groups[key] += 1
            counts[family] += 1
    return selected


def convert(job):
    import numpy as np

    from k1_motion.adapters import bvh_frames, mmm_frames
    from k1_motion.retarget import RETARGET_VERSION
    from k1_motion.robot import ROOT
    from k1_motion.streaming import retarget_at_control_rate

    record, directory, max_seconds = job
    output = Path(directory)
    key = hashlib.sha256(record["source_motion_id"].encode()).hexdigest()[:16]
    base = {
        "id": key,
        "source_motion_id": record["source_motion_id"],
        "capture_group": record["capture_group"],
        "split": "train",
        "dataset": record["dataset"],
        "family": record["family"],
        "source_path": record["path"],
        "annotations": record["annotations"],
        "source_duration_seconds": record["duration_seconds"],
        "physics_qualified": False,
        "human_reviewed": False,
        "training_eligible": True,
        "flat_ground": "candidate_annotation_screen_only",
        "retarget_version": RETARGET_VERSION,
        "source_revision": os.environ["K1_SOURCE_REVISION"],
        "license_dataset": record["dataset"],
        "license_directory": str(ROOT / "licenses" / record["dataset"]),
        "addition": True,
    }
    try:
        path = ROOT / record["path"]
        frames = list(
            bvh_frames(path, record["dataset"], record["source_motion_id"], max_seconds)
            if record["format"] == "BVH"
            else mmm_frames(
                path,
                ROOT / "data/reference/mmmpy_lite/mmmpy_lite/data/models/mmm/mmm.urdf",
                record.get("track_index", 0),
                record["source_motion_id"],
                max_seconds,
            )
        )
        if len(frames) < 2:
            raise ValueError("Too few canonical human frames")
        human = {
            "times": np.array([f.source_time for f in frames]),
            "positions": np.stack([f.positions for f in frames]),
            "orientations": np.stack([f.orientations for f in frames]),
        }
        base["human_path"] = f"human/{key}.npz"
        np.savez_compressed(output / base["human_path"], **human)
        robot = robot_model()
        base["model_signature"] = robot.signature
        clip, reports = retarget_at_control_rate(robot, human, base)
        rejected = Counter(reason for report in reports for reason in report["rejection_reasons"])
        base.update(
            {
                "frames": len(clip.times),
                "retargeted_seconds": float(clip.times[-1] - clip.times[0]),
                "canonical_human_seconds": float(human["times"][-1] - human["times"][0]),
                "kinematics_accepted": not rejected,
                "rejection_counts": dict(rejected),
                "rejection_count_unit": "control_ticks",
                "rms_landmark_error_m": float(np.mean([r["rms_landmark_error_m"] for r in reports])),
                "retarget_p95_ms": float(np.percentile([r["seconds"] for r in reports], 95) * 1000),
                "sampling": clip.metadata["sampling"],
                "calibration": clip.metadata["calibration"],
                "reference_path": f"clips/{key}.npz",
            }
        )
        clip.metadata.update(base)
        clip.save(output / base["reference_path"])
        return base
    except (ValueError, KeyError, IndexError, FileNotFoundError) as error:
        return {**base, "kinematics_accepted": False, "error": f"{type(error).__name__}: {error}"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-per-family", type=int, default=80)
    parser.add_argument("--max-per-group", type=int, default=3)
    parser.add_argument("--max-seconds", type=float, default=20)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if min(args.target_per_family, args.max_per_group, args.max_seconds, args.workers) <= 0:
        raise ValueError("Selection, duration, and worker limits must be positive")
    root = Path(__file__).resolve().parents[1]
    frozen, revision = freeze_source(root)
    sys.path.insert(0, str(frozen))
    os.environ["K1_MOTION_ROOT"] = str(root)
    os.environ["K1_SOURCE_REVISION"] = revision
    from k1_motion.corpus import family_of

    source, output = Path(args.library), Path(args.output)
    parent_index = (source / "index.jsonl").read_bytes()
    parents = list(map(json.loads, parent_index.splitlines()))
    held_out = [r for r in parents if r["split"] != "train"]
    for row in parents:
        if row["split"] == "train" and family_of(row) == "external_support_or_recovery":
            # Kinematic validity is independent of suitability for this task.
            row["training_eligible"] = False
            row["training_exclusion_reason"] = "annotation_requires_external_support_recovery_or_inversion"
    inventories = [
        json.loads(line)
        for dataset in ("bandai_namco", "lafan1", "kit_motion_language")
        for line in (root / "manifests" / f"{dataset}.motions.jsonl").read_text().splitlines()
    ]
    selected = additions(parents, inventories, args.target_per_family, args.max_per_group)
    if {r["capture_group"] for r in selected} & {r["capture_group"] for r in held_out}:
        raise ValueError("An added training recording overlaps a held-out capture group")
    output.mkdir(parents=True, exist_ok=False)
    (output / "human").mkdir()
    (output / "clips").mkdir()
    # Both libraries are immutable. Share existing artifacts rather than redoing
    # validated conversion or copying hundreds of MB between SSD directories.
    for row in parents:
        for key in ("human_path", "reference_path"):
            if key in row:
                try:
                    os.link(source / row[key], output / row[key])
                except OSError:
                    shutil.copyfile(source / row[key], output / row[key])
    (output / "added-selection.json").write_text(json.dumps(selected, indent=2) + "\n")
    started, results = time.monotonic(), list(parents)
    with (output / "index.jsonl").open("x") as stream, ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in parents:
            stream.write(json.dumps(row) + "\n")
        stream.flush()
        for row in pool.map(convert, ((r, str(output), args.max_seconds) for r in selected)):
            results.append(row)
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            if (len(results) - len(parents)) % 25 == 0:
                print(json.dumps({"added_converted": len(results) - len(parents), "additions": len(selected)}), flush=True)
    if [r for r in results if r["split"] != "train"] != held_out:
        raise AssertionError("Held-out rows changed")
    train = [r for r in results if r["split"] == "train" and r["kinematics_accepted"] and r.get("training_eligible", True)]
    report = {
        "parent_library": str(source.resolve()),
        "parent_index_sha256": hashlib.sha256(parent_index).hexdigest(),
        "source_revision": revision,
        "build_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "selection_settings": vars(args),
        "selected": len(results),
        "additions": len(selected),
        "accepted_training_recordings": len(train),
        "accepted_training_capture_groups": len({r["capture_group"] for r in train}),
        "accepted_training_hours": sum(r["retargeted_seconds"] for r in train) / 3600,
        "accepted_train_families": dict(Counter(r["family"] for r in train)),
        "eligible_train_parents_excluded": [r["id"] for r in parents if not r.get("training_eligible", True)],
        "kinematic_rejections": sum(not r["kinematics_accepted"] for r in results),
        "held_out_rows_unchanged": True,
        "held_out_recordings": len(held_out),
        "training_held_out_group_overlap": False,
        "physics_qualified_hours": 0.0,
        "elapsed_seconds": time.monotonic() - started,
        "scope": "Training expansion and annotation screen; kinematic candidates, no physics qualification. Existing evaluation denominators preserved.",
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
