#!/usr/bin/env python3
"""Resumable BONES-SEED proportional-BVH to K1 GMR conversion and validation."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = Path("/mnt/storage/k1-motion/datasets/bones-seed")
DEFAULT_OUTPUT = Path("/mnt/storage/k1-motion/derived/bones-seed-k1-gmr-v1")
MAX_REJECTED_MOTION_FRACTION = 0.20


def stable_id(filename):
    return hashlib.sha256(f"bones-seed/{filename}".encode()).hexdigest()[:20]


def parent_name(filename):
    stem = Path(filename).stem
    return stem[:-2] if stem.endswith("_M") else stem


def controller_family(row):
    fields = (
        "content_name",
        "content_type_of_movement",
        "content_short_description",
        "content_short_description_2",
        "content_technical_description",
        "content_natural_desc_1",
    )
    text = " ".join(str(row.get(key, "")) for key in fields).lower()
    category = str(row.get("category", "")).lower()
    for family, pattern in (
        ("inversion_or_stunt", r"handstand|cartwheel|somersault|(?:back|front)[ -]?flip"),
        ("fall_or_recovery", r"fall|get(?:ting)? up|recover"),
        ("climb", r"climb|ladder|stairs"),
        ("crawl", r"crawl|all fours|hands and knees"),
        ("sit_or_kneel", r"\bsit|kneel"),
        ("object_interaction", r"object|pick(?:ing)? up|carry|throw|pull|push"),
        ("squat", r"squat|squating|crouch|lunge"),
        ("kick", r"\bkick"),
        ("punch", r"punch|strike|striking|boxing"),
        ("jump", r"jump|\bhop"),
        ("transition", r"transition|start(?:ing)?|stop(?:ping)?|accelerat|decelerat"),
        ("run", r"run|jog|sprint|dash"),
        ("walk", r"walk"),
        ("turn", r"turn|pivot"),
        ("dance", r"danc"),
        ("bow", r"\bbow"),
        ("gesture", r"reach|point|gesture|wave"),
        ("idle_stance", r"idle|stance|stand"),
    ):
        if re.search(pattern, text):
            return family
    if category in {"object interaction", "object manipulation"}:
        return "object_interaction"
    return "other"


def split_for_parent(parent, seed="bones-seed-k1-v1"):
    bucket = int.from_bytes(hashlib.sha256(f"{seed}/{parent}".encode()).digest()[:4], "big") % 100
    return "train" if bucket < 80 else "validation" if bucket < 90 else "test"


def shard_for_capture_group(capture_group, shard_count):
    """Keep a complete capture group, including mirrors, on one worker host."""
    digest = hashlib.sha256(f"bones-seed-conversion-v1/{capture_group}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % shard_count


def records(metadata_path):
    import pandas as pd

    frame = pd.read_parquet(metadata_path).fillna("")
    for row in frame.to_dict("records"):
        filename = str(row["filename"])
        parent = parent_name(filename)
        yield {
            "id": stable_id(filename),
            "filename": filename,
            "source_motion_id": f"bones_seed/{filename}",
            "source_path": str(row["move_soma_proportional_path"]),
            "source_frames": int(row["move_duration_frames"]),
            "is_mirror": bool(row["is_mirror"]),
            "original_parent": parent,
            "capture_group": f"bones_seed/{row['take_name']}",
            "actor_uid": str(row["actor_uid"]),
            "take_name": str(row["take_name"]),
            "motion_category": str(row["category"]),
            "motion_package": str(row["package"]),
            "family": controller_family(row),
            "split": split_for_parent(f"bones_seed/{row['take_name']}"),
            "annotations": [
                str(row[key]) for key in (
                    "content_name", "content_short_description", "content_natural_desc_1"
                ) if str(row[key])
            ],
        }


def validate_mirror_lineage(source):
    originals = {row["original_parent"]: row for row in source if not row["is_mirror"]}
    mirrors = [row for row in source if row["is_mirror"]]
    for mirror in mirrors:
        original = originals.get(mirror["original_parent"])
        if original is None:
            raise ValueError(f"Mirror has no original: {mirror['filename']}")
        for key in ("capture_group", "split", "motion_category", "motion_package", "family", "actor_uid"):
            if mirror[key] != original[key]:
                raise ValueError(f"Mirror lineage mismatch for {mirror['filename']}: {key}")
    return {
        "originals": len(originals),
        "mirrors": len(mirrors),
        "originals_without_mirror": len(originals) - len(mirrors),
    }


def _convert(job):
    record, dataset_root, output, max_seconds = job
    sys.path.insert(0, str(ROOT / "src"))
    from k1_motion.adapters import bvh_frames
    from k1_motion.contracts import MotionClip
    from k1_motion.robot import K1Model
    from k1_motion.streaming import retarget_at_control_rate

    dataset_root, output = Path(dataset_root), Path(output)
    base = {
        **record,
        "dataset": "bones_seed",
        "physics_qualified": False,
        "human_reviewed": False,
        "training_eligible": False,
        "valid_motion_contract": "zero rejected control ticks",
    }
    try:
        path = dataset_root / record["source_path"]
        frames = list(
            bvh_frames(
                path, "bones_seed", record["source_motion_id"], max_seconds, target_hz=50.0
            )
        )
        if len(frames) < 2:
            raise ValueError("Too few canonical human frames")
        human = {
            "times": np.array([f.source_time for f in frames]),
            "positions": np.stack([f.positions for f in frames]),
            "orientations": np.stack([f.orientations for f in frames]),
        }
        robot = K1Model()
        clip, reports = retarget_at_control_rate(robot, human, base)
        invalid = np.array([not bool(v) for v in clip.values["valid"]], dtype=bool)
        invalid_fraction = float(invalid.mean())
        reason_ticks = Counter(
            reason for report in reports for reason in set(report["rejection_reasons"])
        )
        # A motion containing a known-invalid tick is not valid training
        # supervision. The 20% allowance applies to whole rejected motions in a
        # type, never to collision/penetration/error frames inside a kept clip.
        accepted = not invalid.any()
        base.update({
            "frames": len(clip.times),
            "source_duration_seconds": float(human["times"][-1] - human["times"][0]),
            "retargeted_seconds": float(clip.times[-1] - clip.times[0]),
            "valid_ticks": int((~invalid).sum()),
            "rejected_ticks": int(invalid.sum()),
            "rejected_tick_fraction": invalid_fraction,
            "kinematics_accepted": accepted,
            "rejection_counts": dict(reason_ticks),
            "rejection_count_unit": "control_ticks",
            "rms_landmark_error_m": float(np.mean([r["rms_landmark_error_m"] for r in reports])),
            "retarget_p95_ms": float(np.percentile([r["seconds"] for r in reports], 95) * 1000),
            "model_signature": robot.signature,
            "retarget_version": clip.metadata["retarget_version"],
            "sampling": clip.metadata["sampling"],
            "calibration": clip.metadata["calibration"],
            "reference_path": f"clips/{record['id']}.npz",
        })
        clip.metadata.update(base)
        clip.save(output / base["reference_path"])
        reloaded = MotionClip.load(output / base["reference_path"])
        if len(reloaded.times) != len(clip.times) or reloaded.metadata["source_motion_id"] != record["source_motion_id"]:
            raise ValueError("Atomic output reload did not preserve the clip contract")
        return base
    except (ValueError, KeyError, IndexError) as error:
        # Data/fit failures remain in the denominator. Programming, I/O and
        # resource failures propagate and stop the campaign.
        return {
            **base,
            "kinematics_accepted": False,
            "rejected_tick_fraction": 1.0,
            "error": f"{type(error).__name__}: {error}",
        }


def load_completed(index_path):
    done = {}
    if index_path.exists():
        for line in index_path.read_text().splitlines():
            row = json.loads(line)
            done[row["id"]] = row
    return done


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bind_campaign(output, dataset_root, metadata, args):
    sys.path.insert(0, str(ROOT / "src"))
    from k1_motion.reference_velocity_clock import PLAYBACK_RETARGET_VERSION
    from k1_motion.robot import K1Model

    archive = dataset_root / "soma_proportional.tar.gz"
    contract = {
        "version": 1,
        "dataset_repo": "bones-studio/seed",
        "metadata_sha256": file_sha256(metadata),
        # Avoid an extra 45.5-GB read: the immutable Hub artifact is bound by
        # its final size/mtime and the metadata paths; the download client has
        # already verified the content-addressed transfer.
        "source_archive_size": archive.stat().st_size,
        "source_archive_mtime_ns": archive.stat().st_mtime_ns,
        "converter_sha256": file_sha256(Path(__file__)),
        "adapter_sha256": file_sha256(ROOT / "src/k1_motion/adapters.py"),
        "retarget_sha256": file_sha256(ROOT / "src/k1_motion/retarget.py"),
        "streaming_sha256": file_sha256(ROOT / "src/k1_motion/streaming.py"),
        "retarget_version": PLAYBACK_RETARGET_VERSION,
        "model_signature": K1Model().signature,
        "workers": args.workers,
        "max_seconds": args.max_seconds,
        "limit": args.limit,
        "canary_per_category": args.canary_per_category,
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "valid_motion_contract": "zero rejected control ticks",
        "minimum_type_acceptance": 0.80,
    }
    path = output / "campaign.json"
    if path.exists():
        prior = json.loads(path.read_text())
        if prior != contract:
            raise ValueError("Campaign contract changed; use a new output version instead of mixing results")
    else:
        path.write_text(json.dumps(contract, indent=2) + "\n")
    return contract


def summarize(rows, expected):
    by_category = defaultdict(list)
    by_family = defaultdict(list)
    # Mirrors are augmentations of originals. They are processed and audited,
    # but may not make an acceptance denominator easier to pass.
    originals = [row for row in rows if not row.get("is_mirror", False)]
    for row in originals:
        by_category[row["motion_category"]].append(row)
        by_family[row["family"]].append(row)

    def groups(values):
        result = {}
        for name, members in sorted(values.items()):
            accepted = sum(bool(r["kinematics_accepted"]) for r in members)
            fraction = accepted / len(members)
            result[name] = {
                "motions": len(members),
                "accepted": accepted,
                "rejected": len(members) - accepted,
                "accepted_fraction": fraction,
                "passes_80_percent_gate": fraction >= 1 - MAX_REJECTED_MOTION_FRACTION,
            }
        return result

    categories, families = groups(by_category), groups(by_family)
    complete = len(rows) == expected
    return {
        "expected_metadata_rows": expected,
        "processed_rows": len(rows),
        "complete": complete,
        "original_motions": len(originals),
        "mirrored_augmentations": len(rows) - len(originals),
        "accepted_motions": sum(bool(r["kinematics_accepted"]) for r in rows),
        "rejected_motions": sum(not bool(r["kinematics_accepted"]) for r in rows),
        "accepted_original_motions": sum(bool(r["kinematics_accepted"]) for r in originals),
        "rejected_original_motions": sum(not bool(r["kinematics_accepted"]) for r in originals),
        "valid_motion_gate": "zero rejected control ticks",
        "per_type_gate": "accepted original motions >= 80 percent",
        "categories": categories,
        "controller_families": families,
        "all_categories_pass": complete and all(v["passes_80_percent_gate"] for v in categories.values()),
        "all_controller_families_pass": complete and all(v["passes_80_percent_gate"] for v in families.values()),
        "physics_qualified_hours": 0.0,
        "scope": "Causal GMR and static MuJoCo kinematic/contact validation; dynamic physics qualification remains separate.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--max-seconds", type=float, help="smoke/canary truncation only")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--canary-per-category", type=int)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    if args.workers < 1 or (args.max_seconds is not None and args.max_seconds <= 0):
        raise ValueError("workers and max-seconds must be positive")
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index must be in [0, shard-count)")
    if args.canary_per_category is not None and args.shard_count != 1:
        raise ValueError("Run the global canary before partitioning production conversion")
    if args.max_seconds is not None and args.limit is None and args.canary_per_category is None:
        raise ValueError("Production conversion cannot truncate motions with --max-seconds")
    metadata = args.dataset_root / "metadata/seed_metadata_v004.parquet"
    source = list(records(metadata))
    lineage = validate_mirror_lineage(source)
    if args.canary_per_category is not None:
        if args.canary_per_category < 1:
            raise ValueError("canary-per-category must be positive")
        category_counts, family_counts = Counter(), Counter()
        canary_by_id = {}
        for row in source:
            if row["is_mirror"]:
                continue
            if category_counts[row["motion_category"]] < args.canary_per_category:
                canary_by_id[row["id"]] = row
                category_counts[row["motion_category"]] += 1
            if family_counts[row["family"]] < args.canary_per_category:
                canary_by_id[row["id"]] = row
                family_counts[row["family"]] += 1
        source = list(canary_by_id.values())
    elif args.shard_count > 1:
        source = [
            row for row in source
            if shard_for_capture_group(row["capture_group"], args.shard_count) == args.shard_index
        ]
    if args.limit is not None:
        source = source[: args.limit]
    missing = [r["source_path"] for r in source if not (args.dataset_root / r["source_path"]).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} source BVH files are missing; first: {missing[0]}")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "clips").mkdir(exist_ok=True)
    contract = bind_campaign(args.output, args.dataset_root, metadata, args)
    index = args.output / "index.jsonl"
    completed = load_completed(index)
    pending = [r for r in source if r["id"] not in completed]
    started = time.monotonic()
    with index.open("a") as stream, ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = ((r, str(args.dataset_root), str(args.output), args.max_seconds) for r in pending)
        # Small batches amortize process-queue overhead while retaining enough
        # load balancing for the dataset's widely varying clip durations.
        for count, row in enumerate(pool.map(_convert, jobs, chunksize=2), 1):
            completed[row["id"]] = row
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            if count % 25 == 0:
                print(json.dumps({"converted_this_run": count, "remaining": len(pending) - count}), flush=True)
    ordered = [completed[r["id"]] for r in source]
    report = summarize(ordered, len(source))
    report.update({
        "dataset_root": str(args.dataset_root),
        "output": str(args.output),
        "workers": args.workers,
        "campaign_contract": contract,
        "mirror_lineage": lineage,
        "elapsed_seconds_this_run": time.monotonic() - started,
    })
    (args.output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    if not (report["all_categories_pass"] and report["all_controller_families_pass"]):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
