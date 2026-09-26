#!/usr/bin/env python3
"""Append genuine knee-support and squat references without changing old exports."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]


def rows(path):
    with Path(path).open() as stream:
        for line in stream:
            if not line.endswith("\n"):
                raise ValueError(f"Incomplete ledger: {path}")
            yield json.loads(line)


def atomic_json(path, value):
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def worker_init(snapshot):
    os.environ["K1_MOTION_ROOT"] = str(ROOT)
    sys.path.insert(0, str(snapshot))
    for name in list(sys.modules):
        if name == "k1_motion" or name.startswith("k1_motion."):
            del sys.modules[name]


def convert(job):
    from k1_motion.adapters import bvh_frames
    from k1_motion.contracts import MotionClip
    from k1_motion.low_pose import LOW_POSE_SOURCE, LOW_POSE_VERSION, correct_low_pose_ground, retarget_low_pose
    from k1_motion.low_pose_contract import low_pose_reference_audit
    from k1_motion.low_pose_validation import audit_low_pose
    from k1_motion.recovery_geometry import recovery_model
    record, dataset, output, revision = job
    started = time.monotonic()
    output = Path(output)
    family = "squat" if record["family"] == "squat" else "kneel"
    base = {**record, "source_family": record["family"], "family": family,
            "dataset": "bones_seed", "retarget_version": LOW_POSE_VERSION,
            "source_adapter": LOW_POSE_SOURCE, "source_revision": revision,
            "physics_qualified": False, "training_eligible": False, "human_reviewed": False,
            "kinematics_accepted": False}
    raw = Path(dataset)/record["source_path"]
    staged = output/"source-cache"/f"{record['id']}.bvh"
    if not staged.exists():
        shutil.copy2(raw, staged)
    try:
        frames = list(bvh_frames(staged, "bones_seed_v2", record["source_motion_id"], target_hz=50))
        if len(frames) < 2:
            raise ValueError("Too few source frames")
        clip, _ = retarget_low_pose(recovery_model("retarget"), frames, base)
    except ValueError as error:
        return {**base, "source_error": str(error), "elapsed_seconds": time.monotonic()-started}
    clip, correction = correct_low_pose_ground(clip)
    attempt = output/"attempts"/f"{record['id']}.npz"
    clip.save(attempt)
    loaded = MotionClip.load(attempt)
    audit = audit_low_pose(loaded, frames, family)
    base.update(model_signature=clip.metadata["model_signature"], calibration=clip.metadata["calibration"],
                sampling=clip.metadata["sampling"], ground_correction=correction,
                low_pose_settings=clip.metadata["low_pose_settings"],
                frames=len(clip.times), retargeted_seconds=float(clip.times[-1]-clip.times[0]),
                rejected_ticks=int((~clip.values["valid"].astype(bool)).sum()),
                recovery_audit=audit, low_pose_reference_audit=low_pose_reference_audit(audit),
                kinematics_accepted=audit["accepted"], rms_landmark_error_m=audit["mean_directional_landmark_rms_m"],
                human_tracking_measurement="independent_source_directional_rms_robot_morphology",
                attempt_reference_path=str(attempt.resolve()), elapsed_seconds=time.monotonic()-started)
    clip.metadata.update(base)
    if audit["accepted"]:
        destination = output/"clips"/f"{record['id']}.npz"
        base["reference_path"] = str(destination.resolve())
        clip.metadata.update(base)
        clip.save(destination)
        reloaded = MotionClip.load(destination)
        for key in clip.values:
            import numpy as np
            if not np.array_equal(clip.values[key], reloaded.values[key]):
                raise ValueError("Saved low-pose payload changed")
    else:
        clip.save(attempt)
    return base


def main():
    from convert_bones_seed import records
    from freeze_source import freeze_source
    from k1_motion.reference_admission import admit_reference, reference_rejections, take_family
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--base-library", type=Path, required=True)
    p.add_argument("--dataset", type=Path, default=Path("/mnt/storage/k1-motion/datasets/bones-seed"))
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--limit-per-family", type=int, default=0)
    args = p.parse_args()
    output = args.output.resolve()
    for part in (output, output/"source-cache", output/"attempts", output/"clips", output/"pool"):
        part.mkdir(parents=True, exist_ok=True)
    base_rows = list(rows(args.base_library/"index.jsonl"))
    existing = {r["id"] for r in base_rows}
    metadata = args.dataset/"metadata/seed_metadata_v004.parquet"
    registry = list(records(metadata))
    selected = []
    for r in registry:
        if r["is_mirror"] or r["id"] in existing:
            continue
        description = " ".join([r["take_name"], *r["annotations"]]).lower()
        if r["family"] == "squat" or (r["family"] not in {"climb", "crawl", "inversion_or_stunt"}
                                          and re.search(r"kneel|sit.on.heels", description)):
            selected.append(r)
    selected.sort(key=lambda r: r["id"])
    if args.limit_per_family:
        counts = Counter()
        selected_limited = []
        for r in selected:
            family = "squat" if r["family"] == "squat" else "kneel"
            if counts[family] < args.limit_per_family:
                selected_limited.append(r)
                counts[family] += 1
        selected = selected_limited
    snapshot, revision = freeze_source(ROOT)
    contract = {"version": "genuine-low-pose-expansion-v1", "source_revision": revision,
                "source_snapshot": str(snapshot.resolve()), "workers": args.workers,
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "metadata_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
                "base_library": str(args.base_library.resolve()),
                "base_manifest_sha256": hashlib.sha256((args.base_library/"index.jsonl").read_bytes()).hexdigest(),
                "selection": [r["id"] for r in selected], "originals": len(selected),
                "source_family_originals": dict(Counter(r["family"] for r in registry if not r["is_mirror"])),
                "append_only": True, "mirrors": False, "controller_success_used_for_selection": False}
    campaign = output/"campaign.json"
    if campaign.exists() and json.loads(campaign.read_text()) != contract:
        raise ValueError("Campaign changed; use a fresh immutable output version")
    atomic_json(campaign, contract)
    ledger = output/"index.jsonl"
    done = list(rows(ledger)) if ledger.exists() else []
    done_ids = {r["id"] for r in done}
    if len(done_ids) != len(done) or not done_ids <= set(contract["selection"]):
        raise ValueError("Invalid resume ledger")
    started, last = time.monotonic(), 0
    with ledger.open("a") as sink, ProcessPoolExecutor(max_workers=args.workers, initializer=worker_init,
                                                      initargs=(snapshot,)) as workers:
        jobs = [workers.submit(convert, (r, str(args.dataset), str(output), revision)) for r in selected
                if r["id"] not in done_ids]
        for future in as_completed(jobs):
            row = future.result()
            sink.write(json.dumps(row, allow_nan=False)+"\n")
            sink.flush()
            done.append(row)
            if time.monotonic()-last > 15:
                progress = {"running": True, "processed": len(done), "expected": len(selected),
                            "accepted": sum(r["kinematics_accepted"] for r in done),
                            "elapsed_seconds": time.monotonic()-started}
                atomic_json(output/"status.json", progress)
                print(json.dumps(progress), flush=True)
                last = time.monotonic()
    held_out = {take_family(r["capture_group"]) for r in registry if r["split"] != "train"}
    new_rows, admission_reasons = [], Counter()
    for row in sorted(done, key=lambda r: r["id"]):
        rejected = reference_rejections(row, held_out)
        if rejected:
            admission_reasons.update(rejected)
        else:
            new_rows.append(admit_reference(row))
    pool = base_rows + new_rows
    if len({r["id"] for r in pool}) != len(pool):
        raise ValueError("Duplicate pool original")
    if {take_family(r["capture_group"]) for r in pool if r["split"] == "train"} & held_out:
        raise ValueError("Held-out take family leaked into training")
    index_text = "".join(json.dumps(r, sort_keys=True, allow_nan=False)+"\n" for r in pool)
    index = output/"pool/index.jsonl"
    if index.exists() and index.read_text() != index_text:
        raise ValueError("Frozen pool changed")
    index.write_text(index_text)
    summary = {"complete": True, "originals_processed": len(done), "base_originals_preserved": len(base_rows),
               "new_reference_quality_passes": sum(r["kinematics_accepted"] for r in done),
               "new_admitted_originals": len(new_rows),
               "added_families": {f: {"processed": sum(r["family"] == f for r in done),
                    "quality_passes": sum(r["family"] == f and r["kinematics_accepted"] for r in done),
                    "splits": dict(Counter(r["split"] for r in new_rows if r["family"] == f)),
                    "train_take_families": len({take_family(r["capture_group"]) for r in new_rows
                                                if r["family"] == f and r["split"] == "train"}),
                    "train_hours": sum(r["retargeted_seconds"] for r in new_rows if r["family"] == f
                                       and r["split"] == "train")/3600,
                    "train_phases": dict(Counter(r["recovery_audit"]["phase"] for r in new_rows
                                                if r["family"] == f and r["split"] == "train")),
                    "rejection_reasons": dict(Counter(reason for r in done if r["family"] == f
                        and not r["kinematics_accepted"] for reason in r.get("recovery_audit", {})
                        .get("rejection_reasons", ["source_error"])))} for f in ("kneel", "squat")},
               "admission_rejections": dict(admission_reasons),
               "pool_originals": len(pool), "pool_splits": dict(Counter(r["split"] for r in pool)),
               "pool_train_families": dict(Counter(r["family"] for r in pool if r["split"] == "train")),
               "pool_manifest_sha256": hashlib.sha256(index_text.encode()).hexdigest(),
               "elapsed_seconds": time.monotonic()-started, "physics_qualified": False,
               "controller_success_used_for_selection": False, "hardware_verified": False}
    atomic_json(output/"summary.json", summary)
    atomic_json(output/"status.json", {"running": False, "processed": len(done), "expected": len(selected),
                                       "accepted": summary["new_reference_quality_passes"], "complete": True})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
