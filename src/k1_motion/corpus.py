"""Versioned motion preparation with whole-recording splits and retained rejects."""

from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np

from .adapters import bvh_frames, mmm_frames
from .contracts import MotionClip
from .retarget import RETARGET_VERSION, Retargeter
from .robot import K1Model, ROOT


def recording_split(group, seed="k1-motion-v1"):
    bucket = int.from_bytes(hashlib.sha256(f"{seed}/{group}".encode()).digest()[:4], "big") % 100
    return "train" if bucket < 80 else "validation" if bucket < 90 else "test"


def family_of(record):
    text = " ".join(record.get("annotations", [])).lower()
    if re.search(
        r"obstacle|furniture|chair|table|stairs|ladder|wall|ground|crawl|fall|get.?up|push|sit|lie|lying|crouch under"
        r"|kneel|lay(?:s|ing)? down|all fours|on (?:his |her |its |their |the |both |one )?knees?\b",
        text,
    ):
        return "external_support_or_recovery"
    if re.search(r"\b(?:hand|head)[ -]?stands?\b|cartwheel|somersault|(?:back|front)[ -]?flip", text):
        return "external_support_or_recovery"
    for name, pattern in (
        ("squat", r"squat|knee bend"),
        ("kick", r"kick"),
        ("jump", r"jump|hop"),
        ("turn", r"turn"),
        ("run", r"run|sprint|dash"),
        ("walk", r"walk"),
        ("dance", r"dance"),
        ("reach", r"raise|reach|wave|byebye|bye|guide|call|aim"),
        ("bow", r"bow"),
        ("punch", r"punch|fight"),
        ("stance", r"stand|idle"),
    ):
        if re.search(pattern, text):
            return name
    return "unclassified"


def select_records(limit=40):
    buckets = defaultdict(list)
    for dataset in ("bandai_namco", "lafan1", "kit_motion_language"):
        for line in (ROOT / "manifests" / f"{dataset}.motions.jsonl").read_text().splitlines():
            row = json.loads(line)
            family = family_of(row)
            if family not in ("external_support_or_recovery", "unclassified") and record_duration_ok(row):
                buckets[(family, dataset)].append({**row, "family": family})
    # Interleave families/datasets; prefer short clips to qualify a varied set quickly.
    for rows in buckets.values():
        rows.sort(key=lambda row: (row["duration_seconds"], row["source_motion_id"]))
    selected, groups = [], set()
    while buckets and len(selected) < limit:
        for key in sorted(list(buckets)):
            rows = buckets[key]
            while rows and rows[0]["capture_group"] in groups:
                rows.pop(0)
            if not rows:
                del buckets[key]
                continue
            row = rows.pop(0)
            selected.append(row)
            groups.add(row["capture_group"])
            if len(selected) == limit:
                break
    return selected


def record_duration_ok(record):
    # Subsecond fragments cannot establish sustained tracking. Keep original
    # inventory intact; this is an explicit development-library selection rule.
    return record["duration_seconds"] >= 2.0


def _convert(job):
    record, out, max_seconds = job
    out = Path(out)
    key = hashlib.sha256(record["source_motion_id"].encode()).hexdigest()[:16]
    base = {
        "id": key,
        "source_motion_id": record["source_motion_id"],
        "capture_group": record["capture_group"],
        "split": recording_split(record["capture_group"]),
        "dataset": record["dataset"],
        "family": record["family"],
        "source_path": record["path"],
        "annotations": record["annotations"],
        "source_duration_seconds": record["duration_seconds"],
        "physics_qualified": False,
        "human_reviewed": False,
        "retarget_version": RETARGET_VERSION,
        "flat_ground": "candidate_annotation_screen_only",
    }
    try:
        path = ROOT / record["path"]
        if record["format"] == "BVH":
            frames = bvh_frames(path, record["dataset"], record["source_motion_id"], max_seconds)
        else:
            frames = mmm_frames(
                path,
                ROOT / "data/reference/mmmpy_lite/mmmpy_lite/data/models/mmm/mmm.urdf",
                record.get("track_index", 0),
                record["source_motion_id"],
                max_seconds,
            )
        robot = K1Model()
        retargeter = Retargeter(robot)
        human, refs, reports = [], [], []
        for frame in frames:
            if not human:
                retargeter.calibrate(frame)
            human.append(frame)
            refs.append(retargeter.process(frame))
            reports.append(retargeter.last_report)
        if len(refs) < 2:
            raise ValueError("Too few frames")
        rejected = Counter(reason for r in reports for reason in r["rejection_reasons"])
        base.update(
            {
                "frames": len(refs),
                "retargeted_seconds": refs[-1].source_time - refs[0].source_time,
                "kinematics_accepted": not rejected,
                "rejection_counts": dict(rejected),
                "rms_landmark_error_m": float(np.mean([r["rms_landmark_error_m"] for r in reports])),
                "retarget_p95_ms": float(np.percentile([r["seconds"] for r in reports], 95) * 1000),
                "root_projection_p95_m": float(np.percentile([r["root_projection_m"] for r in reports], 95)),
                "contact_anchor_error_p95_m": float(
                    np.percentile([r["contact_anchor_error_m"] for r in reports], 95)
                ),
                "model_signature": robot.signature,
                "calibration": retargeter.calibration.metadata(),
                "license_dataset": record["dataset"],
                "license_directory": str(ROOT / "licenses" / record["dataset"]),
                "reference_path": f"clips/{key}.npz",
                "human_path": f"human/{key}.npz",
            }
        )
        MotionClip.from_references(refs, base).save(out / base["reference_path"])
        (out / "human").mkdir(exist_ok=True)
        np.savez_compressed(
            out / base["human_path"],
            times=np.array([f.source_time for f in human]),
            positions=np.stack([f.positions for f in human]),
            orientations=np.stack([f.orientations for f in human]),
        )
        return base
    except (ValueError, KeyError, IndexError, FileNotFoundError) as exc:
        return {**base, "kinematics_accepted": False, "error": f"{type(exc).__name__}: {exc}"}


def prepare(output, limit=40, max_seconds=8.0, workers=4):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    records = select_records(limit)
    (output / "selection.json").write_text(json.dumps(records, indent=2) + "\n")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_convert, ((r, str(output), max_seconds) for r in records)))
    with (output / "index.jsonl").open("w") as stream:
        for row in results:
            stream.write(json.dumps(row) + "\n")
    accepted = [r for r in results if r["kinematics_accepted"]]
    report = {
        "selected": len(records),
        "converted": sum("reference_path" in r for r in results),
        "kinematics_accepted": len(accepted),
        "rejected": len(results) - len(accepted),
        "families": dict(Counter(r["family"] for r in results)),
        "accepted_families": dict(Counter(r["family"] for r in accepted)),
        "splits": dict(Counter(r["split"] for r in accepted)),
        "source_hours": sum(r["source_duration_seconds"] for r in results) / 3600,
        "retargeted_hours": sum(r.get("retargeted_seconds", 0) for r in results) / 3600,
        "kinematically_accepted_hours": sum(r["retargeted_seconds"] for r in accepted) / 3600,
        "physics_qualified_hours": 0.0,
        "human_reviewed": False,
        "elapsed_seconds": time.monotonic() - start,
        "split_rule": "sha256(seed/capture_group); related takes/windows/mirrors must inherit group",
        "scope": "Offline kinematic candidates. Not a qualified training corpus or behavioral result.",
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
