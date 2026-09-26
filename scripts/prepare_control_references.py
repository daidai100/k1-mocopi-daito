#!/usr/bin/env python3
"""Rebuild an immutable selection at live control cadence, retaining every rejection."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

from freeze_source import freeze_source


def convert(job):
    import numpy as np

    from k1_motion.retarget import RETARGET_VERSION
    from k1_motion.robot import K1Model
    from k1_motion.streaming import retarget_at_control_rate

    row, source, output = job
    source, output = Path(source), Path(output)
    base = {
        **row,
        "parent_kinematics_accepted": row["kinematics_accepted"],
        "parent_rejection_counts": row.get("rejection_counts", {}),
        "source_revision": os.environ["K1_SOURCE_REVISION"],
        "retarget_version": RETARGET_VERSION,
        "physics_qualified": False,
        "human_reviewed": False,
    }
    # Rows whose source conversion failed remain explicit unavailable recordings.
    if not row.get("human_path"):
        return {**base, "rebuild_status": "canonical_human_unavailable"}
    shutil.copyfile(source / row["human_path"], output / row["human_path"])
    try:
        with np.load(source / row["human_path"], allow_pickle=False) as saved:
            human = {key: saved[key] for key in ("times", "positions", "orientations")}
        robot = K1Model()
        if row["model_signature"] != robot.signature:
            raise ValueError("Parent reference robot contract mismatch")
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
                "root_projection_p95_m": float(np.percentile([r["root_projection_m"] for r in reports], 95)),
                "contact_anchor_error_p95_m": float(
                    np.percentile([r["contact_anchor_error_m"] for r in reports], 95)
                ),
                "sampling": clip.metadata["sampling"],
                "calibration": clip.metadata["calibration"],
                "rebuild_status": "converted",
            }
        )
        base.pop("error", None)
        clip.metadata.update(base)
        clip.save(output / row["reference_path"])
        return base
    except (ValueError, KeyError, IndexError, FileNotFoundError) as exc:
        base.pop("reference_path", None)
        return {
            **base,
            "kinematics_accepted": False,
            "rebuild_status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    frozen, revision = freeze_source(root)
    sys.path.insert(0, str(frozen))
    os.environ["K1_MOTION_ROOT"] = str(root)
    os.environ["K1_SOURCE_REVISION"] = revision
    source, output = Path(args.library), Path(args.output)
    index = (source / "index.jsonl").read_bytes()
    rows = list(map(json.loads, index.splitlines()))
    output.mkdir(parents=True, exist_ok=False)
    (output / "human").mkdir()
    (output / "clips").mkdir()
    if (source / "selection.json").exists():
        shutil.copyfile(source / "selection.json", output / "selection.json")
    started = time.monotonic()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool, (output / "index.jsonl").open("x") as stream:
        for result in pool.map(convert, ((row, str(source), str(output)) for row in rows)):
            results.append(result)
            stream.write(json.dumps(result) + "\n")
            stream.flush()
            if len(results) % 25 == 0:
                print(json.dumps({"converted_rows": len(results), "selected": len(rows)}), flush=True)
    accepted = [r for r in results if r["kinematics_accepted"]]
    report = {
        "parent_library": str(source.resolve()),
        "parent_index_sha256": hashlib.sha256(index).hexdigest(),
        "source_revision": revision,
        "selected": len(results),
        "converted": sum(r["rebuild_status"] == "converted" for r in results),
        "kinematics_accepted": len(accepted),
        "rejected": len(results) - len(accepted),
        "splits": dict(Counter(r["split"] for r in accepted)),
        "families": dict(Counter(r["family"] for r in results)),
        "accepted_families": dict(Counter(r["family"] for r in accepted)),
        "newly_accepted": [
            r["id"] for r in results if r["kinematics_accepted"] and not r["parent_kinematics_accepted"]
        ],
        "newly_rejected": [
            r["id"] for r in results if not r["kinematics_accepted"] and r["parent_kinematics_accepted"]
        ],
        "source_hours": sum(r["source_duration_seconds"] for r in results) / 3600,
        "retargeted_hours": sum(
            r.get("retargeted_seconds", 0) for r in results if r["rebuild_status"] == "converted"
        )
        / 3600,
        "kinematically_accepted_hours": sum(r["retargeted_seconds"] for r in accepted) / 3600,
        "physics_qualified_hours": 0.0,
        "elapsed_seconds": time.monotonic() - started,
        "scope": "Same selected recordings and splits; newest human frame at each control tick; no physics qualification",
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
