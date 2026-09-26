#!/usr/bin/env python3
"""Audit contact consistency without treating kinematic validity as physics success."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np


def audit(directory):
    root = Path(directory)
    rows = [json.loads(s) for s in (root / "index.jsonl").read_text().splitlines()]
    trials = []
    for row in rows:
        trial = {k: row[k] for k in ("id", "family", "split", "kinematics_accepted")}
        trial["training_eligible"] = row.get("training_eligible", True)
        if "reference_path" in row:
            with np.load(root / row["reference_path"]) as clip:
                contact = (clip["contacts"][1:] > 0.5) & (clip["contacts"][:-1] > 0.5)
                dt = np.diff(clip["times"])
                speed = np.linalg.norm(
                    np.diff(clip["landmarks"][:, [11, 15], :2], axis=0) / dt[:, None, None], axis=-1
                )
                trial.update(
                    {
                        "planted_samples": int(contact.sum()),
                        "planted_foot_speed_p95_mps": float(np.percentile(speed[contact], 95))
                        if contact.any()
                        else None,
                        "planted_fraction_over_0p2_mps": float(np.mean(speed[contact] > 0.2))
                        if contact.any()
                        else None,
                        "joint_speed_max_rps": float(
                            np.max(np.abs(np.diff(clip["joint_position"], axis=0) / dt[:, None]))
                        ),
                    }
                )
        trials.append(trial)
    summaries = {}
    for family in sorted({r["family"] for r in rows}):
        subset = [
            r
            for r in trials
            if r["family"] == family
            and r["kinematics_accepted"]
            and r["split"] == "train"
            and r["training_eligible"]
        ]
        speeds = [
            r["planted_foot_speed_p95_mps"] for r in subset if r.get("planted_foot_speed_p95_mps") is not None
        ]
        summaries[family] = {
            "accepted_train_clips": len(subset),
            "median_clip_p95_planted_foot_speed_mps": float(np.median(speeds)) if speeds else None,
            "clips_over_0p2_mps": sum(s > 0.2 for s in speeds),
        }
    return {
        "library": str(root),
        "selected": len(rows),
        "accepted": sum(r["kinematics_accepted"] for r in rows),
        "eligible_accepted_training_recordings": sum(
            r["split"] == "train" and r["kinematics_accepted"] and r.get("training_eligible", True)
            for r in rows
        ),
        "splits": dict(Counter(r["split"] for r in rows if r["kinematics_accepted"])),
        "scope": "Reference-only kinematic consistency, not closed-loop physics qualification",
        "train_families": summaries,
        "trials": trials,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("libraries", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    results = [audit(p) for p in args.libraries]
    Path(args.output).write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps([{k: v for k, v in r.items() if k != "trials"} for r in results], indent=2))
