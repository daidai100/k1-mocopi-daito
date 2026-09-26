#!/usr/bin/env python3
"""Bind train-only locomotion tags and a controller-only initializer anchor."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in (args.bundle / "library/index.jsonl").read_text().splitlines()]
    rows = [r for r in rows if r["split"] == "train" and r.get("training_eligible", True)]
    if len(rows) != 18054 or any(r.get("is_mirror") for r in rows):
        raise ValueError("Expected the full frozen 18,054-original training pool")
    selections = {}
    for row in rows:
        if row["family"] in ("walk", "run", "turn"):
            selections[row["id"]] = "family:" + row["family"]
        elif row["family"] == "transition":
            label = " ".join([row.get("motion_package", ""), row.get("filename", ""),
                              *row.get("annotations", [])]).lower().replace("_", " ")
            if re.search(r"\b(locomotion|walk\w*|jog\w*|run|running|sprint\w*|turn\w*)\b", label):
                selections[row["id"]] = "transition:explicit locomotion source tag"
    payload = {"version": "train-locomotion-phase-v1", "train_ids": [r["id"] for r in rows],
               "locomotion_ids": sorted(selections), "selection_reasons": selections,
               "families": dict(Counter(r["family"] for r in rows)),
               "datasets": dict(Counter(r["dataset"] for r in rows)),
               "locomotion_families": dict(Counter(r["family"] for r in rows if r["id"] in selections)),
               "sampling": {"target_locomotion_transition_share": .5, "start_fraction": .25,
                            "pre_failure_fraction": .5, "uniform_fraction": .25,
                            "bin_seconds": 1, "failure_weight_bounds": [1, 4], "failure_decay_per_update": .99},
               "held_out_outcomes_used": False, "physics_qualified_demonstrations": 0}
    destination = args.bundle / "manifests/rl-beam-curriculum.json"
    if destination.exists():
        raise ValueError("Refusing to overwrite a frozen curriculum")
    destination.write_text(json.dumps(payload, indent=2)+"\n")
    import torch
    from k1_motion.actuation import action_settings
    from k1_motion.robot import K1Model
    saved = torch.load(args.bundle / "initialize.pt", weights_only=True, map_location="cpu")
    saved["action_settings"] = action_settings(K1Model(), json.loads(
        (args.bundle / "configs/controller-pv-arm-feedback-v1.json").read_text()))
    torch.save(saved, args.bundle / "initialize-pv.pt")
    print(json.dumps({"train": len(rows), "locomotion": len(selections),
                      "locomotion_families": payload["locomotion_families"],
                      "manifest_sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
