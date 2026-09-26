#!/usr/bin/env python3
"""Freeze audited GMR references for RL without filtering on controller ability."""
# ruff: noqa: E402
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from convert_bones_seed import records
from k1_motion.contracts import MotionClip
from k1_motion.reference_admission import ADMISSION_VERSION, admit_reference, reference_rejections, take_family
from k1_motion.robot import K1Model


def complete_rows(path):
    with Path(path).open() as stream:
        for line in stream:
            if not line.endswith("\n"):
                break
            yield json.loads(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--per-family", type=int, default=16)
    parser.add_argument("--validation-per-family", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plan_path = args.output / "plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
    else:
        registry = list(records(args.metadata))
        held_out = {take_family(r["capture_group"]) for r in registry if r["split"] != "train"}
        test_groups = {take_family(r["capture_group"]) for r in registry if r["split"] == "test"}
        selected, rejected, seen = [], Counter(), set()
        source_contracts = []
        for source in args.source:
            source = source.resolve()
            contract = json.loads((source / "campaign.json").read_text())
            source_contracts.append({"root": str(source), "contract": contract})
            for row in complete_rows(source / "index.jsonl"):
                if row["id"] in seen:
                    raise ValueError("Duplicate source original or augmentation")
                seen.add(row["id"])
                reasons = reference_rejections(row, held_out)
                if reasons:
                    rejected.update(reasons)
                    continue
                selected.append({"source": str(source), "row": row})
        # A pilot uses one actor/recording per related take family. The full pool
        # retains all admissible originals and their immutable source split.
        by_split = defaultdict(list)
        for item in sorted(selected, key=lambda x: x["row"]["id"]):
            by_split[item["row"]["split"]].append(item)
        pilot_ids, validation_ids = [], []
        for split, limit, destination in (("train", args.per_family, pilot_ids),
                                          ("validation", args.validation_per_family, validation_ids)):
            counts, used = Counter(), set()
            for item in by_split[split]:
                row = item["row"]
                group = take_family(row["capture_group"])
                if group in used or counts[row["family"]] >= limit:
                    continue
                if split == "validation" and group in test_groups:
                    continue
                used.add(group)
                counts[row["family"]] += 1
                destination.append(row["id"])
        plan = {"version": ADMISSION_VERSION, "sources": source_contracts, "selected": selected,
                "pilot_ids": pilot_ids, "validation_ids": validation_ids,
                "rejection_reason_counts": dict(rejected), "source_rows": len(seen),
                "metadata_sha256": hashlib.sha256(args.metadata.read_bytes()).hexdigest(),
                "per_family_limit": args.per_family, "validation_per_family": args.validation_per_family,
                "split_policy": "original labels retained; training excludes every related held-out take family",
                "controller_success_used_for_selection": False}
        plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    for i, source in enumerate(plan["sources"]):
        paths = [item["row"]["reference_path"] for item in plan["selected"] if item["source"] == source["root"]]
        (args.output / f"source-{i}-files.txt").write_text("\n".join(paths) + "\n")
    if args.plan_only:
        print(json.dumps({"pool_originals": len(plan["selected"]), "pilot_originals": len(plan["pilot_ids"]),
                          "validation_originals": len(plan["validation_ids"]),
                          "rejection_reason_counts": plan["rejection_reason_counts"]}, indent=2))
        return
    spec = K1Model()
    pool = args.output / "pool"
    clips = pool / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    output_rows = []
    for item in plan["selected"]:
        row = item["row"]
        source = Path(item["source"]) / row["reference_path"]
        clip = MotionClip.load(source)
        for key in ("id", "capture_group", "split", "model_signature", "source_motion_id"):
            if clip.metadata[key] != row[key]:
                raise ValueError(f"Payload/ledger mismatch: {row['id']} {key}")
        if clip.metadata["model_signature"] != spec.signature or not clip.values["valid"].all():
            raise ValueError("Invalid reference payload")
        if not np.allclose(np.diff(clip.times), spec.control_dt, rtol=0, atol=1e-9):
            raise ValueError("Wrong reference clock")
        destination = clips / f"{row['id']}.npz"
        if not destination.exists():
            shutil.copy2(source, destination)
        output_rows.append(admit_reference({**row, "reference_path": str(destination.resolve()),
                                           "source_reference_path": str(source)}))
    index_text = "".join(json.dumps(r, sort_keys=True) + "\n" for r in output_rows)
    index = pool / "index.jsonl"
    if index.exists() and index.read_text() != index_text:
        raise ValueError("Frozen reference manifest changed")
    index.write_text(index_text)
    pilot = args.output / "pilot"
    pilot.mkdir(exist_ok=True)
    ids = set(plan["pilot_ids"] + plan["validation_ids"])
    pilot_rows = [r for r in output_rows if r["id"] in ids]
    (pilot / "index.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in pilot_rows))
    summary = {"version": ADMISSION_VERSION, "source_rows": plan["source_rows"],
               "pool_originals": len(output_rows), "pool_split_counts": dict(Counter(r["split"] for r in output_rows)),
               "pool_train_hours": sum(r["retargeted_seconds"] for r in output_rows if r["split"] == "train") / 3600,
               "pilot_split_counts": dict(Counter(r["split"] for r in pilot_rows)),
               "pilot_train_families": dict(Counter(r["family"] for r in pilot_rows if r["split"] == "train")),
               "pilot_validation_families": dict(Counter(r["family"] for r in pilot_rows if r["split"] == "validation")),
               "pilot_train_hours": sum(r["retargeted_seconds"] for r in pilot_rows if r["split"] == "train") / 3600,
               "pool_manifest_sha256": hashlib.sha256(index_text.encode()).hexdigest(),
               "retargeted_reference_training_ready": True, "qualified_robot_demonstration_hours": 0,
               "controller_success_used_for_selection": False, "hardware_verified": False,
               "rejection_reason_counts": plan["rejection_reason_counts"]}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
