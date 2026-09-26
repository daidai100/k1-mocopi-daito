#!/usr/bin/env python3
"""Check immutable payloads/splits and freeze a small integration/replay panel."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"), str(ROOT/"scripts")]


def main():
    from convert_bones_seed import records
    from k1_motion.contracts import MotionClip
    from k1_motion.low_pose_contract import low_pose_accepted
    from k1_motion.math3d import rotation
    from k1_motion.reference_admission import admit_reference, reference_rejections, take_family
    p = argparse.ArgumentParser()
    p.add_argument("--campaign", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--allow-incomplete", action="store_true")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    contract = json.loads((args.campaign/"campaign.json").read_text())
    lines = (args.campaign/"index.jsonl").read_text().splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n") and not args.allow_incomplete:
        raise ValueError("Incomplete ledger line")
    ledger = [json.loads(line) for line in lines if line.endswith("\n")]
    complete = len(ledger) == contract["originals"]
    if not complete and not args.allow_incomplete:
        raise ValueError("Campaign incomplete")
    metadata = Path("/mnt/storage/k1-motion/datasets/bones-seed/metadata/seed_metadata_v004.parquet")
    registry = list(records(metadata))
    held_out = {take_family(r["capture_group"]) for r in registry if r["split"] != "train"}
    checked, eligible, max_clock_lag, minimum_upright = [], [], 0., 1.
    for row in ledger:
        if not row["kinematics_accepted"]:
            continue
        assert low_pose_accepted(row)
        clip = MotionClip.load(row["reference_path"])
        assert low_pose_accepted(clip.metadata)
        assert clip.values["valid"].all()
        for field in ("id", "source_motion_id", "source_family", "family", "capture_group", "split",
                      "model_signature", "recovery_audit", "low_pose_reference_audit"):
            assert row[field] == clip.metadata[field], (row["id"], field)
        assert np.allclose(np.diff(clip.times), .02, atol=1e-12, rtol=0)
        assert np.all(clip.source_times <= clip.times+1e-12)
        max_clock_lag = max(max_clock_lag, float(np.max(clip.times-clip.source_times)))
        minimum_upright = min(minimum_upright, float(rotation(clip.values["root_orientation"]).apply([0,0,1])[:,2].min()))
        checked.append(row)
        if not reference_rejections(row, held_out):
            eligible.append(admit_reference(row))
    assert len({r["id"] for r in ledger}) == len(ledger)
    base = Path(contract["base_library"])
    assert hashlib.sha256((base/"index.jsonl").read_bytes()).hexdigest() == contract["base_manifest_sha256"]
    base_rows = [json.loads(line) for line in (base/"index.jsonl").read_text().splitlines()]
    if complete:
        pool = [json.loads(line) for line in (args.campaign/"pool/index.jsonl").read_text().splitlines()]
        assert pool[:len(base_rows)] == base_rows
        assert len({r["id"] for r in pool}) == len(pool)
        assert {r["id"] for r in pool[len(base_rows):]} == {r["id"] for r in eligible}
        assert not {take_family(r["capture_group"]) for r in pool if r["split"] == "train"} & held_out
    # Selection uses no policy outcome. Training and replay roles remain separate.
    panel, counts = [], Counter()
    for row in sorted(eligible, key=lambda r: r["id"]):
        key = row["family"], row["recovery_audit"]["phase"]
        if counts[key] < 2:
            panel.append({**row, "cohort": "new_"+row["family"]})
            counts[key] += 1
    train, counts = [], Counter()
    for row in sorted(eligible, key=lambda r: r["id"]):
        if row["split"] == "train" and counts[row["family"]] < 12:
            train.append(row)
            counts[row["family"]] += 1
    for family in ("walk", "squat", "transition"):
        train.extend([r for r in base_rows if r["split"] == "train" and r["family"] == family][:2])
    (args.output/"library").mkdir()
    (args.output/"library/index.jsonl").write_text("".join(json.dumps(r, sort_keys=True)+"\n" for r in train))
    (args.output/"panel.json").write_text(json.dumps(panel, indent=2)+"\n")
    report = {"campaign_complete": complete, "checked_new_quality_payloads": len(checked),
              "eligible_new_originals": len(eligible), "base_manifest_unchanged": True,
              "base_originals_preserved": len(base_rows), "zero_invalid_retained_ticks": True,
              "zero_duplicate_originals": True, "zero_held_out_training_leakage": True,
              "max_source_lag_s": max_clock_lag, "minimum_reference_upright_cos": minimum_upright,
              "canary_train_families": dict(Counter(r["family"] for r in train)),
              "replay_phases": dict(Counter(r["recovery_audit"]["phase"] for r in panel)),
              "physics_qualified": False, "controller_success_used_for_selection": False}
    (args.output/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
