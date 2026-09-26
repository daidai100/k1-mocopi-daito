#!/usr/bin/env python3
"""Check saved sole-height corrections against their original payloads and splits."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from k1_motion.contracts import MotionClip  # noqa: E402
from k1_motion.reference_admission import reference_rejections, take_family, ground_rl_accepted  # noqa: E402


def check(row):
    clip = MotionClip.load(row["reference_path"])
    source = MotionClip.load(row["source_reference_path"])
    for name in ("times", "source_times", "received_times"):
        if not np.array_equal(getattr(clip, name), getattr(source, name)):
            raise ValueError(f"Changed {name}: {row['id']}")
    for key in ("joint_position", "joint_velocity", "root_orientation", "contacts", "contact_confidence", "valid"):
        np.testing.assert_array_equal(clip.values[key], source.values[key], err_msg=row["id"])
    np.testing.assert_array_equal(clip.values["root_position"][:, :2], source.values["root_position"][:, :2])
    height = clip.values["root_position"][:, 2] - source.values["root_position"][:, 2]
    assert height.min() > -1e-12 and height.max() <= .05 + 1e-12
    np.testing.assert_allclose(clip.values["landmarks"][:, :, :2], source.values["landmarks"][:, :, :2],
                               atol=0, rtol=0)
    np.testing.assert_allclose(clip.values["landmarks"][:, :, 2],
                               source.values["landmarks"][:, :, 2] + height[:, None], atol=1e-12, rtol=0)
    np.testing.assert_allclose(clip.values["root_velocity"][1:, 2] - source.values["root_velocity"][1:, 2],
                               np.diff(height) / np.diff(clip.times), atol=1e-11, rtol=0)
    np.testing.assert_array_equal(clip.values["root_velocity"][:, [0, 1, 3, 4, 5]],
                                  source.values["root_velocity"][:, [0, 1, 3, 4, 5]])
    assert clip.values["valid"].all()
    if not row["kinematics_accepted"]:
        assert ground_rl_accepted(row)
        assert clip.metadata["rl_reference_audit"] == row["rl_reference_audit"]
    return {"id": row["id"], "family": row["family"],
            "corrected": row["walking_status"] != "v4_pass_preserved",
            "max_lift_m": float(height.max()),
            "rms_lift_m": float(np.sqrt(np.mean(height**2)))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    rows = [json.loads(line) for line in (args.source / "index.jsonl").open()]
    accepted = [r for r in rows if r["kinematics_accepted"] or ground_rl_accepted(r)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(check, accepted, chunksize=8))
    library = [json.loads(line) for line in (args.source / "pool/index.jsonl").open()]
    held_out = {take_family(r["capture_group"]) for r in library if r["split"] != "train"}
    assert len({r["id"] for r in library}) == len(library)
    for row in library:
        assert not reference_rejections(row, held_out), row["id"]
        assert Path(row["reference_path"]).is_file()
        assert not row.get("is_mirror")
    corrected = [r for r in results if r["corrected"]]
    report = {"checked_payloads": len(results), "checked_families": dict(Counter(r["family"] for r in results)),
              "checked_walking_payloads": sum(r["family"] == "walk" for r in results),
              "corrected_payloads": len(corrected),
              "all_motion_and_clock_invariants_passed": True,
              "pool_records": len(library), "pool_splits": dict(Counter(r["split"] for r in library)),
              "duplicate_ids": 0, "train_heldout_take_overlap": 0,
              "median_corrected_rms_lift_mm": float(np.median([r["rms_lift_m"] for r in corrected]) * 1000)
              if corrected else 0.0,
              "p95_corrected_rms_lift_mm": float(np.percentile([r["rms_lift_m"] for r in corrected], 95) * 1000)
              if corrected else 0.0,
              "max_corrected_lift_mm": max((r["max_lift_m"] for r in corrected), default=0) * 1000,
              "controller_performance_claim": False}
    (args.source / "payload-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
