#!/usr/bin/env python3
"""Development panel; outputs are audited candidates, not a training manifest."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]


def convert(job):
    from k1_motion.adapters import bvh_frames
    from k1_motion.contracts import MotionClip
    from k1_motion.low_pose import correct_low_pose_ground, retarget_low_pose
    from k1_motion.low_pose_validation import audit_low_pose
    from k1_motion.recovery_geometry import recovery_model
    row, output, dataset = job
    started = time.monotonic()
    output = Path(output)
    family = "squat" if row["family"] == "squat" else "kneel"
    try:
        frames = list(bvh_frames(Path(dataset)/row["source_path"], "bones_seed_v2", row["source_motion_id"], target_hz=50))
        clip, reports = retarget_low_pose(recovery_model("retarget"), frames, {**row, "source_family": row["family"], "family": family})
        clip, correction = correct_low_pose_ground(clip)
        path = output / "attempts" / f"{row['id']}.npz"
        clip.save(path)
        audit = audit_low_pose(MotionClip.load(path), frames, family)
        result = {**row, "family": family, "audit": audit, "attempt_reference_path": str(path.resolve()),
                  "rejected_ticks": int((~clip.values["valid"].astype(bool)).sum()),
                  "ground_correction": correction,
                  "seconds": time.monotonic()-started}
    except Exception as error:
        result = {**row, "error": f"{type(error).__name__}: {error}", "seconds": time.monotonic()-started}
    (output/f"{row['id']}.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    return result


def main():
    from convert_bones_seed import records
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--ids", nargs="+")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--dataset", default="/mnt/storage/k1-motion/datasets/bones-seed")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    ids = set(args.ids or ["c618399abf2f4b96be42", "8bf6f18bef93391199d9", "0d019e78ff370e947fb9",
                          "ce36fc71e3d41d269bf6", "bbf8e7db83baaada3aee", "bc663759ca3026805fc3"])
    rows = [r for r in records(Path(args.dataset)/"metadata/seed_metadata_v004.parquet") if r["id"] in ids]
    with ProcessPoolExecutor(max_workers=args.workers) as workers:
        jobs = [workers.submit(convert, (r, args.output, args.dataset)) for r in rows]
        results = []
        for future in as_completed(jobs):
            r = future.result()
            results.append(r)
            print(json.dumps(r), flush=True)
    (args.output/"results.json").write_text(json.dumps(results, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
