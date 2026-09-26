#!/usr/bin/env python3
"""Take-balanced temporal pose/velocity novelty relative to the previous corpus."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]


def choose(rows, per_group=3):
    from k1_motion.reference_admission import take_family
    counts, result = Counter(), []
    for row in sorted(rows, key=lambda r: r["id"]):
        key = take_family(row["capture_group"])
        if counts[key] < per_group:
            result.append(row)
            counts[key] += 1
    return result


def describe(row):
    from k1_motion.contracts import MotionClip
    from k1_motion.math3d import rotation
    from k1_motion.recovery_geometry import recovery_model
    clip = MotionClip.load(row["reference_path"])
    spec = recovery_model("audit")
    q = clip.values["joint_position"]
    dq = np.vstack([np.zeros((1, 22)), np.diff(q, axis=0) / np.diff(clip.times)[:, None]])
    inv = rotation(clip.values["root_orientation"]).inv()
    knees = np.asarray(clip.metadata.get("knee_contacts", np.zeros((len(q), 2))))
    features = np.c_[2 * (q - spec.limits.mean(axis=1)) / np.diff(spec.limits, axis=1).ravel(),
                     dq / 6., inv.apply(clip.values["root_velocity"][:, :3]),
                     inv.apply(clip.values["root_velocity"][:, 3:]) / 3.,
                     clip.values["root_position"][:, 2] / .5, clip.values["contacts"], knees]
    frames = np.unique(np.linspace(0, len(q) - 1, min(16, len(q))).astype(int))
    pairs = np.c_[features[np.maximum(frames - 5, 0)], features[frames]].astype(np.float32)
    return pairs


def main():
    import torch
    from k1_motion.reference_admission import take_family
    from prepare_broad_references import atomic_json, rows
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-library", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a new neighborhood report path")
    started = time.monotonic()
    base = [r for r in rows(args.base_library / "index.jsonl") if r["split"] == "train"]
    old_ids = {r["id"] for r in base}
    added = [r for r in rows(args.library / "index.jsonl") if r["split"] == "train" and r["id"] not in old_ids]
    old_selected, new_selected = choose(base), choose(added)
    if not new_selected:
        raise ValueError("No added training clips")
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        old_features = list(executor.map(describe, old_selected, chunksize=8))
        new_features = list(executor.map(describe, new_selected, chunksize=8))
    # Compute actual nearest distances, not an approximate index or fitted rank.
    # Batching bounds memory; the GPU is used only for this read-only diagnostic.
    torch.set_num_threads(2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    a = torch.as_tensor(np.concatenate(old_features), device=device)
    b = torch.as_tensor(np.concatenate(new_features), device=device)
    anorm = a.square().sum(1)
    distances = []
    with torch.inference_mode():
        for start in range(0, len(b), 128):
            part = b[start:start + 128]
            squared = (part.square().sum(1, keepdim=True) + anorm[None] - 2 * part @ a.T).clamp(min=0)
            distances.extend((squared.min(1).values / a.shape[1]).sqrt().cpu().tolist())
    by_family = {}
    offset = 0
    for row, feature in zip(new_selected, new_features):
        by_family.setdefault(row["family"], []).extend(distances[offset:offset + len(feature)])
        offset += len(feature)
    report = {"version": "take-balanced-temporal-neighborhood-v1", "complete": True,
              "base_library": str(args.base_library.resolve()), "library": str(args.library.resolve()),
              "baseline_clips_sampled": len(old_selected), "added_clips_sampled": len(new_selected),
              "baseline_take_families": len({take_family(r["capture_group"]) for r in old_selected}),
              "added_take_families_sampled": len({take_family(r["capture_group"]) for r in new_selected}),
              "baseline_descriptors": len(a), "added_descriptors": len(b), "descriptor_dimensions": a.shape[1],
              "sampling": "At most 3 originals per related take family; 16 evenly spaced phases per clip",
              "descriptor": "Current and 100ms-past joint pose, causal joint speed, body-frame root velocity, height, feet/knee support labels",
              "units": "RMS of scaled mixed features; not a tracking error or physical acceptance threshold",
              "nearest_baseline_rms_median": float(np.median(distances)),
              "nearest_baseline_rms_p90": float(np.percentile(distances, 90)),
              "fraction_above_diagnostic_rms_0p1": float(np.mean(np.asarray(distances) > .1)),
              "by_family": {f: {"descriptors": len(v), "median": float(np.median(v)),
                                "p90": float(np.percentile(v, 90)),
                                "fraction_above_diagnostic_rms_0p1": float(np.mean(np.asarray(v) > .1))}
                            for f, v in sorted(by_family.items())},
              "device": device, "elapsed_seconds": time.monotonic() - started,
              "held_out_test_used": False, "policy_outcome_used": False,
              "limitations": "Take/phase sample only. Novelty can include undesirable variation; it is not proof of human-motion coverage or learnability."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
