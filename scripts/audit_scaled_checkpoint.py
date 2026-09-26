#!/usr/bin/env python3
"""Compare training/reference scaling and replay three training-only canaries."""

import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from k1_motion.contracts import MotionClip
from k1_motion.control_validation import replay_clip
from k1_motion.export import export_checkpoint
from k1_motion.learning import MotionLibrary, Policy
from k1_motion.reference_scale import scale_clip, scale_library
from k1_motion.robot import K1Model

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--checkpoint", type=Path, required=True)
p.add_argument("--library", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
torch.set_num_threads(1)
export_checkpoint(a.checkpoint, a.output / "actor.pt")
robot = K1Model()
policy = Policy(a.output / "actor.pt", robot.signature)
contract = policy.metadata["reference_scale"]
rows = [json.loads(x) for x in (a.library / "index.jsonl").read_text().splitlines()]
selected = []
for family in ("walk", "jump", "gesture"):
    candidates = [
        r for r in rows if r["split"] == "train" and r["family"] == family and 50 <= r.get("frames", 0) <= 250
    ]
    row = min(candidates, key=lambda r: (r["frames"], r["id"])).copy()
    row["reference_path"] = str(
        ((a.library / "index.jsonl").resolve().parent / row["reference_path"]).resolve()
    )
    selected.append(row)
library_dir = a.output / "library"
library_dir.mkdir()
(library_dir / "index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in selected))
library = MotionLibrary(library_dir, robot, "cpu", storage="packed")
scale_library(library, robot, contract["scale"])
results = []
for row, offset, length in zip(library.rows, library.offsets.tolist(), library.lengths.tolist()):
    original = MotionClip.load(row["reference_path"])
    scaled = scale_clip(original, robot, contract)
    errors = {
        k: float(np.max(np.abs(scaled.values[k] - library.values[k][offset : offset + length].numpy())))
        for k in ("root_position", "root_velocity", "landmarks")
    }
    if max(errors.values()) > 1e-6:
        raise ValueError(f"Training/replay mismatch: {row['id']}: {errors}")
    source_path = float(
        np.linalg.norm(np.diff(original.values["root_position"][:, :2], axis=0), axis=1).sum()
    )
    scaled_path = float(np.linalg.norm(np.diff(scaled.values["root_position"][:, :2], axis=0), axis=1).sum())
    replay = replay_clip(robot, policy, original, a.output / (row["id"] + ".npz"))
    results.append(
        dict(
            id=row["id"],
            family=row["family"],
            frames=length,
            parity_errors=errors,
            source_xy_path_m=source_path,
            scaled_xy_path_m=scaled_path,
            path_scale_ratio=scaled_path / source_path if source_path > 1e-8 else None,
            replay=replay,
        )
    )
report = dict(
    checkpoint=str(a.checkpoint.resolve()),
    reference_scale=contract,
    training_only=True,
    scope="Implementation parity and execution canaries, not held-out behavioral acceptance",
    execution_errors=0,
    results=results,
)
(a.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
print(
    json.dumps(
        [{k: r[k] for k in ("family", "frames", "parity_errors", "path_scale_ratio")} for r in results],
        indent=2,
    )
)
