#!/usr/bin/env python3
"""Numerically inventory local AMASS archives; do not equate them to K1 references."""

import argparse
import io
import json
from pathlib import Path
import tarfile

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("archives", type=Path, nargs="+")
parser.add_argument("--output", type=Path, default=Path("reports/amass-inventory.json"))
args = parser.parse_args()
report = {
    "archives": [],
    "retargeted_hours": 0.0,
    "physics_qualified_hours": 0.0,
    "body_model_required": "licensed SMPL+H with AMASS-compatible shape; DMPL model for vertex deformation",
}
for path in args.archives:
    result = {"path": str(path), "bytes": path.stat().st_size, "motions": [], "errors": []}
    if path.stat().st_size == 0:
        result["errors"].append("empty_archive")
    else:
        try:
            with tarfile.open(path, "r|bz2") as archive:
                for member in archive:
                    if not member.isfile() or not member.name.endswith(".npz"):
                        continue
                    try:
                        if member.size > 256 * 1024 * 1024:
                            raise ValueError("NPZ exceeds bounded inventory size")
                        with np.load(
                            io.BytesIO(archive.extractfile(member).read()), allow_pickle=False
                        ) as data:
                            pose, trans = data["poses"], data["trans"]
                            fps = float(data["mocap_framerate"])
                            if pose.ndim != 2 or pose.shape[1] != 156 or trans.shape != (len(pose), 3):
                                raise ValueError("Expected AMASS SMPL+H (156) pose/trans")
                            for field in ("poses", "trans", "betas", "dmpls"):
                                if not np.isfinite(data[field]).all():
                                    raise ValueError(f"Nonfinite {field}")
                            if not 1 < fps < 1000:
                                raise ValueError("Invalid framerate")
                            result["motions"].append(
                                {
                                    "source_id": member.name,
                                    "frames": len(pose),
                                    "fps": fps,
                                    "seconds": len(pose) / fps,
                                    "pose_dimensions": 156,
                                    "gender": str(data["gender"]),
                                    "numerical_validation": True,
                                }
                            )
                    except Exception as exc:
                        result["errors"].append({"member": member.name, "error": str(exc)})
        except tarfile.TarError as exc:
            result["errors"].append(str(exc))
    result["numerically_validated_hours"] = sum(m["seconds"] for m in result["motions"]) / 3600
    report["archives"].append(result)
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2) + "\n")
print(
    json.dumps(
        {
            "archives": [
                {k: v for k, v in a.items() if k != "motions"} | {"motions": len(a["motions"])}
                for a in report["archives"]
            ]
        },
        indent=2,
    )
)
