#!/usr/bin/env python3
"""Compare a small original-only K1 motion-scaling panel without promoting data."""

import argparse
import json
from pathlib import Path

import numpy as np

from k1_motion.adapters import bvh_frames
from k1_motion.recovery_validation import audit_recovery, retarget_recovery
from k1_motion.robot import K1Model


ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "artifacts/next-reward-campaign-20260922/retarget-speed80-v1"


def sources():
    rows = json.loads((PANEL / "panel.json").read_text())
    result = {row["preview_label"]: (PANEL / row["id"] / "source.bvh", row["source_motion_id"])
              for row in rows}
    result["high_jump"] = (Path("/mnt/storage/k1-motion/datasets/bones-seed/soma_proportional/"
                                "bvh/230529/high_jump_R_001__A409.bvh"),
                           "bones_seed/high_jump_R_001__A409")
    return result


def metrics(clip, audit):
    root = clip.values["root_position"]
    velocity = np.linalg.norm(clip.values["root_velocity"][:, :2], axis=1)
    joint_speed = np.abs(np.diff(clip.values["joint_position"], axis=0) /
                         np.diff(clip.times)[:, None])
    return {
        "duration_s": float(clip.times[-1] - clip.times[0]),
        "net_xy_m": float(np.linalg.norm(root[-1, :2] - root[0, :2])),
        "path_xy_m": float(np.linalg.norm(np.diff(root[:, :2], axis=0), axis=1).sum()),
        "rise_above_initial_m": float(root[:, 2].max() - root[0, 2]),
        "root_xy_speed_p95_m_s": float(np.percentile(velocity, 95)),
        "root_xy_speed_max_m_s": float(velocity.max()),
        "joint_speed_max_rad_s": float(joint_speed.max()),
        "invalid_ticks": int(np.count_nonzero(clip.values["valid"] == 0)),
        "strict_geometry_accepted": bool(audit["accepted"]),
        "strict_rejection_reasons": audit["rejection_reasons"],
        "max_ground_penetration_m": audit["max_ground_penetration_m"],
        "max_self_penetration_m": audit["max_self_penetration_m"],
        "stance_slip_p95_m_s": audit["stance_slip_p95_m_s"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts/reference-motion-scale-preview-20260923")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    robot = K1Model()
    result = {"version": "reference-motion-scale-preview-v1", "model_signature": robot.signature,
              "training_eligible": False, "physics_qualified": False, "motions": {}}
    for label, (source, source_id) in sources().items():
        frames = list(bvh_frames(source, "bones_seed", source_id, target_hz=50.0))
        human = {"times": np.array([f.source_time for f in frames]),
                 "positions": np.stack([f.positions for f in frames]),
                 "orientations": np.stack([f.orientations for f in frames])}
        variants = {}
        for name, profile in (("baseline", None), ("robot_fit_70", "robot-fit-70-v1")):
            metadata = {"source_motion_id": source_id, "is_mirror": False,
                        "physics_qualified": False, "training_eligible": False}
            clip, reports = retarget_recovery(robot, human, metadata, motion_profile=profile)
            mean_error = float(np.mean([r["rms_landmark_error_m"] for r in reports]))
            audit = audit_recovery(clip, clip, reports, {"rms_landmark_error_m": mean_error})
            clip.save(args.output / f"{label}-{name}.npz")
            variants[name] = {"retarget_version": clip.metadata["retarget_version"],
                              "motion_scale": clip.metadata.get("motion_scale"),
                              "metrics": metrics(clip, audit)}
        result["motions"][label] = {"source": str(source), "variants": variants}
        (args.output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
