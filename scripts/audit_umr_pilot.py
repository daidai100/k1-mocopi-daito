#!/usr/bin/env python3
"""Compare native UMR poses with a matched BONES-SEED V4 original.

UMR is an offline trajectory optimizer. This audit checks the saved poses on
the production K1 model; it does not claim causal retargeting or dynamic safety.
"""

import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from k1_motion.contracts import JOINT_NAMES, MotionClip  # noqa: E402
from k1_motion.recovery_geometry import geometry_forward, recovery_model  # noqa: E402
from k1_motion.recovery_validation import RECOVERY_GATES  # noqa: E402
from k1_motion.retarget_speed import retarget_speed_contract  # noqa: E402


def load_row(index, filename):
    with index.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["filename"] == filename:
                if row["is_mirror"]:
                    raise ValueError("Pilot requires an original motion")
                return row
    raise ValueError(f"Baseline original not found: {filename}")


def load_umr_qpos(path, expected_xml, filename):
    # Upstream UMR writes robot_joint_names as an object array. Only pass a
    # locally generated UMR result here; NumPy must unpickle that one field.
    with np.load(path, allow_pickle=True) as result:
        qpos = np.asarray(result["qpos"], dtype=float)
        fps = float(np.asarray(result["fps"]).reshape(-1)[0])
        frame_ids = np.asarray(result["frame_ids"], dtype=int).reshape(-1)
        joints = tuple(str(x) for x in result["robot_joint_names"])
        robot_xml = Path(str(result["robot_xml"])).resolve()
        source_key = str(result["source_sequence_key"])
        source_format = str(result["source_format"])
    if source_key != filename or source_format != "soma_bvh_boneseed_proportional":
        raise ValueError(f"UMR source differs from matched BONES-SEED original: {source_key}, {source_format}")
    if (robot_xml.parent != expected_xml.resolve().parent
            or not robot_xml.name.endswith(".floating_mjcf.xml")
            or not robot_xml.is_file()):
        raise ValueError(f"UMR did not use a generated production K1 MJCF: {robot_xml}")
    if joints != JOINT_NAMES:
        raise ValueError(f"UMR joint order differs from production K1: {joints}")
    if qpos.ndim != 2 or qpos.shape != (len(frame_ids), 29):
        raise ValueError(f"Unexpected UMR qpos/frame IDs: {qpos.shape}, {frame_ids.shape}")
    if not np.isfinite(qpos).all() or not np.isfinite(fps) or fps <= 0:
        raise ValueError("Nonfinite UMR poses or frame rate")
    if np.any(np.diff(frame_ids) <= 0) or np.any(frame_ids < 0):
        raise ValueError("UMR frame IDs must increase")
    if not np.allclose(np.linalg.norm(qpos[:, 3:7], axis=1), 1, atol=1e-5):
        raise ValueError("UMR root quaternions are not unit length")
    return qpos, (frame_ids - frame_ids[0]) / fps, fps


def audit_path(qpos):
    robot = recovery_model("audit")
    model, data = robot.model, robot.data
    speed_limits = np.asarray(retarget_speed_contract(robot)["joint_velocity_limits_rad_s"])
    speed = np.abs(np.diff(qpos[:, 7:], axis=0)) / robot.control_dt
    limit_error = np.maximum(model.jnt_range[1:, 0] - qpos[:, 7:],
                             qpos[:, 7:] - model.jnt_range[1:, 1])
    self_depth = ground_depth = 0.0
    self_bad = ground_bad = 0
    samples = 0
    delta = np.zeros(model.nv)
    for i, pose in enumerate(qpos):
        fractions = (1.0,) if i == 0 else np.arange(1, robot.substeps + 1) / robot.substeps
        if i:
            mujoco.mj_differentiatePos(model, delta, 1.0, qpos[i - 1], pose)
        for fraction in fractions:
            data.qpos[:] = qpos[i - 1] if i else pose
            if i:
                mujoco.mj_integratePos(model, data.qpos, delta, float(fraction))
            geometry_forward(model, data)
            self_sample = ground_sample = 0.0
            for contact in data.contact[:data.ncon]:
                depth = max(0.0, -float(contact.dist))
                bodies = model.geom_bodyid[[contact.geom1, contact.geom2]]
                if 0 in bodies:
                    ground_sample = max(ground_sample, depth)
                else:
                    self_sample = max(self_sample, depth)
            self_depth = max(self_depth, self_sample)
            ground_depth = max(ground_depth, ground_sample)
            self_bad += self_sample > RECOVERY_GATES["self_penetration_tolerance_m"]
            ground_bad += ground_sample > RECOVERY_GATES["max_ground_penetration_m"]
            samples += 1
    return {
        "audit_hz": 1 / model.opt.timestep,
        "samples": samples,
        "max_ground_penetration_m": ground_depth,
        "ground_penetration_samples": int(ground_bad),
        "max_self_penetration_m": self_depth,
        "self_collision_samples": int(self_bad),
        "joint_limit_excess_max_rad": float(limit_error.max(initial=0)),
        "joint_speed_max_rad_s": float(speed.max(initial=0)),
        "joint_speed_excess_max_rad_s": float((speed - speed_limits).max(initial=0)),
        "compared_geometry_speed_gates_passed": not (
            ground_bad or self_bad or np.any(limit_error > 1e-6)
            or np.any(speed > speed_limits + 1e-6)
        ),
        "physics_qualified": False,
        "training_eligible": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--umr", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--filename", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    row = load_row(args.baseline_root / "index.jsonl", args.filename)
    baseline_path = args.baseline_root / (row.get("reference_path") or row["attempt_reference_path"])
    baseline = MotionClip.load(baseline_path)
    robot = recovery_model("audit")
    if baseline.metadata["model_signature"] != robot.signature:
        raise ValueError("Baseline model signature differs from production K1")
    raw, source_times, fps = load_umr_qpos(args.umr, robot.model_path, args.filename)
    playback = baseline.times - baseline.times[0]
    if source_times[-1] + 1 / fps < playback[-1]:
        raise ValueError("UMR result does not cover the full baseline duration")
    indices = np.searchsorted(source_times, playback, side="right") - 1
    if np.any(indices < 0):
        raise ValueError("UMR result starts after the baseline")
    qpos = raw[indices]
    audit = audit_path(qpos)
    report = {
        "filename": args.filename,
        "original_id": row["id"],
        "family": row["family"],
        "source": "BONES-SEED SOMA proportional original",
        "controller_hz": 1 / robot.control_dt,
        "umr_source_fps": fps,
        "umr_source_frames": len(raw),
        "compared_control_ticks": len(qpos),
        "sampling": "latest UMR pose at each 50 Hz K1 tick; UMR trajectory optimization itself is offline",
        "model_signature": robot.signature,
        "baseline": {
            "method": row["retarget_version"],
            "kinematics_accepted": row["kinematics_accepted"],
            "rejected_ticks": row["rejected_ticks"],
            "recovery_audit": row.get("recovery_audit"),
        },
        "umr": audit,
        "comparison_limits": "Ground, self penetration, joint limit, and joint speed checks only; no stance slip, source-relative fidelity, controller replay, or dynamic physics qualification",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"filename": args.filename, "baseline_accepted": row["kinematics_accepted"],
                      "umr_compared_gates_pass": audit["compared_geometry_speed_gates_passed"],
                      "umr_max_ground_mm": 1000 * audit["max_ground_penetration_m"],
                      "umr_max_self_mm": 1000 * audit["max_self_penetration_m"]}))


if __name__ == "__main__":
    main()
