#!/usr/bin/env python3
"""Prepare one admitted K1 reference and train it with Booster Train's native task.

This is a new native BeyondMimic experiment, not a continuation of K1's universal
causal learner. Preparation works without Isaac; training requires its interpreter.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party/booster_assets/src"))

from k1_motion.contracts import JOINT_NAMES, MotionClip  # noqa: E402
from k1_motion.robot import K1Model  # noqa: E402

TASK = "Booster-K1-MJ_Dance_004-v0"
BOOSTER_REVISION = "651b7a53f2ffaf2d5629d0604d065cc385e29c6b"


def save_native_motion(robot, clip, path):
    """Write the fields expected by Booster MotionLoader from the pinned MJCF FK."""
    names = [robot.model.body(i).name for i in range(1, robot.model.nbody)]
    count, bodies = len(clip.times), len(names)
    fields = {
        "joint_pos": clip.values["joint_position"].astype(np.float32),
        "joint_vel": clip.values["joint_velocity"].astype(np.float32),
        "body_pos_w": np.empty((count, bodies, 3), dtype=np.float32),
        "body_quat_w": np.empty((count, bodies, 4), dtype=np.float32),
        "body_lin_vel_w": np.empty((count, bodies, 3), dtype=np.float32),
        "body_ang_vel_w": np.empty((count, bodies, 3), dtype=np.float32),
        "joint_names": np.asarray(JOINT_NAMES), "body_names": np.asarray(names),
        "fps": np.asarray(round(1 / robot.control_dt)),
    }
    velocity = np.empty(6, dtype=float)
    for frame in range(count):
        quat = clip.values["root_orientation"][frame]
        robot.data.qpos[:] = np.r_[clip.values["root_position"][frame], quat,
                                   clip.values["joint_position"][frame]]
        root_velocity = clip.values["root_velocity"][frame]
        robot.data.qvel[:3] = root_velocity[:3]
        robot.data.qvel[3:6] = Rotation.from_quat(quat[[1, 2, 3, 0]]).inv().apply(root_velocity[3:])
        robot.data.qvel[6:] = clip.values["joint_velocity"][frame]
        mujoco.mj_forward(robot.model, robot.data)
        for index in range(bodies):
            body_id = index + 1
            fields["body_pos_w"][frame, index] = robot.data.xpos[body_id]
            fields["body_quat_w"][frame, index] = robot.data.xquat[body_id]
            mujoco.mj_objectVelocity(robot.model, robot.data, mujoco.mjtObj.mjOBJ_BODY,
                                     body_id, velocity, 0)
            fields["body_ang_vel_w"][frame, index] = velocity[:3]
            fields["body_lin_vel_w"][frame, index] = velocity[3:]
    if any(not np.isfinite(v).all() for k, v in fields.items() if k not in ("body_names", "joint_names")):
        raise ValueError("Nonfinite native motion fields")
    np.savez(path, **fields)


def selected_row(library, motion_id, signature):
    rows = [json.loads(line) for line in (library / "index.jsonl").read_text().splitlines()]
    matches = [row for row in rows if row.get("id") == motion_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one reference with id {motion_id!r}; found {len(matches)}")
    row = matches[0]
    if row.get("split") != "train" or not row.get("training_eligible", True):
        raise ValueError("Selected reference is not training_eligible in the train split")
    if not row.get("kinematics_accepted", False):
        raise ValueError("Selected reference lacks accepted kinematics")
    if row.get("model_signature") != signature:
        raise ValueError("Reference robot model signature mismatch")
    if row.get("mirrored", False) or row.get("is_mirror", False):
        raise ValueError("Mirrored reference requires an explicit separate experiment")
    return row


def prepare(library, motion_id, output):
    robot = K1Model()
    row = selected_row(library, motion_id, robot.signature)
    relative_path = Path(row["reference_path"])
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError("Reference path escapes the library")
    # The current corpus intentionally uses symlinks into shared clip storage.
    clip_path = (library / relative_path).resolve(strict=True)
    clip = MotionClip.load(clip_path)
    if clip.metadata.get("model_signature") != robot.signature:
        raise ValueError("Reference payload robot model signature mismatch")
    if not np.all(clip.values["valid"]):
        raise ValueError("Reference contains invalid ticks")
    if not np.allclose(np.diff(clip.times), robot.control_dt, rtol=0, atol=1e-6):
        raise ValueError("Native Booster export requires a uniformly sampled 50 Hz reference")
    booster_names = __import__("booster_assets.motions", fromlist=["K1_JOINT_NAMES"]).K1_JOINT_NAMES
    aliases = {"aaleft_shoulder_pitch_joint": "left_shoulder_pitch_joint",
               "aaright_shoulder_pitch_joint": "right_shoulder_pitch_joint"}
    canonical = [aliases.get(name, name) for name in JOINT_NAMES]
    if list(booster_names) != canonical:
        raise ValueError("Booster K1 CSV joint order changed")
    quat = clip.values["root_orientation"]
    # Booster csv_to_npz.py expects xyzw; our reference contract uses wxyz.
    csv = np.concatenate((clip.values["root_position"], quat[:, [1, 2, 3, 0]],
                          clip.values["joint_position"]), axis=1)
    if csv.shape[1] != 29 or not np.isfinite(csv).all():
        raise ValueError("Native motion export is invalid")
    output.mkdir(parents=True, exist_ok=False)
    np.savetxt(output / "motion.csv", csv, delimiter=",", fmt="%.9g")
    save_native_motion(robot, clip, output / "motion.npz")
    receipt = {
        "status": "prepared_only", "native_task": TASK, "booster_train_revision": BOOSTER_REVISION,
        "library": str(library.resolve()), "motion_id": motion_id,
        "reference_path": str(clip_path), "model_signature": robot.signature,
        "frames": len(clip.times), "fps": round(1 / robot.control_dt),
        "joint_order": canonical, "split": row["split"],
        "training_eligible": True, "physics_qualified": bool(row.get("physics_qualified", False)),
        "capture_group": row.get("capture_group"), "family": row.get("family"),
    }
    (output / "prepare.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--motion-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.num_envs < 1 or args.iterations < 1:
        parser.error("num-envs and iterations must be positive")
    library, output = args.library.resolve(), args.output.resolve()
    receipt = prepare(library, args.motion_id, output)
    print(json.dumps(receipt, indent=2))
    if args.prepare_only:
        return
    upstream = ROOT / "third_party/booster_train"
    paths = [ROOT / "src", ROOT / "third_party/booster_assets/src",
             upstream / "source/booster_train"]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(map(str, paths)) + os.pathsep + env.get("PYTHONPATH", "")
    motion = output / "motion.npz"
    with np.load(motion) as data:
        required = {"joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
                    "body_lin_vel_w", "body_ang_vel_w", "joint_names", "body_names", "fps"}
        if not required.issubset(data.files) or data["joint_pos"].shape[0] < 2:
            raise ValueError("Booster motion conversion did not produce a trainable NPZ")
    receipt["status"] = "converted"
    (output / "prepare.json").write_text(json.dumps(receipt, indent=2) + "\n")
    subprocess.run([sys.executable, str(ROOT / "scripts/booster_native_train.py"),
                    "--motion", str(motion), "--output", str(output / "training"),
                    "--num-envs", str(args.num_envs), "--iterations", str(args.iterations),
                    "--seed", str(args.seed), "--device", args.device], env=env, check=True)


if __name__ == "__main__":
    main()
