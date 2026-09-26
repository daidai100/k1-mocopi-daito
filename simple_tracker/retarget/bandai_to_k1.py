"""Batch retarget BVH clips (Bandai Namco, LAFAN1) to Booster K1 with GMR.

Run with the GMR venv (it only needs GMR + mujoco + numpy/scipy):

    ~/ws/mocopi2beyondmimic-ref/GMR/.venv/bin/python simple_tracker/retarget/bandai_to_k1.py \
        --bandai ~/ws/k1-mocopi-data/bandai/dataset --out ~/ws/k1-mocopi-data/k1_bandai --workers 20
    # LAFAN1: long takes are cut into usable segments (see ``segment_mask``)
    ... bandai_to_k1.py --format lafan1 --bandai ~/ws/k1-mocopi-data/lafan1 --out ~/ws/k1-mocopi-data/k1_lafan1 \
        --calib dance1_subject1.bvh --calib_frame auto --segment

GMR itself is not modified. This script
  * loads Bandai BVH (6 channels per joint, cm, y-up) with GMR's vendored parser,
  * derives the GMR rotation offsets automatically from a neutral standing frame,
    instead of hand-copying another source's IK table,
  * scales each actor by leg length (K1 hip->ankle / human hip->ankle),
  * grounds each clip so the feet rest on z=0,
  * writes one ``<clip>.npz`` per clip (50 Hz root pos, root quat wxyz, 22 dof in
    booster_assets K1_JOINT_NAMES order) plus ``index.json`` with quality metrics.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

GMR_ROOT = Path(os.environ.get("GMR_ROOT", Path.home() / "ws/mocopi2beyondmimic-ref/GMR"))
sys.path.insert(0, str(GMR_ROOT))

import mujoco as mj  # noqa: E402
import general_motion_retargeting.utils.lafan_vendor.utils as bvh_utils  # noqa: E402
from general_motion_retargeting import params  # noqa: E402
from general_motion_retargeting.utils.lafan_vendor.extract import read_bvh  # noqa: E402

HERE = Path(__file__).resolve().parent
IK_CONFIG = HERE / "bvh_bandai_to_k1.json"  # replaced in main() per --format (before the worker fork)
OUT_FPS = 50
SEGMENT = False

# Source bone names -> the Bandai names used throughout this file.
BONE_ALIASES = {
    "bandai": {},
    "lafan1": {
        "LeftUpLeg": "UpperLeg_L", "LeftLeg": "LowerLeg_L", "LeftFoot": "Foot_L", "LeftToe": "Toes_L",
        "RightUpLeg": "UpperLeg_R", "RightLeg": "LowerLeg_R", "RightFoot": "Foot_R", "RightToe": "Toes_R",
        "LeftArm": "UpperArm_L", "LeftForeArm": "LowerArm_L", "LeftHand": "Hand_L", "LeftShoulder": "Shoulder_L",
        "RightArm": "UpperArm_R", "RightForeArm": "LowerArm_R", "RightHand": "Hand_R", "RightShoulder": "Shoulder_R",
    },
}  # fmt: skip
FORMAT = "bandai"

# K1 body <- Bandai bone, (table1 pos, table1 rot, table2 pos, table2 rot).
# The K1 arm is ~0.16 m long, so like GMR's smplx_to_k1 table the K1 hand is
# matched to the (scaled) human elbow.  Arm orientation costs are low because a
# 4-dof arm cannot follow a 7-dof human arm's orientation exactly.
BODY_MAP = {
    "Trunk": ("Hips", 0, 10, 10, 5),
    "Head_2": ("Head", 0, 5, 5, 2),
    "Left_Hip_Yaw": ("UpperLeg_L", 0, 10, 10, 5),
    "Right_Hip_Yaw": ("UpperLeg_R", 0, 10, 10, 5),
    "Left_Shank": ("LowerLeg_L", 0, 10, 10, 5),
    "Right_Shank": ("LowerLeg_R", 0, 10, 10, 5),
    "left_foot_link": ("Foot_L", 0, 10, 100, 50),
    "right_foot_link": ("Foot_R", 0, 10, 100, 50),
    "Left_Arm_3": ("UpperArm_L", 0, 2, 10, 1),
    "Right_Arm_3": ("UpperArm_R", 0, 2, 10, 1),
    "left_hand_link": ("LowerArm_L", 0, 2, 10, 1),
    "right_hand_link": ("LowerArm_R", 0, 2, 10, 1),
}

K1_HIP_TO_ANKLE = 0.437  # Left_Hip_Pitch -> left_foot_link at zero pose (K1_serial.xml)
K1_FOOT_LINK_STANDING_Z = 0.025  # foot_link origin above the sole box bottom

Y_UP_TO_Z_UP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
Y_UP_TO_Z_UP_QUAT = R.from_matrix(Y_UP_TO_Z_UP).as_quat(scalar_first=True)


def load_bandai(path: str):
    """Returns (pos[T,B,3] m, quat[T,B,4] wxyz, bone_names, fps) in z-up, bones renamed to Bandai names."""
    anim = read_bvh(path)
    gq, gp = bvh_utils.quat_fk(anim.quats, anim.pos, anim.parents)
    with open(path) as f:
        frame_time = next(float(line.split(":")[1]) for line in f if line.startswith("Frame Time"))
    pos = gp @ Y_UP_TO_Z_UP.T / 100.0
    quat = bvh_utils.quat_mul(Y_UP_TO_Z_UP_QUAT, gq)
    aliases = BONE_ALIASES[FORMAT]
    return pos, quat, [aliases.get(b, b) for b in anim.bones], 1.0 / frame_time


def leg_length(pos, bones):
    """Mean hip->ankle distance over both legs (constant per actor, taken from frame 0)."""
    i = {b: k for k, b in enumerate(bones)}
    lengths = []
    for s in ("L", "R"):
        a, b, c = pos[0, i[f"UpperLeg_{s}"]], pos[0, i[f"LowerLeg_{s}"]], pos[0, i[f"Foot_{s}"]]
        lengths.append(np.linalg.norm(b - a) + np.linalg.norm(c - b))
    return float(np.mean(lengths))


def facing_yaw(pos, bones, t=0):
    i = {b: k for k, b in enumerate(bones)}
    left = pos[t, i["UpperLeg_L"]] - pos[t, i["UpperLeg_R"]]
    fwd = np.cross(left, [0.0, 0.0, 1.0])
    return float(np.arctan2(fwd[1], fwd[0]))


def robot_calibration_pose(model, arm_abduction):
    """K1 standing pose matching the calibration human: legs straight, arms hanging."""
    data = mj.MjData(model)
    data.qpos[3] = 1.0
    names = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, j) for j in range(model.njnt)]
    roll = np.pi / 2 - arm_abduction
    data.qpos[model.jnt_qposadr[names.index("Left_Shoulder_Roll")]] = -roll
    data.qpos[model.jnt_qposadr[names.index("Right_Shoulder_Roll")]] = roll
    mj.mj_kinematics(model, data)
    return data


def pick_calibration_frame(pos, bones, fps) -> int:
    """Straightest-legged frame with both feet at the same height within the first 30 s."""
    i = {b: k for k, b in enumerate(bones)}
    best, best_t = -1.0, 0
    for t in range(min(len(pos), int(30 * fps))):
        if abs(pos[t, i["Foot_L"], 2] - pos[t, i["Foot_R"], 2]) > 0.02:
            continue
        straight = []
        for s in ("L", "R"):
            a = pos[t, i[f"UpperLeg_{s}"]] - pos[t, i[f"LowerLeg_{s}"]]
            b = pos[t, i[f"Foot_{s}"]] - pos[t, i[f"LowerLeg_{s}"]]
            straight.append(-np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b))  # 1 = straight knee
        if min(straight) > best:
            best, best_t = min(straight), t
    return best_t


def calibrate(calib_bvh: str, out_path: Path, frame="0"):
    """Compute GMR rotation offsets so the neutral human frame maps to the K1 neutral pose.

    GMR sets each target to ``R_human * R_offset``; choosing
    ``R_offset = R_human_cal^-1 * Rz(yaw_cal) * R_robot_cal`` makes a neutral human
    produce the neutral robot, whatever the rig's local bone axes are.
    """
    pos, quat, bones, fps = load_bandai(calib_bvh)
    t0 = pick_calibration_frame(pos, bones, fps) if frame == "auto" else int(frame)
    pos, quat = pos[t0:], quat[t0:]
    i = {b: k for k, b in enumerate(bones)}
    yaw = facing_yaw(pos, bones)
    # abduction of the human upper arm from vertical, averaged over both arms
    abd = []
    for s in ("L", "R"):
        v = pos[0, i[f"LowerArm_{s}"]] - pos[0, i[f"UpperArm_{s}"]]
        abd.append(np.arccos(np.clip(-v[2] / np.linalg.norm(v), -1, 1)))
    model = mj.MjModel.from_xml_path(str(params.ROBOT_XML_DICT["booster_k1"]))
    data = robot_calibration_pose(model, float(np.mean(abd)))
    rz = R.from_euler("z", yaw)

    table1, table2 = {}, {}
    for body, (bone, p1, r1, p2, r2) in BODY_MAP.items():
        r_h = R.from_quat(quat[0, i[bone]], scalar_first=True)
        r_r = R.from_quat(data.xquat[mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body)], scalar_first=True)
        off = (r_h.inv() * rz * r_r).as_quat(scalar_first=True).tolist()
        table1[body] = [bone, p1, r1, [0.0, 0.0, 0.0], off]
        table2[body] = [bone, p2, r2, [0.0, 0.0, 0.0], off]
    cfg = {
        "robot_root_name": "Trunk",
        "human_root_name": "Hips",
        "ground_height": 0.0,
        "human_height_assumption": 1.8,
        "use_ik_match_table1": True,
        "use_ik_match_table2": True,
        # replaced per clip by the leg-length ratio (see retarget_clip)
        "human_scale_table": {bone: 0.55 for bone, *_ in BODY_MAP.values()},
        "ik_match_table1": table1,
        "ik_match_table2": table2,
        "_calibration": {"bvh": os.path.basename(calib_bvh), "frame": t0, "arm_abduction": float(np.mean(abd))},
    }
    out_path.write_text(json.dumps(cfg, indent=1))
    return cfg


def make_retargeter():
    key = f"bvh_{FORMAT}_auto"
    params.IK_CONFIG_DICT.setdefault(key, {})["booster_k1"] = IK_CONFIG
    from general_motion_retargeting import GeneralMotionRetargeting

    return GeneralMotionRetargeting(key, "booster_k1", verbose=False)


def segment_mask(root_pos, root_quat, dof, feet, fps=OUT_FPS):
    """Frames a flat-ground standing tracker should learn from (for long mixed takes like LAFAN1).

    Drops lying/crawling (trunk tilt, low trunk), fast running (1 s mean speed), flight
    phases (both feet high), and violent joint motion; invalid spans are padded by 0.5 s.
    """
    up = 1.0 - 2.0 * (root_quat[:, 1] ** 2 + root_quat[:, 2] ** 2)
    speed = np.linalg.norm(np.gradient(root_pos[:, :2], 1.0 / fps, axis=0), axis=1)
    speed_1s = np.convolve(speed, np.ones(int(fps)) / fps, mode="same")
    dof_vel = np.abs(np.gradient(dof, 1.0 / fps, axis=0)).max(axis=1)
    ok = (up > 0.7) & (root_pos[:, 2] > 0.40) & (speed_1s < 1.3) & (feet.min(axis=1) < 0.10) & (dof_vel < 25.0)
    pad = int(0.5 * fps)
    bad = np.convolve(~ok, np.ones(2 * pad + 1), mode="same") > 0
    return ~bad


def split_segments(mask, min_len):
    segs, start = [], None
    for t, v in enumerate(np.r_[mask, False]):
        if v and start is None:
            start = t
        elif not v and start is not None:
            if t - start >= min_len:
                segs.append((start, t))
            start = None
    return segs


def resample(root_pos, root_quat, dof, fps_in, fps_out=OUT_FPS):
    n = root_pos.shape[0]
    t_in = np.arange(n) / fps_in
    t_out = np.arange(0.0, t_in[-1] + 1e-9, 1.0 / fps_out)
    lerp = lambda x: np.stack([np.interp(t_out, t_in, x[:, k]) for k in range(x.shape[1])], axis=1)  # noqa: E731
    rot = Slerp(t_in, R.from_quat(root_quat, scalar_first=True))(t_out)
    return lerp(root_pos), rot.as_quat(scalar_first=True), lerp(dof)


def foot_heights(model, data, qpos):
    lf = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "left_foot_link")
    rf = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "right_foot_link")
    out = np.zeros((qpos.shape[0], 2))
    for t in range(qpos.shape[0]):
        data.qpos[:] = qpos[t]
        mj.mj_kinematics(model, data)
        out[t] = data.xpos[lf, 2], data.xpos[rf, 2]
    return out


_RETARGETER = None


def retarget_clip(args):
    bvh_path, out_dir = args
    global _RETARGETER
    name = Path(bvh_path).stem
    try:
        if _RETARGETER is None:
            _RETARGETER = make_retargeter()
        rt = _RETARGETER
        pos, quat, bones, fps = load_bandai(bvh_path)
        i = {b: k for k, b in enumerate(bones)}
        scale = K1_HIP_TO_ANKLE / leg_length(pos, bones)
        for key in rt.human_scale_table:
            rt.human_scale_table[key] = scale
        # fresh IK state per clip, initialised upright at the first root position
        rt.configuration.data.qpos[:] = 0.0
        rt.configuration.data.qpos[3] = 1.0
        rt.configuration.update(rt.configuration.data.qpos)

        needed = {bone for bone, *_ in BODY_MAP.values()}
        qpos, err = [], []
        for t in range(pos.shape[0]):
            frame = {b: [pos[t, i[b]].copy(), quat[t, i[b]].copy()] for b in needed}
            # the first frame gets extra IK iterations to converge from the zero pose
            for _ in range(20 if t == 0 else 1):
                q = rt.retarget(frame)
            qpos.append(q.copy())
            err.append(rt.error2())
        qpos = np.asarray(qpos)

        root_pos, root_quat, dof = resample(qpos[:, :3], qpos[:, 3:7], qpos[:, 7:], fps)
        # light smoothing against IK jitter (sigma = 1 frame at 50 Hz)
        dof = gaussian_filter1d(dof, sigma=1.0, axis=0, mode="nearest")
        root_pos = gaussian_filter1d(root_pos, sigma=1.0, axis=0, mode="nearest")

        # ground: robust lowest foot_link height -> standing foot_link height
        model, data = rt.model, mj.MjData(rt.model)
        qp = np.concatenate([root_pos, root_quat, dof], axis=1)
        feet = foot_heights(model, data, qp)
        root_pos[:, 2] += K1_FOOT_LINK_STANDING_Z - np.percentile(feet.min(axis=1), 5)
        feet += K1_FOOT_LINK_STANDING_Z - np.percentile(feet.min(axis=1), 5)

        dt = 1.0 / OUT_FPS
        lo = model.jnt_range[1:, 0]
        hi = model.jnt_range[1:, 1]
        segs = split_segments(segment_mask(root_pos, root_quat, dof, feet), 3 * OUT_FPS) if SEGMENT else [(0, len(dof))]
        results = []
        for k, (a, b) in enumerate(segs):
            seg_name = f"{name}_s{k:02d}" if SEGMENT else name
            d, rp, rq, ft = dof[a:b], root_pos[a:b], root_quat[a:b], feet[a:b]
            dof_vel = np.gradient(d, dt, axis=0)
            root_vel = np.gradient(rp, dt, axis=0)
            metrics = {
                "frames": int(d.shape[0]),
                "seconds": float(d.shape[0] * dt),
                "scale": float(scale),
                "ik_err_mean": float(np.mean(err)),
                "ik_err_p95": float(np.percentile(err, 95)),
                "max_dof_vel": float(np.abs(dof_vel).max()),
                "p99_dof_vel": float(np.percentile(np.abs(dof_vel), 99)),
                "root_speed_p95": float(np.percentile(np.linalg.norm(root_vel[:, :2], axis=1), 95)),
                "root_z_min": float(rp[:, 2].min()),
                "foot_penetration": float(K1_FOOT_LINK_STANDING_Z - ft.min()),
                "limit_margin_min": float(np.min(np.minimum(d - lo, hi - d))),
            }
            if SEGMENT:
                metrics["source"] = name
                metrics["source_frames"] = [int(a), int(b)]
            np.savez_compressed(
                Path(out_dir) / f"{seg_name}.npz",
                fps=OUT_FPS,
                root_pos=rp.astype(np.float32),
                root_quat=rq.astype(np.float32),
                dof_pos=d.astype(np.float32),
            )
            results.append((seg_name, metrics))
        return name, results, None
    except Exception as exc:  # keep the batch going, record the failure
        return name, None, repr(exc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bandai", type=Path, required=True, help="dataset root (searched recursively for *.bvh)")
    ap.add_argument("--format", choices=sorted(BONE_ALIASES), default="bandai")
    ap.add_argument("--calib_frame", default="0", help="frame index or 'auto' (straightest legs)")
    ap.add_argument("--segment", action="store_true", help="cut long takes into usable segments")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="only the first N clips (smoke test)")
    ap.add_argument("--match", type=str, default="", help="only clips whose name contains this")
    ap.add_argument(
        "--calib",
        type=str,
        default="Bandai-Namco-Research-Motiondataset-2/data/dataset-2_raise-up-both-hands_normal_001.bvh",
    )
    a = ap.parse_args()
    global FORMAT, IK_CONFIG, SEGMENT
    FORMAT, SEGMENT = a.format, a.segment
    IK_CONFIG = HERE / f"bvh_{a.format}_to_k1.json"
    a.out.mkdir(parents=True, exist_ok=True)
    cfg = calibrate(str(a.bandai / a.calib), IK_CONFIG, a.calib_frame)
    print("calibration:", cfg["_calibration"])

    clips = sorted(str(p) for p in a.bandai.rglob("*.bvh") if a.match in p.name)
    if a.limit:
        clips = clips[: a.limit]
    print(f"{len(clips)} clips -> {a.out}")
    index = {}
    with mp.Pool(a.workers) as pool:
        for k, (name, results, error) in enumerate(pool.imap_unordered(retarget_clip, [(c, a.out) for c in clips])):
            if error is not None:
                index[name] = {"error": error}
            for seg_name, metrics in results or []:
                index[seg_name] = metrics
            if k % 100 == 0 or error or SEGMENT:
                secs = sum(m["seconds"] for _, m in results or [])
                print(k, name, error or f"{len(results)} segment(s) {secs:.1f}s", flush=True)
    with tempfile.NamedTemporaryFile("w", dir=a.out, delete=False) as f:
        json.dump({"ik_config": cfg, "clips": index}, f, indent=1)
    os.replace(f.name, a.out / "index.json")
    print("done", sum("error" not in v for v in index.values()), "clips/segments,", len(clips), "sources")


if __name__ == "__main__":
    main()
