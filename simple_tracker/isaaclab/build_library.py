"""Build the multi-clip K1 reference library from per-clip retarget outputs.

Batched replacement for booster_train's one-clip-per-launch ``csv_to_npz.py``: body
poses come from the same Isaac Lab K1 articulation used for training (thousands of
frames per kinematic update), velocities are finite differences of those poses.

    cd ~/ws/beyondmimic && .venv-isaaclab/bin/python ~/ws/k1-mocopi/simple_tracker/isaaclab/build_library.py \
        --src ~/ws/k1-mocopi-data/k1_bandai --out ~/ws/k1-mocopi-data/k1_bandai_library.npz --headless
"""

import argparse
import json
import re
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--src", type=Path, required=True, help="directory with <clip>.npz + index.json")
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--batch", type=int, default=4096)
parser.add_argument("--exclude", type=str, default=r"_(run|dash)_", help="regex of clip names to drop")
parser.add_argument("--max_root_speed", type=float, default=1.5, help="p95 horizontal root speed [m/s]")
parser.add_argument("--max_penetration", type=float, default=0.05)
parser.add_argument("--min_root_z", type=float, default=0.30)
parser.add_argument("--max_dof_vel", type=float, default=30.0)
parser.add_argument("--min_seconds", type=float, default=0.5)
parser.add_argument(
    "--foot_link_height", type=float, default=0.027, help="foot_link origin above the sole when standing flat"
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import ArticulationCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul  # noqa: E402

from booster_assets.motions import K1_JOINT_NAMES  # noqa: E402
from booster_train.assets.robots.booster import BOOSTER_K1_CFG  # noqa: E402

TRACKED_BODIES = [
    "Trunk", "Head_2", "Left_Hip_Roll", "Left_Shank", "left_foot_link", "Right_Hip_Roll", "Right_Shank",
    "right_foot_link", "Left_Arm_2", "Left_Arm_3", "left_hand_link", "Right_Arm_2", "Right_Arm_3", "right_hand_link",
]  # fmt: skip


@configclass
class SceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = BOOSTER_K1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def select_clips(src: Path):
    index = json.loads((src / "index.json").read_text())["clips"]
    keep, dropped = [], {}
    for name, m in sorted(index.items()):
        reason = None
        if "error" in m:
            reason = "retarget_error"
        elif re.search(args.exclude, name):
            reason = "excluded_category"
        elif m["root_speed_p95"] > args.max_root_speed:
            reason = "too_fast"
        elif m["foot_penetration"] > args.max_penetration:
            reason = "penetration"
        elif m["root_z_min"] < args.min_root_z:
            reason = "too_low"
        elif m["max_dof_vel"] > args.max_dof_vel:
            reason = "dof_vel"
        elif m["seconds"] < args.min_seconds:
            reason = "too_short"
        if reason:
            dropped[reason] = dropped.get(reason, 0) + 1
        else:
            keep.append(name)
    return keep, dropped


def so3_derivative(q: torch.Tensor, dt: float) -> torch.Tensor:
    """q: (T, B, 4) wxyz -> angular velocity (T, B, 3) in world frame (central differences)."""
    if q.shape[0] < 3:
        return torch.zeros(*q.shape[:-1], 3, device=q.device)
    rel = quat_mul(q[2:], quat_conjugate(q[:-2]))
    w = axis_angle_from_quat(rel) / (2.0 * dt)
    return torch.cat([w[:1], w, w[-1:]], dim=0)


def main():
    names, dropped = select_clips(args.src)
    print(f"[build_library] keeping {len(names)} clips, dropped {dropped}")

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=args.device, dt=0.02))
    scene = InteractiveScene(SceneCfg(num_envs=args.batch, env_spacing=0.0))
    sim.reset()
    robot = scene["robot"]
    dev = sim.device
    joint_ids = robot.find_joints(K1_JOINT_NAMES, preserve_order=True)[0]  # isaac index per K1_JOINT_NAMES entry
    body_ids = robot.find_bodies(TRACKED_BODIES, preserve_order=True)[0]
    origins = scene.env_origins

    clips = [np.load(args.src / f"{n}.npz") for n in names]
    fps = int(clips[0]["fps"])
    lens = np.array([c["dof_pos"].shape[0] for c in clips])
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    F = int(lens.sum())
    root_pos = np.concatenate([c["root_pos"] for c in clips])
    root_quat = np.concatenate([c["root_quat"] for c in clips])
    dof_mj = np.concatenate([c["dof_pos"] for c in clips])
    # each clip starts at the origin in xy
    for s, n in zip(starts, lens):
        root_pos[s : s + n, :2] -= root_pos[s, :2]

    joint_pos = np.zeros((F, robot.num_joints), np.float32)
    joint_pos[:, joint_ids] = dof_mj
    body_pos = np.zeros((F, len(body_ids), 3), np.float32)
    body_quat = np.zeros((F, len(body_ids), 4), np.float32)

    default_root = robot.data.default_root_state.clone()
    for b0 in range(0, F, args.batch):
        b1 = min(F, b0 + args.batch)
        n = b1 - b0
        ids = torch.arange(n, device=dev)
        rs = default_root[:n].clone()
        rs[:, :3] = torch.tensor(root_pos[b0:b1], device=dev) + origins[:n]
        rs[:, 3:7] = torch.tensor(root_quat[b0:b1], device=dev)
        rs[:, 7:] = 0.0
        robot.write_root_state_to_sim(rs, env_ids=ids)
        jp = torch.tensor(joint_pos[b0:b1], device=dev)
        robot.write_joint_state_to_sim(jp, torch.zeros_like(jp), env_ids=ids)
        sim.forward()
        sim.render()
        scene.update(sim.get_physics_dt())
        body_pos[b0:b1] = (robot.data.body_pos_w[:n][:, body_ids] - origins[:n, None]).cpu().numpy()
        body_quat[b0:b1] = robot.data.body_quat_w[:n][:, body_ids].cpu().numpy()
        if b0 == 0:
            err = np.abs(body_pos[:n, 0] - root_pos[:n]).max()
            print(f"[build_library] root pose check (should be ~0): {err:.2e}")
        print(f"[build_library] {b1}/{F}", flush=True)

    # Re-ground in the training model: the URDF hangs the legs 1.5 cm lower than the
    # MJCF used for retargeting, so the robust lowest foot must be put back on the floor.
    feet = [TRACKED_BODIES.index("left_foot_link"), TRACKED_BODIES.index("right_foot_link")]
    shifts = []
    for s, n in zip(starts, lens):
        sl = slice(s, s + n)
        dz = args.foot_link_height - np.percentile(body_pos[sl][:, feet, 2].min(axis=1), 5)
        body_pos[sl, :, 2] += dz
        shifts.append(dz)
    print(f"[build_library] ground shift mean {np.mean(shifts):.4f} m, range [{np.min(shifts):.4f}, {np.max(shifts):.4f}]")

    dt = 1.0 / fps
    joint_vel = np.zeros_like(joint_pos)
    lin_vel = np.zeros_like(body_pos)
    ang_vel = np.zeros_like(body_pos)
    for s, n in zip(starts, lens):
        sl = slice(s, s + n)
        joint_vel[sl] = np.gradient(joint_pos[sl], dt, axis=0) if n > 1 else 0.0
        lin_vel[sl] = np.gradient(body_pos[sl], dt, axis=0) if n > 1 else 0.0
        ang_vel[sl] = so3_derivative(torch.tensor(body_quat[sl], device=dev), dt).cpu().numpy()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        fps=fps,
        joint_names=np.array(robot.joint_names),
        body_names=np.array(TRACKED_BODIES),
        clip_names=np.array(names),
        clip_start=starts,
        clip_len=lens,
        joint_pos=joint_pos,
        joint_vel=joint_vel.astype(np.float32),
        body_pos_w=body_pos,
        body_quat_w=body_quat,
        body_lin_vel_w=lin_vel.astype(np.float32),
        body_ang_vel_w=ang_vel.astype(np.float32),
    )
    summary = {"clips": len(names), "frames": F, "hours": F / fps / 3600, "dropped": dropped}
    Path(str(args.out) + ".json").write_text(json.dumps(summary, indent=1))
    print("[build_library] wrote", args.out, summary)


if __name__ == "__main__":
    main()
    simulation_app.close()
