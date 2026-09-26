"""mocopi -> K1 reference, one frame at a time (causal), using GMR's ``bvh_mocopi`` IK config.

Run with the GMR venv (GMR checkout with the mocopi IK config, see memory notes):

  # offline check: replay a mocopi BVH through the live path, write a tracker clip for sim2sim
  ~/ws/mocopi2beyondmimic-ref/GMR/.venv/bin/python simple_tracker/live/mocopi_retarget.py bvh \
      ~/ws/mocopi2beyondmimic-ref/mocopi_samples/MCPM_20260922_164010.bvh --out mocopi_live_check.npz

  # test the robot-side UDP path without a mocopi device: stream a BVH in real time
  ~/ws/mocopi2beyondmimic-ref/GMR/.venv/bin/python simple_tracker/live/mocopi_retarget.py bvh <file.bvh> --stream

  # live: mocopi app UDP (port 12351) -> K1 qpos to booster_deploy (127.0.0.1:12400)
  ~/ws/mocopi2beyondmimic-ref/GMR/.venv/bin/python simple_tracker/live/mocopi_retarget.py udp

Scale uses the K1/human leg-length ratio (as for the Bandai training data) and the
ground is calibrated once from the first second (stand still at start).
UDP mode is untested against a real device: the conversion assumes the UDP bone
local transforms match the app's BVH export (checked only on BVH so far).
"""

from __future__ import annotations

import argparse
import os
import socket
import struct
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
GMR_ROOT = Path(os.environ.get("GMR_ROOT", Path.home() / "ws/mocopi2beyondmimic-ref/GMR"))
sys.path.insert(0, str(GMR_ROOT))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))

import mujoco as mj  # noqa: E402
from causal_reference import CausalReference  # noqa: E402

K1_HIP_TO_ANKLE = 0.437
K1_FOOT_LINK_STANDING_Z = 0.025  # MJCF convention (CausalReference adds the URDF offset)
CALIB_FRAMES = 50
OUT_PORT = 12400
Y_UP_TO_Z_UP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
Q_Y_UP_TO_Z_UP = R.from_matrix(Y_UP_TO_Z_UP)


def leg_length(frame) -> float:
    return float(
        np.mean(
            [
                np.linalg.norm(frame[f"{s}_low_leg"][0] - frame[f"{s}_up_leg"][0])
                + np.linalg.norm(frame[f"{s}_foot"][0] - frame[f"{s}_low_leg"][0])
                for s in ("l", "r")
            ]
        )
    )


class MocopiToK1:
    def __init__(self):
        from general_motion_retargeting import GeneralMotionRetargeting

        self.rt = GeneralMotionRetargeting("bvh_mocopi", "booster_k1", verbose=False)
        self.model = self.rt.model
        self.data = mj.MjData(self.model)
        self.feet = [mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, n) for n in ("left_foot_link", "right_foot_link")]
        self.scale = None
        self.ground = None
        self._calib = []

    def _foot_min_z(self, qpos):
        self.data.qpos[:] = qpos
        mj.mj_kinematics(self.model, self.data)
        return min(self.data.xpos[i, 2] for i in self.feet)

    def step(self, frame):
        """frame: GMR human dict (bone -> [pos, quat wxyz], z-up, m). Returns grounded qpos or None while calibrating."""
        if self.scale is None:
            self.scale = K1_HIP_TO_ANKLE / leg_length(frame)
            for k in self.rt.human_scale_table:
                self.rt.human_scale_table[k] = self.scale
            for _ in range(20):  # converge from the zero pose
                q = self.rt.retarget(frame)
        q = self.rt.retarget(frame).copy()
        if self.ground is None:
            self._calib.append(self._foot_min_z(q))
            if len(self._calib) < CALIB_FRAMES:
                return None
            self.ground = K1_FOOT_LINK_STANDING_Z - float(np.min(self._calib))
            print(f"[mocopi] calibrated: scale {self.scale:.3f}, ground offset {self.ground:+.3f} m", flush=True)
        q[2] += self.ground
        return q


def udp_packet_to_frame(packet):
    """k1_motion.mocopi Packet (local bone transforms, Sony y-up) -> GMR human dict (z-up)."""
    from k1_motion.mocopi import NAMES, PARENTS

    local_q = R.from_quat(packet.transforms[:, :4])  # xyzw
    local_p = packet.transforms[:, 4:7]
    g_r, g_p = [None] * 27, np.zeros((27, 3))
    for i, parent in enumerate(PARENTS):
        if parent < 0:
            g_r[i], g_p[i] = local_q[i], local_p[i]
        else:
            g_r[i] = g_r[parent] * local_q[i]
            g_p[i] = g_p[parent] + g_r[parent].apply(local_p[i])
    frame = {}
    for i, name in enumerate(NAMES):
        frame[name] = [Y_UP_TO_Z_UP @ g_p[i], (Q_Y_UP_TO_Z_UP * g_r[i]).as_quat(scalar_first=True)]
    frame["LeftFootMod"] = list(frame["l_foot"])
    frame["RightFootMod"] = list(frame["r_foot"])
    return frame


def run_bvh(path: Path, out: Path | None, stream: tuple[str, int] | None = None):
    from general_motion_retargeting.utils.lafan1 import load_bvh_file

    frames, _ = load_bvh_file(str(path), format="mocopi")  # drops the bind-pose frame 0
    conv = MocopiToK1()
    causal = CausalReference(fps=50.0)
    rows, qposes = [], []
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if stream else None
    t_next = time.monotonic()
    for fr in frames:
        q = conv.step(fr)
        if q is None:
            continue
        qposes.append(q)
        rows.append(causal.update(q))
        if tx is not None:  # real-time 50 Hz stream, same packet format as udp mode
            t_next += 0.02
            time.sleep(max(0.0, t_next - time.monotonic()))
            tx.sendto(struct.pack("<30d", float(len(rows)), *q.astype(np.float64)), stream)
    if out is None:
        return
    stack = lambda k: np.stack([r[k] for r in rows]).astype(np.float32)  # noqa: E731
    n = len(rows)
    np.savez(
        out,
        fps=50,
        clip_names=np.array([path.stem]),
        clip_start=np.array([0]),
        clip_len=np.array([n]),
        joint_pos=stack("joint_pos"),
        joint_vel=stack("joint_vel"),
        body_pos_w=stack("root_pos")[:, None],
        body_quat_w=stack("root_quat")[:, None],
        body_lin_vel_w=stack("root_lin_vel_w")[:, None],
        body_ang_vel_w=stack("root_ang_vel_w")[:, None],
        qpos_mjcf=np.stack(qposes).astype(np.float32),
    )
    print(f"wrote {out}: {n} frames ({n / 50:.1f}s) after {CALIB_FRAMES}-frame calibration")


def run_udp(listen_port: int, out_host: str, out_port: int):
    from k1_motion.mocopi import decode_packet

    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("0.0.0.0", listen_port))
    rx.settimeout(0.5)
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    conv = MocopiToK1()
    last_number, seq, peer = None, 0, None
    print(f"[mocopi] listening on udp/{listen_port}, sending K1 qpos to {out_host}:{out_port}", flush=True)
    while True:
        try:
            raw, src = rx.recvfrom(65535)
        except socket.timeout:
            continue
        if peer is None:
            peer = src  # pin the first sender, like k1_motion.mocopi.UdpReceiver
        if src != peer:
            continue
        try:
            pkt = decode_packet(raw)
        except ValueError as exc:
            print("[mocopi] rejected packet:", exc, flush=True)
            continue
        if pkt.kind != "frame":
            continue
        # drop duplicates / out-of-order frames (uint32 wrap-aware)
        if last_number is not None and not 0 < (pkt.frame_number - last_number) & 0xFFFFFFFF < 0x80000000:
            continue
        last_number = pkt.frame_number
        q = conv.step(udp_packet_to_frame(pkt))
        if q is None:
            continue
        seq += 1
        tx.sendto(struct.pack("<30d", float(seq), *q.astype(np.float64)), (out_host, out_port))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    b = sub.add_parser("bvh")
    b.add_argument("bvh", type=Path)
    b.add_argument("--out", type=Path, default=None)
    b.add_argument("--stream", action="store_true", help="also send qpos in real time like udp mode")
    b.add_argument("--to-host", default="127.0.0.1")
    b.add_argument("--to-port", type=int, default=OUT_PORT)
    u = sub.add_parser("udp")
    u.add_argument("--listen", type=int, default=12351)
    u.add_argument("--to-host", default="127.0.0.1")
    u.add_argument("--to-port", type=int, default=OUT_PORT)
    a = ap.parse_args()
    if a.mode == "bvh":
        run_bvh(a.bvh, a.out, (a.to_host, a.to_port) if a.stream else None)
    else:
        run_udp(a.listen, a.to_host, a.to_port)


if __name__ == "__main__":
    main()
