"""A concatenated multi-clip K1 reference library (built by ``build_library.py``)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch


class MotionLibrary:
    """All clips concatenated along time; ``clip_start``/``clip_len`` index into the frame axis.

    Stored per frame: joint pos/vel (robot joint order) and pose/velocity of the tracked
    bodies in each clip's own world frame.
    """

    def __init__(self, path: str, body_names: Sequence[str], joint_names: Sequence[str], device: str):
        data = np.load(path, allow_pickle=False)
        self.fps = float(data["fps"])
        lib_bodies = data["body_names"].tolist()
        lib_joints = data["joint_names"].tolist()
        b = [lib_bodies.index(n) for n in body_names]
        j = [lib_joints.index(n) for n in joint_names]
        t = lambda x: torch.tensor(x, dtype=torch.float32, device=device)  # noqa: E731
        self.joint_pos = t(data["joint_pos"][:, j])
        self.joint_vel = t(data["joint_vel"][:, j])
        self.body_pos_w = t(data["body_pos_w"][:, b])
        self.body_quat_w = t(data["body_quat_w"][:, b])
        self.body_lin_vel_w = t(data["body_lin_vel_w"][:, b])
        self.body_ang_vel_w = t(data["body_ang_vel_w"][:, b])
        self.clip_names = data["clip_names"].tolist()
        self.clip_start = torch.tensor(data["clip_start"], dtype=torch.long, device=device)
        self.clip_len = torch.tensor(data["clip_len"], dtype=torch.long, device=device)
        self.num_frames = self.joint_pos.shape[0]
        self.num_clips = len(self.clip_names)
        # frame -> clip id / clip end (exclusive)
        self.frame_clip = torch.repeat_interleave(torch.arange(self.num_clips, device=device), self.clip_len)
        self.frame_clip_end = (self.clip_start + self.clip_len)[self.frame_clip]

        # one-second sampling bins that never straddle two clips
        bins_per_clip = torch.clamp(torch.ceil(self.clip_len / self.fps).long(), min=1)
        self.bin_start = []
        self.bin_len = []
        for s, n, k in zip(self.clip_start.tolist(), self.clip_len.tolist(), bins_per_clip.tolist()):
            edges = np.linspace(0, n, k + 1).round().astype(int)
            self.bin_start += (s + edges[:-1]).tolist()
            self.bin_len += np.diff(edges).tolist()
        self.bin_start = torch.tensor(self.bin_start, dtype=torch.long, device=device)
        self.bin_len = torch.tensor(self.bin_len, dtype=torch.long, device=device)
        self.num_bins = self.bin_start.numel()
        self.frame_bin = torch.repeat_interleave(torch.arange(self.num_bins, device=device), self.bin_len)
        print(
            f"[MotionLibrary] {self.num_clips} clips, {self.num_frames} frames "
            f"({self.num_frames / self.fps / 3600:.2f} h), {self.num_bins} bins, {len(body_names)} bodies"
        )
