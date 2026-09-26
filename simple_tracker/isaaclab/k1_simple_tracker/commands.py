"""Multi-clip motion command for the simple K1 tracker.

Differences from booster_train's single-clip ``MotionCommand``:
  * samples (clip, frame) from a concatenated library with failure-weighted
    one-second bins (BeyondMimic's adaptive sampling applied across clips),
  * a clip ending is a truncation (see ``mdp.motion_end``), not a teleport,
  * reference velocities are also expressed in the robot-heading-aligned frame
    (``body_*_vel_relative_w``) so no reward depends on the absolute world yaw
    or xy position, which a drifting live mocopi stream cannot provide.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_apply_inverse,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

from .motion_library import MotionLibrary

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class MultiMotionCommand(CommandTerm):
    cfg: MultiMotionCommandCfg

    def __init__(self, cfg: MultiMotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(cfg.anchor_body_name)
        self.motion_anchor_body_index = cfg.body_names.index(cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )
        self.motion = MotionLibrary(cfg.library_file, cfg.body_names, self.robot.joint_names, self.device)

        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.clip_end = torch.full((self.num_envs,), 2, dtype=torch.long, device=self.device)
        nb = len(cfg.body_names)
        self.body_pos_relative_w = torch.zeros(self.num_envs, nb, 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, nb, 4, device=self.device)
        self.body_quat_relative_w[..., 0] = 1.0
        self.body_lin_vel_relative_w = torch.zeros(self.num_envs, nb, 3, device=self.device)
        self.body_ang_vel_relative_w = torch.zeros(self.num_envs, nb, 3, device=self.device)

        self.bin_failed = torch.zeros(self.motion.num_bins, device=self.device)
        self._current_bin_failed = torch.zeros(self.motion.num_bins, device=self.device)
        self.bin_time_prob = self.motion.bin_len.float() / self.motion.bin_len.sum()

        for key in (
            "error_anchor_height",
            "error_anchor_gravity",
            "error_body_pos",
            "error_body_rot",
            "error_body_lin_vel",
            "error_body_ang_vel",
            "error_joint_pos",
            "sampling_entropy",
            "sampling_top1_prob",
        ):
            self.metrics[key] = torch.zeros(self.num_envs, device=self.device)

    # ------------------------------------------------------------------ reference
    @property
    def command(self) -> torch.Tensor:
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion.joint_pos[self.time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.joint_vel[self.time_steps]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self.body_pos_w[:, self.motion_anchor_body_index]

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.body_quat_w[:, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.body_lin_vel_w[:, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.body_ang_vel_w[:, self.motion_anchor_body_index]

    # ---------------------------------------------------------------------- robot
    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    # -------------------------------------------------------------------- updates
    def _update_metrics(self):
        g = self.robot.data.GRAVITY_VEC_W
        self.metrics["error_anchor_height"] = (self.anchor_pos_w[:, 2] - self.robot_anchor_pos_w[:, 2]).abs()
        self.metrics["error_anchor_gravity"] = torch.norm(
            quat_apply_inverse(self.anchor_quat_w, g) - quat_apply_inverse(self.robot_anchor_quat_w, g), dim=-1
        )
        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(-1)
        self.metrics["error_body_rot"] = quat_error_magnitude(
            self.body_quat_relative_w, self.robot_body_quat_w
        ).mean(-1)
        self.metrics["error_body_lin_vel"] = torch.norm(
            self.body_lin_vel_relative_w - self.robot_body_lin_vel_w, dim=-1
        ).mean(-1)
        self.metrics["error_body_ang_vel"] = torch.norm(
            self.body_ang_vel_relative_w - self.robot_body_ang_vel_w, dim=-1
        ).mean(-1)
        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)

    def _sample_frames(self, env_ids: torch.Tensor):
        # record failures (terminations that are not time-outs / clip ends) per bin
        failed = self._env.termination_manager.terminated[env_ids]
        if torch.any(failed):
            bins = self.motion.frame_bin[self.time_steps[env_ids][failed]]
            self._current_bin_failed += torch.bincount(bins, minlength=self.motion.num_bins).float()

        fail_sum = self.bin_failed.sum()
        if fail_sum > 0:
            u = self.cfg.adaptive_uniform_ratio
            prob = (1.0 - u) * self.bin_failed / fail_sum + u * self.bin_time_prob
        else:
            prob = self.bin_time_prob
        bins = torch.multinomial(prob, len(env_ids), replacement=True)
        start = self.motion.bin_start[bins]
        frame = start + (sample_uniform(0.0, 1.0, (len(env_ids),), self.device) * self.motion.bin_len[bins]).long()
        end = self.motion.frame_clip_end[frame]
        frame = torch.minimum(frame, end - 2).clamp(min=self.motion.clip_start[self.motion.frame_clip[frame]])
        self.time_steps[env_ids] = frame
        self.clip_end[env_ids] = end

        H = -(prob * (prob + 1e-12).log()).sum() / torch.log(torch.tensor(float(self.motion.num_bins)))
        self.metrics["sampling_entropy"][:] = H
        self.metrics["sampling_top1_prob"][:] = prob.max()

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        env_ids = torch.as_tensor(env_ids, device=self.device)
        self._sample_frames(env_ids)
        if self.cfg.play:
            self.time_steps[env_ids] = self.motion.clip_start[self.motion.frame_clip[self.time_steps[env_ids]]]

        root_pos = self.body_pos_w[env_ids, 0].clone()
        root_ori = self.body_quat_w[env_ids, 0].clone()
        root_lin_vel = self.body_lin_vel_w[env_ids, 0].clone()
        root_ang_vel = self.body_ang_vel_w[env_ids, 0].clone()
        n = len(env_ids)

        r = torch.tensor([self.cfg.pose_range.get(k, (0.0, 0.0)) for k in "x y z roll pitch yaw".split()], device=self.device)
        d = sample_uniform(r[:, 0], r[:, 1], (n, 6), device=self.device)
        root_pos += d[:, :3]
        root_ori = quat_mul(quat_from_euler_xyz(d[:, 3], d[:, 4], d[:, 5]), root_ori)
        r = torch.tensor(
            [self.cfg.velocity_range.get(k, (0.0, 0.0)) for k in "x y z roll pitch yaw".split()], device=self.device
        )
        d = sample_uniform(r[:, 0], r[:, 1], (n, 6), device=self.device)
        root_lin_vel += d[:, :3]
        root_ang_vel += d[:, 3:]

        joint_pos = self.joint_pos[env_ids] + sample_uniform(
            *self.cfg.joint_position_range, (n, self.robot.num_joints), self.device
        )
        limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos = torch.clip(joint_pos, limits[..., 0], limits[..., 1])
        self.robot.write_joint_state_to_sim(joint_pos, self.joint_vel[env_ids], env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos, root_ori, root_lin_vel, root_ang_vel], dim=-1), env_ids=env_ids
        )

    def _update_command(self):
        self.time_steps += 1
        # safety net: normally the motion_end termination resets an env before this triggers
        over = torch.where(self.time_steps >= self.clip_end)[0]
        if len(over) > 0:
            self.time_steps[over] = self.clip_end[over] - 1

        nb = len(self.cfg.body_names)
        anchor_pos = self.anchor_pos_w[:, None, :].expand(-1, nb, -1)
        anchor_quat = self.anchor_quat_w[:, None, :].expand(-1, nb, -1)
        robot_anchor_pos = self.robot_anchor_pos_w[:, None, :].expand(-1, nb, -1)
        robot_anchor_quat = self.robot_anchor_quat_w[:, None, :].expand(-1, nb, -1)

        # reference re-expressed at the robot's xy position and heading, keeping reference heights
        delta_pos = robot_anchor_pos.clone()
        delta_pos[..., 2] = anchor_pos[..., 2]
        delta_ori = yaw_quat(quat_mul(robot_anchor_quat, quat_inv(anchor_quat)))
        self.body_quat_relative_w = quat_mul(delta_ori, self.body_quat_w)
        self.body_pos_relative_w = delta_pos + quat_apply(delta_ori, self.body_pos_w - anchor_pos)
        self.body_lin_vel_relative_w = quat_apply(delta_ori, self.body_lin_vel_w)
        self.body_ang_vel_relative_w = quat_apply(delta_ori, self.body_ang_vel_w)

        a = self.cfg.adaptive_alpha
        self.bin_failed = a * self._current_bin_failed + (1 - a) * self.bin_failed
        self._current_bin_failed.zero_()

    def _set_debug_vis_impl(self, debug_vis: bool):
        pass

    def _debug_vis_callback(self, event):
        pass


@configclass
class MultiMotionCommandCfg(CommandTermCfg):
    class_type: type = MultiMotionCommand

    play: bool = False
    asset_name: str = MISSING
    library_file: str = MISSING
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}
    joint_position_range: tuple[float, float] = (-0.1, 0.1)

    # share of samples drawn uniformly over time; the rest follow the failure EMA
    adaptive_uniform_ratio: float = 0.5
    adaptive_alpha: float = 0.001
