"""Observation, reward and termination terms for the simple K1 tracker.

Everything the *policy* sees about the reference is invariant to the reference's
world xy position and yaw, i.e. computable from a live mocopi stream whose global
position/heading drifts:  reference joint pos/vel, reference root gravity
direction, root linear velocity in its heading frame, root angular velocity in
its body frame and root height.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.envs.mdp import *  # noqa: F401, F403
from isaaclab.utils.math import quat_apply_inverse, quat_error_magnitude, subtract_frame_transforms, yaw_quat

from booster_train.tasks.manager_based.beyond_mimic.mdp.events import (  # noqa: F401
    randomize_joint_default_pos,
    randomize_rigid_body_com,
)

from .commands import MultiMotionCommand, MultiMotionCommandCfg  # noqa: F401

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

_GRAVITY = (0.0, 0.0, -1.0)


def _cmd(env, name: str) -> MultiMotionCommand:
    return env.command_manager.get_term(name)


def _body_ids(command: MultiMotionCommand, body_names: list[str] | None) -> list[int]:
    return [i for i, n in enumerate(command.cfg.body_names) if body_names is None or n in body_names]


# ------------------------------------------------------------------ observations
def ref_root_gravity_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    c = _cmd(env, command_name)
    g = torch.tensor(_GRAVITY, device=env.device).expand(env.num_envs, 3)
    return quat_apply_inverse(c.anchor_quat_w, g)


def ref_root_lin_vel_h(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    c = _cmd(env, command_name)
    return quat_apply_inverse(yaw_quat(c.anchor_quat_w), c.anchor_lin_vel_w)


def ref_root_ang_vel_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    c = _cmd(env, command_name)
    return quat_apply_inverse(c.anchor_quat_w, c.anchor_ang_vel_w)


def ref_root_height(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    c = _cmd(env, command_name)
    return (c.anchor_pos_w[:, 2] - env.scene.env_origins[:, 2]).unsqueeze(-1)


def robot_root_height(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    c = _cmd(env, command_name)
    return (c.robot_anchor_pos_w[:, 2] - env.scene.env_origins[:, 2]).unsqueeze(-1)


def robot_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    c = _cmd(env, command_name)
    nb = len(c.cfg.body_names)
    pos, _ = subtract_frame_transforms(
        c.robot_anchor_pos_w[:, None].expand(-1, nb, -1),
        c.robot_anchor_quat_w[:, None].expand(-1, nb, -1),
        c.robot_body_pos_w,
        c.robot_body_quat_w,
    )
    return pos.reshape(env.num_envs, -1)


def ref_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Heading-aligned reference bodies expressed in the robot root frame (critic only)."""
    c = _cmd(env, command_name)
    nb = len(c.cfg.body_names)
    pos, _ = subtract_frame_transforms(
        c.robot_anchor_pos_w[:, None].expand(-1, nb, -1),
        c.robot_anchor_quat_w[:, None].expand(-1, nb, -1),
        c.body_pos_relative_w,
        c.body_quat_relative_w,
    )
    return pos.reshape(env.num_envs, -1)


# ----------------------------------------------------------------------- rewards
def tracking_body_pos(env: ManagerBasedRLEnv, command_name: str, std: float, body_names=None) -> torch.Tensor:
    c = _cmd(env, command_name)
    i = _body_ids(c, body_names)
    err = torch.sum(torch.square(c.body_pos_relative_w[:, i] - c.robot_body_pos_w[:, i]), dim=-1).mean(-1)
    return torch.exp(-err / std**2)


def tracking_body_ori(env: ManagerBasedRLEnv, command_name: str, std: float, body_names=None) -> torch.Tensor:
    c = _cmd(env, command_name)
    i = _body_ids(c, body_names)
    err = (quat_error_magnitude(c.body_quat_relative_w[:, i], c.robot_body_quat_w[:, i]) ** 2).mean(-1)
    return torch.exp(-err / std**2)


def tracking_body_lin_vel(env: ManagerBasedRLEnv, command_name: str, std: float, body_names=None) -> torch.Tensor:
    c = _cmd(env, command_name)
    i = _body_ids(c, body_names)
    err = torch.sum(torch.square(c.body_lin_vel_relative_w[:, i] - c.robot_body_lin_vel_w[:, i]), dim=-1).mean(-1)
    return torch.exp(-err / std**2)


def tracking_body_ang_vel(env: ManagerBasedRLEnv, command_name: str, std: float, body_names=None) -> torch.Tensor:
    c = _cmd(env, command_name)
    i = _body_ids(c, body_names)
    err = torch.sum(torch.square(c.body_ang_vel_relative_w[:, i] - c.robot_body_ang_vel_w[:, i]), dim=-1).mean(-1)
    return torch.exp(-err / std**2)


# ------------------------------------------------------------------ terminations
def motion_end(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """The current clip has no next frame: truncate (use with ``time_out=True``)."""
    c = _cmd(env, command_name)
    return c.time_steps >= c.clip_end - 1


def bad_anchor_height(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    c = _cmd(env, command_name)
    return (c.anchor_pos_w[:, 2] - c.robot_anchor_pos_w[:, 2]).abs() > threshold


def bad_anchor_gravity(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    c = _cmd(env, command_name)
    g = torch.tensor(_GRAVITY, device=env.device).expand(env.num_envs, 3)
    ref = quat_apply_inverse(c.anchor_quat_w, g)[:, 2]
    rob = quat_apply_inverse(c.robot_anchor_quat_w, g)[:, 2]
    return (ref - rob).abs() > threshold


def bad_body_height(env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names=None) -> torch.Tensor:
    c = _cmd(env, command_name)
    i = _body_ids(c, body_names)
    err = (c.body_pos_relative_w[:, i, 2] - c.robot_body_pos_w[:, i, 2]).abs()
    return torch.any(err > threshold, dim=-1)
