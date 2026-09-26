"""booster_deploy policy for the simple multi-clip K1 tracker (sim2sim and real robot).

The observation must match ``k1_simple_tracker.env_cfg.ObservationsCfg.PolicyCfg``:

    ref_joint_pos(22) ref_joint_vel(22)            sim (Isaac) joint order
    ref_root_gravity_b(3)                           reference trunk frame
    ref_root_lin_vel_h(3)                           reference heading frame
    ref_root_ang_vel_b(3)                           reference trunk frame
    ref_root_height(1)
    base_ang_vel_b(3) projected_gravity_b(3)        robot IMU
    joint_pos - default(22) joint_vel(22)           sim joint order
    last_action(22)

No term depends on the reference's world xy/yaw, so the reference may come from a
recorded clip (``ClipReference``) or from a live, drifting mocopi stream (any object
with the same ``step()`` interface).
"""

from __future__ import annotations

import re
from dataclasses import MISSING

import numpy as np
import torch

from booster_deploy.controllers.base_controller import BaseController, Policy
from booster_deploy.controllers.controller_cfg import ControllerCfg, MujocoControllerCfg, PolicyCfg
from booster_deploy.robots.k1 import K1_CFG
from booster_deploy.utils.isaaclab import math as lab_math
from booster_deploy.utils.isaaclab.configclass import configclass
from booster_deploy.utils.policy_runner import create_policy_runner

GRAVITY = torch.tensor([0.0, 0.0, -1.0])


class ClipReference:
    """Plays library clips back to back: hold first frame, play, hold last frame."""

    def __init__(self, library: str, clip_regex: str, hold_s: float, device: str, max_clips: int = 0):
        z = np.load(library)
        names = z["clip_names"].tolist()
        idx = [i for i, n in enumerate(names) if re.search(clip_regex, n)]
        if max_clips:
            idx = idx[:max_clips]
        if not idx:
            raise ValueError(f"no clip matches {clip_regex!r}")
        self.fps = float(z["fps"])
        hold = int(round(hold_s * self.fps))
        jp, jv, bp, bq, lv, av, self.segments = [], [], [], [], [], [], []
        t = 0
        for i in idx:
            s, n = int(z["clip_start"][i]), int(z["clip_len"][i])
            sl = np.r_[np.full(hold, s), np.arange(s, s + n), np.full(hold, s + n - 1)]
            moving = np.r_[np.zeros(hold), np.ones(n), np.zeros(hold)][:, None]
            jp.append(z["joint_pos"][sl])
            jv.append(z["joint_vel"][sl] * moving)
            bp.append(z["body_pos_w"][sl, 0])
            bq.append(z["body_quat_w"][sl, 0])
            lv.append(z["body_lin_vel_w"][sl, 0] * moving)
            av.append(z["body_ang_vel_w"][sl, 0] * moving)
            self.segments.append((names[i], t, t + len(sl)))
            t += len(sl)
        t_ = lambda x: torch.tensor(np.concatenate(x), dtype=torch.float32, device=device)  # noqa: E731
        self.joint_pos, self.joint_vel = t_(jp), t_(jv)
        self.root_pos, self.root_quat = t_(bp), t_(bq)
        self.root_lin_vel_w, self.root_ang_vel_w = t_(lv), t_(av)
        self.num_frames = self.joint_pos.shape[0]
        self.frame = 0

    def reset(self):
        self.frame = 0

    @property
    def done(self) -> bool:
        return self.frame >= self.num_frames - 1

    def step(self):
        """Returns the reference for this control step and advances one frame."""
        f = min(self.frame, self.num_frames - 1)
        self.frame += 1
        return (
            self.joint_pos[f],
            self.joint_vel[f],
            self.root_pos[f],
            self.root_quat[f],
            self.root_lin_vel_w[f],
            self.root_ang_vel_w[f],
        )

    def clip_at(self, frame: int) -> str:
        return next((n for n, a, b in self.segments if a <= frame < b), self.segments[-1][0])


class SimpleTrackerPolicy(Policy):
    def __init__(self, cfg: SimpleTrackerPolicyCfg, controller: BaseController):
        super().__init__(cfg, controller)
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self._model = create_policy_runner(cfg.checkpoint_path, self.device)
        self.robot = controller.robot
        self.robot.data.to(cfg.device)
        self.action_scale = (0.25 * self.robot.effort_limit / self.robot.joint_stiffness).to(self.device)
        self.default_joint_pos = self.robot.default_joint_pos.to(self.device)
        self.gravity = GRAVITY.to(self.device)
        self.reference = ClipReference(
            cfg.library_path, cfg.clip_regex, cfg.hold_s, cfg.device, max_clips=cfg.max_clips
        )

    def reset(self) -> None:
        self.reference.reset()
        self.last_action = torch.zeros(self.robot.num_joints, device=self.device)

    def compute_observation(self) -> torch.Tensor:
        jp, jv, self.ref_root_pos, self.ref_root_quat, lv, av = self.reference.step()
        self.ref_joint_pos = jp
        q = self.ref_root_quat
        r2s = self.robot.data.real2sim_joint_indexes
        obs = torch.cat(
            (
                jp,
                jv,
                lab_math.quat_apply_inverse(q, self.gravity),
                lab_math.quat_apply_inverse(lab_math.yaw_quat(q), lv),
                lab_math.quat_apply_inverse(q, av),
                self.ref_root_pos[2:3],
                self.robot.data.root_ang_vel_b,
                lab_math.quat_apply_inverse(self.robot.data.root_quat_w, self.gravity),
                self.robot.data.joint_pos[r2s] - self.default_joint_pos[r2s],
                self.robot.data.joint_vel[r2s],
                self.last_action,
            ),
            dim=-1,
        )
        return obs.reshape(1, -1)

    def inference(self) -> torch.Tensor:
        with torch.no_grad():
            obs = self.compute_observation()
            action = self._model(obs).flatten()
        self.last_action = action

        if hasattr(self.controller, "set_reference_qpos"):
            s2r = self.robot.data.sim2real_joint_indexes
            self.controller.set_reference_qpos(torch.cat([self.ref_root_pos, self.ref_root_quat, self.ref_joint_pos[s2r]]))

        if self.cfg.enable_safety_fallback:
            up_robot = lab_math.quat_apply_inverse(self.robot.data.root_quat_w, self.gravity)
            up_ref = lab_math.quat_apply_inverse(self.ref_root_quat, self.gravity)
            if torch.dot(up_robot, up_ref) < 0.5:
                print("\n[simple_tracker] large trunk orientation error, stopping for safety")
                self.controller.stop()
        if self.cfg.stop_at_end and self.reference.done:
            self.controller.stop()

        s2r = self.robot.data.sim2real_joint_indexes
        return action[s2r] * self.action_scale + self.default_joint_pos


@configclass
class SimpleTrackerPolicyCfg(PolicyCfg):
    constructor = SimpleTrackerPolicy
    checkpoint_path: str = MISSING  # exported TorchScript (.pt) or ONNX
    library_path: str = MISSING
    clip_regex: str = r"dataset-2_walk_normal_001$"
    max_clips: int = 0
    hold_s: float = 2.0
    stop_at_end: bool = True


# Gains/effort limits must equal BOOSTER_K1_CFG used in training (same values as the
# workspace's k1_mocopi_motion*_own tasks).  Order: head x2, left arm x4, right arm x4,
# left leg x6, right leg x6.
K1_TRAIN_STIFFNESS = [3.9478417602100686] * 10 + [
    30.200989465607023, 21.447961045805584, 17.846013389258083, 60.401978931214046, 35.692026778516166,
    35.692026778516166,
] * 2  # fmt: skip
K1_TRAIN_DAMPING = [0.25132741228] * 10 + [
    3.60497756989125, 2.560161764834957, 2.1302109340993156, 4.806636759855, 4.260421868198631, 4.260421868198631,
] * 2  # fmt: skip
K1_TRAIN_EFFORT = [6.0, 6.0] + [14.0] * 8 + [68.0, 76.0, 38.3, 112.0, 38.3, 38.3] * 2


@configclass
class K1SimpleTrackerControllerCfg(ControllerCfg):
    robot = K1_CFG.replace(  # type: ignore
        joint_stiffness=K1_TRAIN_STIFFNESS,
        joint_damping=K1_TRAIN_DAMPING,
        effort_limit=K1_TRAIN_EFFORT,
    )
    enable_velocity_commands = False
    policy: SimpleTrackerPolicyCfg = SimpleTrackerPolicyCfg()
    mujoco = MujocoControllerCfg(init_pos=[0.0, 0.0, 0.57], visualize_reference_ghost=True)
