"""Bounded arm corrections from measured K1 joints and geometric self-distance.

This is a tracking aid, not a proof of collision avoidance. Only arm joints may
change; the floating base, IMU balance targets and leg commands are untouched.
No simulator position, velocity or contact truth is an input.
"""

import copy

import mujoco
import numpy as np


class ArmCollisionFeedback:
    def __init__(self, robot, clearance=0.025):
        if not np.isfinite(clearance) or not 0 < clearance <= 0.03:
            raise ValueError("Arm collision clearance must be in (0, 0.03] metres")
        self.model = copy.copy(robot.model)
        self.model.geom_margin[:] = 0.03
        self.data = mujoco.MjData(self.model)
        self.data.qpos[:] = robot.neutral_qpos
        self.data.qpos[:3] = [0, 0, 1.5]
        self.data.qpos[3:7] = [1, 0, 0, 0]
        self.clearance = clearance
        self.horizon = 0.04
        self.j1 = np.zeros((3, self.model.nv))
        self.j2 = np.zeros_like(self.j1)
        self.weight = np.r_[np.zeros(2), np.ones(8), np.zeros(12)]
        self.forearms = {self.model.body(f"{side}_elbow_yaw_link").id for side in ("left", "right")}

    def project(self, joint_position, joint_velocity, target):
        q, dq, target = (np.asarray(x) for x in (joint_position, joint_velocity, target))
        if any(x.shape != (22,) or not np.isfinite(x).all() for x in (q, dq, target)):
            raise ValueError("Arm feedback requires finite canonical K1 joint vectors")
        self.data.qpos[7:] = q
        mujoco.mj_forward(self.model, self.data)
        constraints = []
        for contact in self.data.contact[: self.data.ncon]:
            b1, b2 = self.model.geom_bodyid[[contact.geom1, contact.geom2]]
            if not b1 or not b2 or contact.dist >= 0.03:
                continue
            if b1 not in self.forearms and b2 not in self.forearms:
                continue
            mujoco.mj_jac(self.model, self.data, self.j1, None, contact.pos, int(b1))
            mujoco.mj_jac(self.model, self.data, self.j2, None, contact.pos, int(b2))
            gradient = contact.frame[:3] @ (self.j2 - self.j1)[:, 6:]
            # Nearly immovable joint housings must not demand unbounded corrections.
            if np.linalg.norm(gradient * self.weight) < 0.03:
                continue
            closing = min(0.0, float(gradient @ dq))
            constraints.append((gradient.copy(), self.clearance - contact.dist - self.horizon * closing))
        result = target.copy()
        for _ in range(3):
            for gradient, lower in constraints:
                missing = lower - gradient @ (result - q)
                if missing > 0:
                    direction = gradient * self.weight
                    result += direction * missing / (gradient @ direction + 1e-8)
                    result = np.clip(result, target - 0.15, target + 0.15)
        return result

    def project_batch(self, joint_position, joint_velocity, target):
        import torch

        arrays = [x.detach().cpu().numpy() for x in (joint_position, joint_velocity, target)]
        output = np.stack([self.project(q, dq, t) for q, dq, t in zip(*arrays)])
        return torch.as_tensor(output, dtype=target.dtype, device=target.device)
