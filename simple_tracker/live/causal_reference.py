"""Causal reference estimator: K1 qpos stream (GMR/MJCF convention) -> tracker reference.

Numpy only, shared by the live path (mocopi UDP -> GMR -> this, inside booster_deploy)
and the offline check (mocopi BVH -> same per-frame GMR -> this -> clip npz), so the
sim2sim test sees exactly what the robot would see live: backward differences with a
light low-pass instead of the central differences used for the training library.
"""

from __future__ import annotations

import numpy as np

# GMR K1_serial.xml / booster_assets K1_JOINT_NAMES order (qpos[7:])
MJCF_JOINTS = [
    "AAHead_yaw", "Head_pitch", "ALeft_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
    "ARight_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw", "Left_Hip_Pitch",
    "Left_Hip_Roll", "Left_Hip_Yaw", "Left_Knee_Pitch", "Left_Ankle_Pitch", "Left_Ankle_Roll", "Right_Hip_Pitch",
    "Right_Hip_Roll", "Right_Hip_Yaw", "Right_Knee_Pitch", "Right_Ankle_Pitch", "Right_Ankle_Roll",
]  # fmt: skip
# Isaac Lab articulation order used by the policy (= library joint_names = booster_deploy sim_joint_names)
SIM_JOINTS = [
    "AAHead_yaw", "ALeft_Shoulder_Pitch", "ARight_Shoulder_Pitch", "Left_Hip_Pitch", "Right_Hip_Pitch", "Head_pitch",
    "Left_Shoulder_Roll", "Right_Shoulder_Roll", "Left_Hip_Roll", "Right_Hip_Roll", "Left_Elbow_Pitch",
    "Right_Elbow_Pitch", "Left_Hip_Yaw", "Right_Hip_Yaw", "Left_Elbow_Yaw", "Right_Elbow_Yaw", "Left_Knee_Pitch",
    "Right_Knee_Pitch", "Left_Ankle_Pitch", "Right_Ankle_Pitch", "Left_Ankle_Roll", "Right_Ankle_Roll",
]  # fmt: skip
MJCF_TO_SIM = np.array([MJCF_JOINTS.index(n) for n in SIM_JOINTS])

# The training URDF hangs the legs 1.5 cm lower than GMR's MJCF and the library grounds
# foot_link at 0.027 (MJCF grounding uses 0.025), so trunk height gains 1.7 cm.
URDF_TRUNK_Z_OFFSET = 0.017


def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def _quat_to_rotvec(q):
    q = q / np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    s = np.linalg.norm(q[1:])
    if s < 1e-9:
        return 2.0 * q[1:]
    return 2.0 * np.arctan2(s, q[0]) * q[1:] / s


class CausalReference:
    def __init__(self, fps: float = 50.0, smoothing: float = 0.5, z_offset: float = URDF_TRUNK_Z_OFFSET):
        """smoothing: EMA weight of the previous estimate (0 = raw differences)."""
        self.dt = 1.0 / fps
        self.a = smoothing
        self.z_offset = z_offset
        self.reset()

    def reset(self):
        self.prev = None
        self.joint_vel = np.zeros(22)
        self.lin_vel = np.zeros(3)
        self.ang_vel = np.zeros(3)

    def update(self, qpos: np.ndarray) -> dict:
        """qpos: [root_pos(3), root_quat wxyz(4), 22 dof in MJCF order] (GMR output, grounded)."""
        pos = qpos[:3].astype(np.float64).copy()
        pos[2] += self.z_offset
        quat = qpos[3:7] / np.linalg.norm(qpos[3:7])
        dof = qpos[7:][MJCF_TO_SIM].astype(np.float64)
        if self.prev is not None:
            p_pos, p_quat, p_dof = self.prev
            if np.dot(quat, p_quat) < 0:
                quat = -quat
            jv = (dof - p_dof) / self.dt
            lv = (pos - p_pos) / self.dt
            dq = _quat_mul(quat, p_quat * np.array([1, -1, -1, -1]))  # world-frame delta rotation
            av = _quat_to_rotvec(dq) / self.dt
            self.joint_vel = self.a * self.joint_vel + (1 - self.a) * jv
            self.lin_vel = self.a * self.lin_vel + (1 - self.a) * lv
            self.ang_vel = self.a * self.ang_vel + (1 - self.a) * av
        self.prev = (pos, quat, dof)
        return {
            "joint_pos": dof,
            "joint_vel": self.joint_vel.copy(),
            "root_pos": pos,
            "root_quat": quat,
            "root_lin_vel_w": self.lin_vel.copy(),
            "root_ang_vel_w": self.ang_vel.copy(),
        }
