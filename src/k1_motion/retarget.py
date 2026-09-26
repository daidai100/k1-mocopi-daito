"""Causal, bounded damped-least-squares IK using the actual K1 kinematic model."""

import time

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import lsq_linear

from .calibration import Calibration
from .contracts import LANDMARKS, Reference
from .math3d import rotation
from .retarget_speed import retarget_speed_contract, retarget_speed_metadata

RETARGET_VERSION = "causal-ik-v7-contact-root-projection"


class Retargeter:
    def __init__(self, robot, iterations=12, *, speed_profile=None, motion_profile=None):
        self.robot = robot
        self.speed_contract = retarget_speed_contract(robot, speed_profile)
        self.retarget_version = retarget_speed_metadata(robot, RETARGET_VERSION, speed_profile)["retarget_version"]
        if motion_profile is not None:
            from .calibration import motion_scale_contract
            motion_scale_contract(motion_profile)
            self.retarget_version += f"+{motion_profile}"
        self.motion_profile = motion_profile
        self.retarget_velocity_limits = np.asarray(self.speed_contract["joint_velocity_limits_rad_s"])
        self.data = mujoco.MjData(robot.model)
        self.iterations = iterations
        self.calibration = None
        self.previous = None
        self.anchors = [None, None]
        self.contact_state = np.zeros(2, dtype=bool)
        self.last_report = {}
        self.root_correction = np.zeros(3)
        self.ids = np.array([2, 4, 5, 7, 8, 10, 11, 12, 14, 15, 16])
        self.weights = np.repeat([0.4, 0.5, 1.0, 0.5, 1.0, 0.5, 2.0, 1.0, 0.5, 2.0, 1.0], 3)

    def calibrate(self, frame, floor=None, robot_origin=None):
        self.calibration = Calibration.from_neutral(
            frame, self.robot, floor, robot_origin, motion_profile=self.motion_profile)
        self.previous = None
        self.previous_human = None
        self.anchors = [None, None]
        self.contact_state[:] = False
        self.root_correction[:] = 0
        self.data.qpos[:] = self.robot.neutral_qpos
        mujoco.mj_forward(self.robot.model, self.data)
        calibrated = self.calibration.apply(frame)
        self.head_neutral_relative = rotation(calibrated.orientations[0]).inv() * rotation(
            calibrated.orientations[2]
        )
        self.robot_head_neutral = Rotation.from_matrix(
            self.data.site_xmat[self.robot.site_ids[2]].reshape(3, 3)
        )

    def process(self, frame):
        start = time.perf_counter()
        if self.calibration is None:
            raise ValueError("Calibrate before retargeting")
        human = self.calibration.apply(frame)
        previous = self.previous
        dt = frame.source_time - previous.source_time if previous is not None else self.robot.control_dt
        if not 0 < dt <= 0.5:
            raise ValueError("Nonmonotonic input or long gap; reset retarget history before resuming")
        p = human.positions.copy()
        targets = p.copy()
        root_rotation = rotation(human.orientations[0])
        head_relative = root_rotation.inv() * rotation(human.orientations[2])
        head_target = (
            root_rotation * head_relative * self.head_neutral_relative.inv() * self.robot_head_neutral
        )
        root_p = p[0].copy()
        # Match directions with K1 segment lengths; human/robot proportions differ.
        neutral = self.robot.neutral_landmarks
        for side in ("left", "right"):
            chain = [LANDMARKS.index(f"{side}_{x}") for x in ("hip", "knee", "ankle", "toe")]
            chain_arm = [LANDMARKS.index(f"{side}_{x}") for x in ("shoulder", "elbow", "wrist")]
            for indices in (chain, chain_arm):
                first = indices[0]
                targets[first] = root_p + root_rotation.apply(neutral[first] - neutral[0])
                for a, b in zip(indices, indices[1:]):
                    direction = p[b] - p[a]
                    length = np.linalg.norm(direction)
                    if length < 1e-6:
                        raise ValueError("Zero-length human limb")
                    targets[b] = targets[a] + direction / length * np.linalg.norm(neutral[b] - neutral[a])
        targets[2] = root_p + (p[2] - p[0]) * (
            np.linalg.norm(neutral[2] - neutral[0]) / np.linalg.norm(p[2] - p[0])
        )
        # Contact labels use current height and backward velocity; flight is allowed.
        confidence = np.zeros(2)
        foot_targets = {}
        for side, ankle in enumerate((11, 15)):
            speed = (
                0.0
                if self.previous_human is None
                else np.linalg.norm(human.positions[ankle] - self.previous_human.positions[ankle]) / dt
            )
            height = min(p[ankle, 2] - 0.04, p[ankle + 1, 2] - 0.015)
            confidence[side] = np.clip(1 - max(height, 0) / 0.07, 0, 1) * np.clip(1 - speed / 0.5, 0, 1)
            threshold = 0.25 if self.contact_state[side] else 0.55
            self.contact_state[side] = confidence[side] >= threshold
            if self.contact_state[side]:
                if self.anchors[side] is None:
                    self.anchors[side] = targets[ankle].copy()
                    if previous is not None:
                        self.anchors[side][:2] = previous.landmarks[ankle, :2]
                    self.anchors[side][2] = 0.038
                offset = self.anchors[side] - targets[ankle]
                targets[ankle : ankle + 2] += offset
                # Human toe markers are not K1 sole coordinates. Scaling their
                # downward ankle-to-toe ray put flat human feet onto robot toes.
                # A detected stance uses the robot sole geometry and a level
                # contact frame; swing/flight retain the measured limb targets.
                forward = p[ankle + 1, :2] - p[ankle, :2]
                if np.linalg.norm(forward) > 1e-6:
                    yaw = np.arctan2(forward[1], forward[0])
                    foot_targets[ankle] = Rotation.from_euler("z", yaw)
                    targets[ankle + 1] = targets[ankle] + foot_targets[ankle].apply([0.095, 0, -0.02])
            else:
                self.anchors[side] = None
        # Root translation participates in IK: copying the scaled human pelvis
        # exactly can make a planted foot unreachable with K1's leg proportions.
        # Non-contact targets follow this bounded correction; stance anchors stay
        # fixed in the world. All inputs, anchors and bounds remain causal.
        root_lo = root_p + np.maximum(-0.15, self.root_correction - 4.0 * dt)
        root_hi = root_p + np.minimum(0.15, self.root_correction + 4.0 * dt)
        self.data.qpos[:3] = root_p + self.root_correction
        self.data.qpos[3:7] = human.orientations[0]
        seed = self.data.qpos[7:].copy()
        lo, hi = self.robot.limits[:, 0] + 1e-5, self.robot.limits[:, 1] - 1e-5
        if previous is not None:
            step = self.retarget_velocity_limits * dt
            lo, hi = np.maximum(lo, seed - step), np.minimum(hi, seed + step)
        jac = np.zeros((len(self.ids) * 3, self.robot.model.nv))
        jp, jr = np.zeros((3, self.robot.model.nv)), np.zeros((3, self.robot.model.nv))
        columns = np.r_[0:3, 6 : self.robot.model.nv]
        anchored = np.zeros(17, dtype=bool)
        weights = self.weights.copy().reshape(-1, 3)
        for side, ankle in enumerate((11, 15)):
            if self.contact_state[side]:
                anchored[ankle : ankle + 2] = True
                weights[self.ids == ankle] = 12.0
                weights[self.ids == ankle + 1] = 4.0
        weights = weights.ravel()
        for _ in range(self.iterations):
            mujoco.mj_forward(self.robot.model, self.data)
            positions = self.data.site_xpos[self.robot.site_ids]
            projected_targets = targets + (~anchored)[:, None] * (self.data.qpos[:3] - root_p)
            error = (projected_targets[self.ids] - positions[self.ids]).reshape(-1) * weights
            for i, index in enumerate(self.ids):
                mujoco.mj_jacSite(self.robot.model, self.data, jp, jr, self.robot.site_ids[index])
                if not anchored[index]:
                    jp[:, :3] = 0.0  # Target and site translate together.
                jac[i * 3 : (i + 1) * 3] = jp
            j = jac[:, columns] * weights[:, None]
            root_jac = np.zeros((3, 25))
            root_jac[:, :3] = np.eye(3) * 0.5
            j = np.vstack([j, root_jac])
            error = np.r_[error, 0.5 * (root_p - self.data.qpos[:3])]
            j = np.vstack([j, 2.0 * root_jac])
            error = np.r_[error, root_p + self.root_correction - self.data.qpos[:3]]
            # Weak neutral and continuity regularization resolves unobserved twists.
            head_current = Rotation.from_matrix(self.data.site_xmat[self.robot.site_ids[2]].reshape(3, 3))
            head_error = (head_target * head_current.inv()).as_rotvec()
            mujoco.mj_jacSite(self.robot.model, self.data, jp, jr, self.robot.site_ids[2])
            j = np.vstack([j, 0.15 * jr[:, columns]])
            error = np.r_[error, 0.15 * head_error]
            for ankle, foot_target in foot_targets.items():
                foot_current = Rotation.from_matrix(
                    self.data.site_xmat[self.robot.site_ids[ankle]].reshape(3, 3)
                )
                foot_error = (foot_target * foot_current.inv()).as_rotvec()
                mujoco.mj_jacSite(self.robot.model, self.data, jp, jr, self.robot.site_ids[ankle])
                j = np.vstack([j, 0.6 * jr[:, columns]])
                error = np.r_[error, 0.6 * foot_error]
            # Resolve robot self-intersections in the IK solve, instead of merely
            # detecting them after an unconstrained landmark fit. Contact normals
            # and point Jacobians give a local signed-distance gradient. The final
            # collision check remains mandatory if this bounded solve cannot fit.
            for contact in self.data.contact[: self.data.ncon]:
                body1, body2 = self.robot.model.geom_bodyid[[contact.geom1, contact.geom2]]
                if contact.dist >= 0.003:
                    continue
                first_jac = np.zeros((3, self.robot.model.nv))
                second_jac = np.zeros_like(first_jac)
                mujoco.mj_jac(self.robot.model, self.data, first_jac, None, contact.pos, int(body1))
                mujoco.mj_jac(self.robot.model, self.data, second_jac, None, contact.pos, int(body2))
                distance_jac = contact.frame[:3] @ (second_jac - first_jac)
                j = np.vstack([j, 4.0 * distance_jac[columns]])
                error = np.r_[error, 4.0 * min(0.03, 0.003 - contact.dist)]
            j = np.vstack([j, np.sqrt(0.002) * np.eye(25)])
            error = np.r_[
                error, np.zeros(3), np.sqrt(0.002) * ((seed + self.robot.neutral) * 0.5 - self.data.qpos[7:])
            ]
            lower = np.r_[root_lo - self.data.qpos[:3], np.maximum(lo - self.data.qpos[7:], -0.2)]
            upper = np.r_[root_hi - self.data.qpos[:3], np.minimum(hi - self.data.qpos[7:], 0.2)]
            # Solve with active joint/velocity bounds. Clipping an unconstrained
            # update afterwards lets the free pelvis oscillate against joints
            # that cannot follow it, creating exactly the foot slip we avoid.
            delta = lsq_linear(j, error, bounds=(lower, upper), method="bvls", tol=1e-6).x
            self.data.qpos[:3] = np.clip(self.data.qpos[:3] + delta[:3], root_lo, root_hi)
            self.data.qpos[7:] = np.clip(self.data.qpos[7:] + np.clip(delta[3:], -0.2, 0.2), lo, hi)
            if np.linalg.norm(delta) < 1e-4:
                break
        mujoco.mj_forward(self.robot.model, self.data)
        self.root_correction = self.data.qpos[:3].copy() - root_p
        targets += (~anchored)[:, None] * self.root_correction
        root_p = self.data.qpos[:3].copy()
        q = self.data.qpos[7:].copy()
        dq = np.zeros(22) if previous is None else (q - previous.joint_position) / dt
        root_velocity = np.zeros(6)
        if previous is not None:
            root_velocity[:3] = (root_p - previous.root_position) / dt
            root_velocity[3:] = (root_rotation * rotation(previous.root_orientation).inv()).as_rotvec() / dt
        landmarks = self.robot.landmarks(self.data)
        collisions = sum(
            1
            for c in self.data.contact[: self.data.ncon]
            if c.dist < -0.005 and 0 not in self.robot.model.geom_bodyid[[c.geom1, c.geom2]]
        )
        rms = float(np.sqrt(np.mean((targets[self.ids] - landmarks[self.ids]) ** 2)))
        reasons = []
        if collisions:
            reasons.append("self_collision")
        if rms > 0.12:
            reasons.append("landmark_error")
        if root_p[2] < 0.22 or np.min(landmarks[[12, 16], 2]) < -0.08:
            reasons.append("ground_penetration_or_root_height")
        reference = Reference(
            frame.source_time,
            frame.received_time,
            root_p,
            human.orientations[0],
            root_velocity,
            q,
            dq,
            landmarks,
            self.contact_state.astype(float),
            confidence,
            valid=not reasons,
            session=frame.session,
        )
        self.previous, self.previous_human = reference, human
        self.last_report = {
            "retarget_version": self.retarget_version,
            "root_projection_m": float(np.linalg.norm(self.root_correction)),
            "contact_anchor_error_m": float(
                np.max(np.linalg.norm((targets - landmarks)[anchored], axis=-1), initial=0.0)
            ),
            "rms_landmark_error_m": rms,
            "self_collisions": collisions,
            "rejection_reasons": reasons,
            "seconds": time.perf_counter() - start,
            "physics_qualified": False,
        }
        return reference
