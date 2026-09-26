"""Causal, morphology-aware low-pose retargeting on the unchanged K1 model.

This is a new reference task, not a relaxation of the standing/walking audit.
Feet and knees may support the robot. Furniture, hand-supported crawling and
ground contact by the trunk/arms remain outside this task.
"""
import time

import mujoco
import numpy as np
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation

from .calibration import Calibration
from .contracts import HumanFrame, MotionClip, Reference
from .math3d import heading, rotation
from .recovery_geometry import geometry_forward, has_self_penetration
from .retarget_recovery import RecoveryRetargeter
from .streaming import SAMPLING_VERSION, control_schedule
from .retarget_speed import retarget_speed_metadata

LOW_POSE_VERSION = "causal-knee-support-ik-v1"
LOW_POSE_SOURCE = "bones_seed_v2-absolute-translation-zup-floor0"
LOW_POSE_SETTINGS = {
    "source_floor_m": 0.0,
    "root_correction_limit_m": .18,
    "root_correction_speed_m_s": 1.0,
    "knee_surface_height_m": .047,
    "minimum_root_height_m": .12,
    "max_tick_landmark_error_m": .12,
    "source_knee_angle_weight": .25,
    "iterations": 16,
    "initial_iterations": 48,
}
LIMB_CHAINS = ((9, 10, 11, 12), (13, 14, 15, 16), (3, 4, 5), (6, 7, 8))
LOW_POSE_CLEARANCE_VERSION = "causal-knee-and-sole-clearance-v1"
LOW_POSE_CLEARANCE_SETTINGS = {"max_lift_m": .04, "max_rise_m_s": .5, "release_m_s": .1,
                               "clearance_m": .001}


def correct_low_pose_ground(clip):
    from .ground_reference import _correct_ground
    supports = tuple(f"{side}_{part}" for side in ("left", "right")
                     for part in ("ankle_roll_link", "knee_pitch_link"))
    corrected, receipt = _correct_ground(clip, dict(LOW_POSE_CLEARANCE_SETTINGS), supports,
                                         LOW_POSE_CLEARANCE_VERSION)
    receipt["unsupported_ground_contact_samples"] = receipt.pop("nonfoot_ground_contact_samples")
    receipt["allowed_support_bodies"] = list(supports)
    return corrected, receipt


def leg_lengths(positions):
    return np.array([np.linalg.norm(positions[a] - positions[b])
                     + np.linalg.norm(positions[b] - positions[c])
                     for a, b, c in ((9, 10, 11), (13, 14, 15))])


def skeletal_calibration(frame, robot, floor=0.0):
    """Current-frame rigid bone lengths supply scale; pelvis height does not."""
    lengths = leg_lengths(frame.positions)
    if (not np.isfinite(floor) or not np.isfinite(lengths).all()
            or np.any(lengths < .45) or np.any(lengths > 1.4)
            or abs(lengths[0] - lengths[1]) > .1):
        raise ValueError("Invalid metric human skeleton/floor for low-pose calibration")
    scale = float(leg_lengths(robot.neutral_landmarks).mean() / lengths.mean())
    origin = frame.positions[0].copy()
    origin[2] = floor
    return Calibration(origin, heading(frame.orientations[0]), scale, np.zeros(3), float(floor), frame.session)


def source_pose_features(frame, scale):
    p = frame.positions
    flexions = []
    for a, b, c in ((9, 10, 11), (13, 14, 15)):
        upper, lower = p[a] - p[b], p[c] - p[b]
        cosine = float(upper @ lower / (np.linalg.norm(upper) * np.linalg.norm(lower)))
        flexions.append(np.pi - np.arccos(np.clip(cosine, -1, 1)))
    knees = p[[10, 14], 2] * scale
    return {"knee_height_m": knees, "knee_flexion_rad": np.array(flexions),
            "kneeling": (knees < .055) & (np.array(flexions) > 1.0),
            "root_height_m": float(p[0, 2] * scale),
            "hand_support": bool(min(p[5, 2], p[8, 2]) * scale < .035)}


def directional_targets(robot, human):
    """Human segment directions with the robot's segment lengths, no fitting."""
    p = human.positions
    targets = p.copy()
    root_rotation = rotation(human.orientations[0])
    neutral = robot.neutral_landmarks
    for chain in LIMB_CHAINS:
        first = chain[0]
        targets[first] = p[0] + root_rotation.apply(neutral[first] - neutral[0])
        for a, b in zip(chain, chain[1:]):
            direction = p[b] - p[a]
            if np.linalg.norm(direction) < 1e-6:
                raise ValueError("Zero-length human limb")
            targets[b] = targets[a] + direction / np.linalg.norm(direction) * np.linalg.norm(neutral[b] - neutral[a])
    for index in (1, 2):
        delta = p[index] - p[0]
        targets[index] = p[0] + delta / np.linalg.norm(delta) * np.linalg.norm(neutral[index] - neutral[0])
    return targets


class LowPoseRetargeter(RecoveryRetargeter):
    def __init__(self, robot, settings=None, *, speed_profile=None):
        self.settings = {**LOW_POSE_SETTINGS, **(settings or {})}
        if (set(self.settings) != set(LOW_POSE_SETTINGS)
                or not all(np.isfinite(v) for v in self.settings.values())
                or any(v <= 0 for k, v in self.settings.items() if k != "source_floor_m")
                or any(not isinstance(self.settings[k], int) for k in ("iterations", "initial_iterations"))):
            raise ValueError("Invalid low-pose settings")
        super().__init__(robot, iterations=self.settings["iterations"], speed_profile=speed_profile)
        self.knee_state = np.zeros(2, dtype=bool)
        self.knee_anchors = [None, None]

    def _safe_step(self, old_pose):
        # Margins are useful for proximity candidates, not collision truth.
        # MuJoCo's box-box contact manifold with inflated margins can report a
        # negative distance for disjoint shin/foot boxes. Validate the command
        # path using the unchanged zero-margin collision model.
        margins = self.robot.model.geom_margin.copy()
        try:
            self.robot.model.geom_margin[:] = 0
            super()._safe_step(old_pose)
        finally:
            self.robot.model.geom_margin[:] = margins

    def calibrate(self, frame, floor=0.0, robot_origin=None):
        if robot_origin is not None:
            raise ValueError("Low-pose source profile uses an explicit world floor and xy canonicalization")
        self.calibration = skeletal_calibration(frame, self.robot, floor)
        self.previous = self.previous_human = self.previous_control_time = None
        self.anchors, self.knee_anchors = [None, None], [None, None]
        self.contact_state[:] = self.knee_state[:] = False
        self.root_correction[:] = 0
        self.data.qpos[:] = self.robot.neutral_qpos
        geometry_forward(self.robot.model, self.data)
        human = self.calibration.apply(frame)
        self.head_neutral_relative = rotation(human.orientations[0]).inv() * rotation(human.orientations[2])
        self.robot_head_neutral = Rotation.from_matrix(self.data.site_xmat[self.robot.site_ids[2]].reshape(3, 3))

    def process(self, frame, *, control_time):
        started = time.perf_counter()
        if self.calibration is None:
            raise ValueError("Calibrate before low-pose retargeting")
        previous = self.previous
        source_dt = self.robot.control_dt if previous is None else frame.source_time - previous.source_time
        dt = self.robot.control_dt if self.previous_control_time is None else round(control_time - self.previous_control_time, 12)
        if not np.isfinite(control_time) or not 0 < dt <= .5 or not 0 < source_dt <= .5:
            raise ValueError("Invalid low-pose source/control clocks")
        self.previous_control_time = control_time
        human = self.calibration.apply(frame)
        features = source_pose_features(human, 1.0)
        p = human.positions
        targets = directional_targets(self.robot, human)
        root_p, root_rotation = p[0].copy(), rotation(human.orientations[0])
        head_relative = root_rotation.inv() * rotation(human.orientations[2])
        head_target = root_rotation * head_relative * self.head_neutral_relative.inv() * self.robot_head_neutral
        confidence = np.zeros(2)
        anchored = np.zeros(17, dtype=bool)
        weights = np.array([.4, .5, 1., .5, 1., 1., 2., .8, 1., 2., .8])
        foot_targets = {}
        for side, (knee, ankle) in enumerate(((10, 11), (14, 15))):
            knee_height = features["knee_height_m"][side]
            flexion = features["knee_flexion_rad"][side]
            self.knee_state[side] = flexion > 1.0 and knee_height < (.07 if self.knee_state[side] else .045)
            if self.knee_state[side]:
                if self.knee_anchors[side] is None:
                    self.knee_anchors[side] = targets[knee].copy()
                    if previous is not None:
                        self.knee_anchors[side][:2] = previous.landmarks[knee, :2]
                    self.knee_anchors[side][2] = self.settings["knee_surface_height_m"]
                targets[knee] = self.knee_anchors[side]
                anchored[knee] = True
                weights[self.ids == knee] = 12.0
            else:
                self.knee_anchors[side] = None
            speed = 0.0 if self.previous_human is None else np.linalg.norm(
                p[ankle] - self.previous_human.positions[ankle]) / source_dt
            height = min(p[ankle, 2] - .04, p[ankle + 1, 2] - .015)
            # A leg supported by its knee must not receive a flat-foot ankle prior.
            confidence[side] = (np.clip(1 - max(height, 0) / .07, 0, 1)
                                * np.clip(1 - speed / .5, 0, 1) * (not self.knee_state[side]))
            self.contact_state[side] = confidence[side] >= (.25 if self.contact_state[side] else .55)
            if self.contact_state[side]:
                if self.anchors[side] is None:
                    self.anchors[side] = targets[ankle].copy()
                    if previous is not None:
                        self.anchors[side][:2] = previous.landmarks[ankle, :2]
                    self.anchors[side][2] = .038
                targets[ankle] = self.anchors[side]
                forward = p[ankle + 1, :2] - p[ankle, :2]
                foot_targets[ankle] = Rotation.from_euler("z", np.arctan2(forward[1], forward[0]))
                targets[ankle + 1] = targets[ankle] + foot_targets[ankle].apply([.095, 0, -.02])
                anchored[ankle:ankle+2] = True
                weights[self.ids == ankle] = 12.0
                weights[self.ids == ankle + 1] = 4.0
            else:
                self.anchors[side] = None
        limit, rise = self.settings["root_correction_limit_m"], self.settings["root_correction_speed_m_s"]
        root_lo, root_hi = root_p - limit, root_p + limit
        if previous is not None:
            root_lo = np.maximum(root_lo, root_p + self.root_correction - rise * dt)
            root_hi = np.minimum(root_hi, root_p + self.root_correction + rise * dt)
        self.data.qpos[:3] = root_p + self.root_correction
        self.data.qpos[3:7] = human.orientations[0]
        if previous is None:
            # The first reference is an initialization pose, not a commanded
            # transition from neutral. Seed its leg branch from current source
            # directions; interpolating neutral to a folded knee can sweep the
            # foot through the shin even when the final pose is collision-free.
            for side, (hip, knee, ankle, offset) in enumerate(((9, 10, 11, 10), (13, 14, 15, 16))):
                thigh = root_rotation.inv().apply(p[knee] - p[hip])
                foot = root_rotation.inv().apply(p[ankle + 1] - p[ankle])
                pitch = -np.arctan2(thigh[0], -thigh[2])
                roll = np.arctan2(thigh[1], np.hypot(thigh[0], thigh[2]))
                flexion = features["knee_flexion_rad"][side]
                foot_pitch = -np.arctan2(foot[2], foot[0])
                self.data.qpos[7 + offset:7 + offset + 6] = [
                    pitch, roll, 0, flexion,
                    foot_pitch - pitch - flexion if self.knee_state[side] else -pitch-flexion, 0]
            self.data.qpos[7:] = np.clip(self.data.qpos[7:], self.robot.limits[:, 0]+1e-5,
                                         self.robot.limits[:, 1]-1e-5)
            desired_seed = self.data.qpos[7:].copy()
            margins = self.robot.model.geom_margin.copy()
            try:
                self.robot.model.geom_margin[:] = 0
                for fraction in (1., .875, .75, .5, .25, 0.):
                    self.data.qpos[7:] = self.robot.neutral + fraction*(desired_seed-self.robot.neutral)
                    geometry_forward(self.robot.model, self.data)
                    if not has_self_penetration(self.robot.model, self.data, .00001):
                        break
            finally:
                self.robot.model.geom_margin[:] = margins
        seed = self.data.qpos[7:].copy()
        self.control_origin = self.data.qpos.copy()
        lo, hi = self.robot.limits[:, 0] + 1e-5, self.robot.limits[:, 1] - 1e-5
        if previous is not None:
            step = self.retarget_velocity_limits * dt
            lo, hi = np.maximum(lo, seed-step), np.minimum(hi, seed+step)
        columns = np.r_[0:3, 6:self.robot.model.nv]
        weights = np.repeat(weights, 3)
        jac = np.zeros((len(self.ids)*3, self.robot.model.nv))
        jp, jr = np.zeros((3, self.robot.model.nv)), np.zeros((3, self.robot.model.nv))
        iterations = self.settings["initial_iterations"] if previous is None else self.iterations
        for _ in range(iterations):
            geometry_forward(self.robot.model, self.data)
            positions = self.data.site_xpos[self.robot.site_ids]
            projected = targets + (~anchored)[:, None] * (self.data.qpos[:3] - root_p)
            error = (projected[self.ids] - positions[self.ids]).reshape(-1) * weights
            for i, index in enumerate(self.ids):
                mujoco.mj_jacSite(self.robot.model, self.data, jp, jr, self.robot.site_ids[index])
                if not anchored[index]:
                    jp[:, :3] = 0
                jac[i*3:i*3+3] = jp
            j = jac[:, columns] * weights[:, None]
            root_jac = np.zeros((3, 25))
            root_jac[:, :3] = np.eye(3)
            j = np.vstack([j, .4*root_jac, root_jac])
            error = np.r_[error, .4*(root_p-self.data.qpos[:3]), root_p+self.root_correction-self.data.qpos[:3]]
            knee_jac = np.zeros((2, 25))
            knee_jac[0, 3+13] = knee_jac[1, 3+19] = self.settings["source_knee_angle_weight"]
            desired_flexion = np.clip(features["knee_flexion_rad"],
                                     self.robot.limits[[13,19], 0], self.robot.limits[[13,19], 1])
            j = np.vstack([j, knee_jac])
            error = np.r_[error, self.settings["source_knee_angle_weight"] *
                          (desired_flexion-self.data.qpos[[7+13,7+19]])]
            head_current = Rotation.from_matrix(self.data.site_xmat[self.robot.site_ids[2]].reshape(3, 3))
            mujoco.mj_jacSite(self.robot.model, self.data, jp, jr, self.robot.site_ids[2])
            j = np.vstack([j, .15*jr[:, columns]])
            error = np.r_[error, .15*(head_target*head_current.inv()).as_rotvec()]
            for ankle, desired in foot_targets.items():
                current = Rotation.from_matrix(self.data.site_xmat[self.robot.site_ids[ankle]].reshape(3, 3))
                mujoco.mj_jacSite(self.robot.model, self.data, jp, jr, self.robot.site_ids[ankle])
                j = np.vstack([j, .6*jr[:, columns]])
                error = np.r_[error, .6*(desired*current.inv()).as_rotvec()]
            seen_pairs = set()
            for contact in self.data.contact[:self.data.ncon]:
                b1, b2 = self.robot.model.geom_bodyid[[contact.geom1, contact.geom2]]
                clearance, strength = (.002, 18.) if 0 in (b1, b2) else (.008, 20.)
                distance = float(contact.dist)
                normal = contact.frame[:3]
                first_point = second_point = contact.pos
                if 0 not in (b1, b2):
                    pair = (int(contact.geom1), int(contact.geom2))
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    segment = np.zeros(6)
                    distance = mujoco.mj_geomDistance(self.robot.model, self.data, *pair, .03, segment)
                    if abs(distance) > 1e-8:
                        normal = (segment[3:]-segment[:3])/distance
                        first_point, second_point = segment[:3], segment[3:]
                if distance >= clearance:
                    continue
                j1, j2 = np.zeros_like(jp), np.zeros_like(jp)
                mujoco.mj_jac(self.robot.model, self.data, j1, None, first_point, int(b1))
                mujoco.mj_jac(self.robot.model, self.data, j2, None, second_point, int(b2))
                j = np.vstack([j, strength*(normal @ (j2-j1))[columns]])
                error = np.r_[error, strength*min(.03, clearance-distance)]
            j = np.vstack([j, np.sqrt(.002)*np.eye(25)])
            error = np.r_[error, np.zeros(3), np.sqrt(.002)*((seed+self.robot.neutral)*.5-self.data.qpos[7:])]
            lower = np.r_[root_lo-self.data.qpos[:3], np.maximum(lo-self.data.qpos[7:], -.2)]
            upper = np.r_[root_hi-self.data.qpos[:3], np.minimum(hi-self.data.qpos[7:], .2)]
            delta = lsq_linear(j, error, bounds=(lower, upper), method="bvls", tol=1e-6).x
            old_pose = self.data.qpos.copy()
            self.data.qpos[:3] = np.clip(self.data.qpos[:3]+delta[:3], root_lo, root_hi)
            self.data.qpos[7:] = np.clip(self.data.qpos[7:]+delta[3:], lo, hi)
            self._safe_step(old_pose)
            if np.linalg.norm(self.data.qpos-old_pose) < 1e-4:
                break
        geometry_forward(self.robot.model, self.data)
        root = self.data.qpos[:3].copy()
        self.root_correction = root-root_p
        q, landmarks = self.data.qpos[7:].copy(), self.robot.landmarks(self.data)
        reference_targets = directional_targets(self.robot, human)
        relative_error = ((landmarks-root) - (reference_targets-root_p))[self.ids]
        rms = float(np.sqrt(np.mean(relative_error**2)))
        reasons = []
        if rms > self.settings["max_tick_landmark_error_m"]:
            reasons.append("landmark_error")
        if root[2] < self.settings["minimum_root_height_m"]:
            reasons.append("root_below_low_pose_floor")
        velocity, dq = np.zeros(6), np.zeros(22)
        if previous is not None:
            velocity[:3] = (root-previous.root_position)/dt
            velocity[3:] = (root_rotation*rotation(previous.root_orientation).inv()).as_rotvec()/dt
            dq = (q-previous.joint_position)/dt
        ref = Reference(frame.source_time, frame.received_time, root, human.orientations[0], velocity,
                        q, dq, landmarks, self.contact_state.astype(float), confidence,
                        valid=not reasons, session=frame.session)
        self.previous, self.previous_human = ref, human
        self.last_report = {"rms_landmark_error_m": rms, "rejection_reasons": reasons,
                            "knee_contacts": self.knee_state.astype(int).tolist(),
                            "source_kneeling": features["kneeling"].astype(int).tolist(),
                            "source_hand_support": features["hand_support"],
                            "root_correction_m": self.root_correction.tolist(),
                            "seconds": time.perf_counter()-started}
        return ref


def retarget_low_pose(robot, frames, metadata, settings=None, *, speed_profile=None):
    times = np.array([f.source_time for f in frames])
    ticks, indices = control_schedule(times, robot.control_dt)
    solver = LowPoseRetargeter(robot, settings, speed_profile=speed_profile)
    refs, reports, prior = [], [], -1
    for tick, index in zip(ticks, indices):
        if index != prior:
            source = frames[index]
            frame = HumanFrame(source.source_time, source.source_time-times[0], source.frame_number,
                               source.positions, source.orientations, source.source_id, source.session)
            if prior < 0:
                solver.calibrate(frame, floor=solver.settings["source_floor_m"])
            ref = solver.process(frame, control_time=float(tick))
            prior = index
        refs.append(ref)
        reports.append(dict(solver.last_report))
    calibration = {**solver.calibration.metadata(), "method": "first_frame_leg_lengths_explicit_source_floor",
                   "human_leg_lengths_m": leg_lengths(frames[0].positions).tolist()}
    return MotionClip.from_references(refs, {**metadata,
        **retarget_speed_metadata(robot, LOW_POSE_VERSION, speed_profile),
        "source_adapter": LOW_POSE_SOURCE, "model_signature": robot.signature,
        "calibration": calibration, "low_pose_settings": solver.settings,
        "knee_contacts": [r["knee_contacts"] for r in reports],
        "sampling": {"version": SAMPLING_VERSION, "control_dt": robot.control_dt,
                     "source_frames": len(frames), "processed_source_frames": len(set(indices)),
                     "velocity_bounds_clock": "control_clock"},
        "physics_qualified": False, "training_eligible": False}, sample_times=ticks), reports
