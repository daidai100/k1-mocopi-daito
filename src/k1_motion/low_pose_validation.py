"""Independent full-duration geometry and source-pose audit for low support."""
from collections import Counter

import mujoco
import numpy as np

from .low_pose import LOW_POSE_SOURCE, LOW_POSE_VERSION, directional_targets, skeletal_calibration, source_pose_features
from .recovery_geometry import geometry_forward, recovery_model
from .streaming import control_schedule

LOW_POSE_AUDIT_VERSION = "knee-support-source-geometry-v1"
LOW_POSE_GATES = {
    "self_penetration_tolerance_m": .0001,
    "max_ground_penetration_m": .005,
    "stance_slip_p95_m_s": .2,
    "max_mean_directional_landmark_rms_m": .035,
    "max_directional_landmark_rms_m": .08,
    "max_feasible_knee_flexion_error_p95_rad": .3,
    "min_knee_phase_recall": .9,
    "min_knee_phase_duration_s": .3,
    "max_source_hand_support_s": .1,
    "min_vertical_motion_ratio": .7,
    "max_vertical_motion_ratio": 1.3,
}


def audit_low_pose(clip, source_frames, family):
    """Never compare against a broken standing-calibration reference as truth.

    Source directions, flexion, support phases and vertical excursion are
    recomputed independently. Neither IK reports nor semantic labels prove a
    kneel. All saved ticks and interpolated 2ms command poses are checked.
    """
    robot = recovery_model("audit")
    model, data = robot.model, robot.data
    if (clip.metadata.get("model_signature") != robot.signature
            or clip.metadata.get("retarget_version") != LOW_POSE_VERSION
            or clip.metadata.get("source_adapter") != LOW_POSE_SOURCE):
        raise ValueError("Low-pose model/source/retarget version mismatch")
    times = np.array([f.source_time for f in source_frames])
    ticks, indices = control_schedule(times, robot.control_dt)
    if (not np.array_equal(clip.times, ticks)
            or not np.array_equal(clip.source_times, times[indices])
            or not np.array_equal(clip.received_times, times[indices]-times[0])):
        raise ValueError("Low-pose export changed full source duration or causal clocks")
    calibration = skeletal_calibration(source_frames[0], robot)
    features = [source_pose_features(source_frames[i], calibration.scale) for i in indices]
    source_knees = np.array([f["kneeling"] for f in features])
    source_flex = np.array([f["knee_flexion_rad"] for f in features])
    declared_knees = np.asarray(clip.metadata["knee_contacts"])
    if declared_knees.shape != source_knees.shape or not np.isin(declared_knees, [0, 1]).all():
        raise ValueError("Invalid knee support payload")
    feet = [model.body(f"{side}_ankle_roll_link").id for side in ("left", "right")]
    ground = model.geom("ground").id
    knees = [model.body(f"{side}_knee_pitch_link").id for side in ("left", "right")]
    allowed = set(feet + knees)
    qpos = np.c_[clip.values["root_position"], clip.values["root_orientation"], clip.values["joint_position"]]
    velocity = np.zeros(model.nv)
    pairs = Counter()
    max_self = max_ground = forbidden_depth = landmark_mismatch = 0.0
    self_bad = ground_bad = samples = 0
    measured_knees = []
    directional_errors = []
    for index, pose in enumerate(qpos):
        fractions = [1.] if index == 0 else np.arange(1, robot.substeps+1)/robot.substeps
        if index:
            mujoco.mj_differentiatePos(model, velocity, 1., qpos[index-1], pose)
        for fraction in fractions:
            data.qpos[:] = qpos[index-1] if index else pose
            if index:
                mujoco.mj_integratePos(model, data.qpos, velocity, float(fraction))
            geometry_forward(model, data)
            self_depth = ground_depth = 0.
            for c in data.contact[:data.ncon]:
                bodies = model.geom_bodyid[[c.geom1, c.geom2]]
                depth = max(0., -float(c.dist))
                if 0 in bodies:
                    ground_depth = max(ground_depth, depth)
                    if int(max(bodies)) not in allowed:
                        forbidden_depth = max(forbidden_depth, depth)
                else:
                    self_depth = max(self_depth, depth)
                    if depth > LOW_POSE_GATES["self_penetration_tolerance_m"]:
                        pairs[" / ".join(sorted(model.body(int(b)).name for b in bodies))] += 1
            max_self, max_ground = max(max_self, self_depth), max(max_ground, ground_depth)
            self_bad += self_depth > LOW_POSE_GATES["self_penetration_tolerance_m"]
            ground_bad += ground_depth > LOW_POSE_GATES["max_ground_penetration_m"]
            samples += 1
        landmarks = robot.landmarks(data)
        landmark_mismatch = max(landmark_mismatch, float(np.max(abs(landmarks-clip.values["landmarks"][index]))))
        # A low knee marker alone is insufficient: the actual collision shape
        # must be close to the floor, and the joint must genuinely be folded.
        measured = []
        for side, body in enumerate(knees):
            distances = [mujoco.mj_geomDistance(model, data, ground, int(g), .1, None)
                         for g in np.flatnonzero((model.geom_bodyid == body) & (model.geom_contype > 0))]
            measured.append(min(distances, default=.1) <= .01 and landmarks[10 if side == 0 else 14, 2] <= .065
                            and pose[7+(13 if side == 0 else 19)] > 1.)
        measured_knees.append(measured)
        human = calibration.apply(source_frames[indices[index]])
        expected = directional_targets(robot, human)
        relative = landmarks-pose[:3] - (expected-human.positions[0])
        directional_errors.append(float(np.sqrt(np.mean(relative[[2,4,5,7,8,10,11,12,14,15,16]]**2))))
    measured_knees = np.array(measured_knees)
    dt = robot.control_dt
    knee_recall = float((measured_knees & source_knees).sum()/source_knees.sum()) if source_knees.any() else 0.
    knee_duration = float(measured_knees.any(axis=1).sum()*dt)
    source_knee_duration = float(source_knees.any(axis=1).sum()*dt)
    knee_labels_match = float((measured_knees & declared_knees.astype(bool)).sum()/declared_knees.sum()) if declared_knees.any() else 1.
    support = np.c_[clip.values["contacts"], declared_knees] > .5
    stance = support[1:] & support[:-1]
    support_speed = np.linalg.norm(np.diff(clip.values["landmarks"][:, [11,15,10,14], :2], axis=0)/dt, axis=-1)
    slip = float(np.percentile(support_speed[stance], 95)) if stance.any() else 0.
    q = clip.values["joint_position"]
    speed = abs(np.diff(q, axis=0)/dt)
    feasible_flex = np.clip(source_flex, robot.limits[[13,19], 0], robot.limits[[13,19], 1])
    knee_error = float(np.percentile(abs(q[:, [13,19]]-feasible_flex), 95))
    raw_knee_error = float(np.percentile(abs(q[:, [13,19]]-source_flex), 95))
    source_height = np.array([f["root_height_m"] for f in features])
    source_excursion = float(np.ptp(source_height))
    excursion = float(np.ptp(clip.values["root_position"][:, 2]))
    ratio = excursion/source_excursion if source_excursion >= .08 else None
    hand_seconds = float(sum(f["hand_support"] for f in features)*dt)
    standing = (source_flex.max(axis=1) < .5) & (source_height > .4)
    phase = "standing_to_kneeling" if standing[:10].any() and source_knees[-10:].any() else (
        "kneeling_to_standing" if source_knees[:10].any() and standing[-10:].any() else (
        "standing_kneeling_cycle" if standing.any() and source_knees.any() else "kneeling_hold_or_motion"))
    reasons = []
    if not clip.values["valid"].all():
        reasons.append("retarget_invalid_ticks")
    if self_bad:
        reasons.append("self_collision_on_command_path")
    if ground_bad:
        reasons.append("ground_penetration_on_command_path")
    if forbidden_depth > .0001:
        reasons.append("unsupported_body_ground_contact")
    if np.any(q < robot.limits[:, 0]-1e-8) or np.any(q > robot.limits[:, 1]+1e-8):
        reasons.append("joint_position_limit")
    if np.any(speed > np.minimum(robot.velocity_limit, robot.config["command_velocity_limit"])+1e-6):
        reasons.append("control_clock_velocity_limit")
    if landmark_mismatch > 1e-7:
        reasons.append("saved_landmarks_disagree_with_model")
    if slip > LOW_POSE_GATES["stance_slip_p95_m_s"]:
        reasons.append("support_slip")
    if (np.mean(directional_errors) > LOW_POSE_GATES["max_mean_directional_landmark_rms_m"]
            or max(directional_errors) > LOW_POSE_GATES["max_directional_landmark_rms_m"]):
        reasons.append("source_direction_fidelity")
    if knee_error > LOW_POSE_GATES["max_feasible_knee_flexion_error_p95_rad"]:
        reasons.append("source_knee_flexion_fidelity")
    if hand_seconds > LOW_POSE_GATES["max_source_hand_support_s"]:
        reasons.append("hand_supported_motion_not_configured")
    if ratio is not None and not LOW_POSE_GATES["min_vertical_motion_ratio"] <= ratio <= LOW_POSE_GATES["max_vertical_motion_ratio"]:
        reasons.append("source_vertical_motion_fidelity")
    if family == "kneel":
        if min(knee_duration, source_knee_duration) < LOW_POSE_GATES["min_knee_phase_duration_s"]:
            reasons.append("no_genuine_kneeling_phase")
        if min(knee_recall, knee_labels_match) < LOW_POSE_GATES["min_knee_phase_recall"]:
            reasons.append("knee_support_phase_not_preserved")
    elif family == "squat":
        if source_knee_duration > .1 or knee_duration > .1:
            reasons.append("squat_contains_kneeling")
        if float((source_flex.mean(axis=1) > .6).sum()*dt) < .3:
            reasons.append("no_genuine_squat_phase")
    else:
        reasons.append("unsupported_low_pose_family")
    return {"version": LOW_POSE_AUDIT_VERSION, "accepted": not reasons, "family": family,
            "rejection_reasons": reasons, "geometry_audited": True, "audit_hz": 1/model.opt.timestep,
            "samples": samples, "self_collision_samples": int(self_bad), "ground_penetration_samples": int(ground_bad),
            "max_self_penetration_m": max_self, "max_ground_penetration_m": max_ground,
            "unsupported_body_penetration_m": forbidden_depth, "collision_pairs": dict(pairs),
            "joint_speed_max_rad_s": float(speed.max(initial=0)), "stance_slip_p95_m_s": slip,
            "mean_directional_landmark_rms_m": float(np.mean(directional_errors)),
            "max_directional_landmark_rms_m": max(directional_errors), "saved_landmark_error_m": landmark_mismatch,
            "feasible_knee_flexion_error_p95_rad": knee_error,
            "raw_knee_flexion_error_p95_rad": raw_knee_error,
            "knee_flexion_target": "source_clamped_to_unchanged_robot_joint_limits",
            "source_knee_limit_excess_p95_rad": float(np.percentile(abs(source_flex-feasible_flex), 95)),
            "source_knee_limit_excess_fraction": float(np.mean(abs(source_flex-feasible_flex) > .001)),
            "source_kneeling_seconds": source_knee_duration,
            "genuine_kneeling_seconds": knee_duration, "knee_phase_recall": knee_recall,
            "knee_label_precision": knee_labels_match, "source_hand_support_seconds": hand_seconds,
            "source_vertical_excursion_m": source_excursion, "robot_vertical_excursion_m": excursion,
            "vertical_motion_ratio": ratio, "phase": phase if family == "kneel" else "squat",
            "gates": LOW_POSE_GATES, "full_source_duration_preserved": True,
            "physics_qualified": False, "controller_success_required": False}
