"""Fail-closed admission and fall-height contract for audited low-pose goals."""
import math

import numpy as np

LOW_SUPPORT_TASK = "flat_ground_knee_and_foot_support_tracking_v1"
LOW_SUPPORT_ADMISSION = "low-support-rl-reference-v1"


def low_pose_reference_audit(geometry):
    from .low_pose_validation import LOW_POSE_AUDIT_VERSION, LOW_POSE_GATES
    reasons = list(geometry.get("rejection_reasons", []))
    family = geometry.get("family")
    if (geometry.get("version") != LOW_POSE_AUDIT_VERSION
            or geometry.get("gates") != LOW_POSE_GATES
            or not geometry.get("geometry_audited") or geometry.get("audit_hz") != 500
            or not geometry.get("full_source_duration_preserved")
            or geometry.get("knee_flexion_target") != "source_clamped_to_unchanged_robot_joint_limits"
            or not geometry.get("accepted")):
        reasons.append("missing_low_pose_source_geometry_audit")
    bounds = {
        "max_self_penetration_m": .0001, "max_ground_penetration_m": .005,
        "unsupported_body_penetration_m": .0001, "saved_landmark_error_m": 1e-7,
        "joint_speed_max_rad_s": 6+1e-6, "stance_slip_p95_m_s": .2,
        "mean_directional_landmark_rms_m": .035, "max_directional_landmark_rms_m": .08,
        "feasible_knee_flexion_error_p95_rad": .3, "source_hand_support_seconds": .1,
        "self_collision_samples": 0, "ground_penetration_samples": 0,
    }
    for key, limit in bounds.items():
        value = geometry.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= limit:
            reasons.append("low_pose_"+key)
    ratio = geometry.get("vertical_motion_ratio")
    if ratio is not None and (not math.isfinite(ratio) or not .7 <= ratio <= 1.3):
        reasons.append("low_pose_vertical_motion_ratio")
    if family == "kneel":
        for key, minimum in (("source_kneeling_seconds", .3), ("genuine_kneeling_seconds", .3),
                             ("knee_phase_recall", .9), ("knee_label_precision", .9)):
            value = geometry.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < minimum:
                reasons.append("low_pose_"+key)
    elif family != "squat":
        reasons.append("unsupported_low_pose_family")
    return {"version": LOW_SUPPORT_ADMISSION, "accepted": not reasons, "rejection_reasons": reasons,
            "family": family, "simulation_task": LOW_SUPPORT_TASK,
            "geometry_version": geometry.get("version"), "controller_success_required": False,
            "physics_qualified": False}


def low_pose_accepted(row):
    from .low_pose import LOW_POSE_SOURCE, LOW_POSE_VERSION
    receipt = row.get("low_pose_reference_audit", {})
    if (row.get("family") not in {"kneel", "squat"} or row.get("rejected_ticks", 1) != 0
            or row.get("retarget_version") != LOW_POSE_VERSION
            or row.get("source_adapter") != LOW_POSE_SOURCE
            or not row.get("kinematics_accepted") or not receipt.get("accepted")):
        return False
    expected = low_pose_reference_audit(row.get("recovery_audit", {}))
    return expected == receipt and expected["accepted"] and receipt["family"] == row["family"]


def minimum_tracking_height(root_height, *, low_support=False, standing_minimum=.22):
    """Only verified low-pose references get a reference-relative fall floor."""
    height = np.asarray(root_height)
    if not np.isfinite(height).all():
        raise ValueError("Nonfinite reference height")
    return np.maximum(.10, np.minimum(standing_minimum, height-.07)) if low_support else np.full_like(
        height, standing_minimum, dtype=float)


def minimum_tracking_upright(reference_up, *, low_support=False, standing_minimum=.2):
    """Allow an intended deep torso lean, never an unrestricted inversion."""
    up = np.asarray(reference_up)
    if not np.isfinite(up).all() or np.any(abs(up) > 1+1e-6):
        raise ValueError("Invalid reference upright cosine")
    return np.maximum(-.25, np.minimum(standing_minimum, up-.35)) if low_support else np.full_like(
        up, standing_minimum, dtype=float)
