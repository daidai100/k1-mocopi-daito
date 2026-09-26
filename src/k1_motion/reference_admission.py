"""Admission of retargeted references for simulated motion-tracking RL.

The learner supplies balance. Success by a previously trained controller is
diagnostic evidence, never an admission requirement for a motion reference.
"""

import math
import re


ADMISSION_VERSION = "k1-retargeted-rl-reference-v1"
DEFERRED_CONTACT_FAMILIES = {"climb", "crawl", "inversion_or_stunt", "fall_or_recovery", "sit_or_kneel"}
WALKING_RL_ADMISSION_VERSION = "walking-rl-ground-v1"
GROUNDED_RL_ADMISSION_VERSION = "flat-ground-rl-ground-v1"
# Explicitly reviewed foot-supported tasks. Jump/low-support tasks need their
# own phase/contact validation; a family rename must not widen this contract.
GROUNDED_RL_FAMILIES = frozenset({"transition", "turn", "run", "idle_stance", "dance", "squat"})
WALKING_RL_GROUND_LIMITS = {
    "max_penetration_m": 0.025,
    "mean_penetration_m": 0.0005,
    "fraction_over_5mm": 0.02,
    "longest_over_5mm_s": 0.06,
    "max_nonfoot_penetration_m": 0.005,
}


def walking_reference_audit(geometry):
    """RL-specific ground tolerance, separate from strict geometric acceptance."""
    reasons = [r for r in geometry.get("rejection_reasons", [])
               if r != "ground_penetration_on_command_path"]
    profile = geometry.get("ground_profile", {})
    if not geometry.get("geometry_audited") or geometry.get("audit_hz", 0) < 500:
        reasons.append("missing_independent_geometry_audit")
    for key, limit in WALKING_RL_GROUND_LIMITS.items():
        value = profile.get(key)
        if value is None or not math.isfinite(value) or value < 0 or value > limit + 1e-12:
            reasons.append("walking_ground_" + key)
    return {"version": WALKING_RL_ADMISSION_VERSION, "accepted": not reasons,
            "rejection_reasons": reasons, "ground_measurements": profile,
            "ground_limits": WALKING_RL_GROUND_LIMITS,
            "strict_geometry_accepted": bool(geometry.get("accepted")),
            "reset_height_correction_required": not bool(geometry.get("accepted")),
            "controller_success_required": False, "physics_qualified": False}


def walking_rl_accepted(row):
    audit = row.get("rl_reference_audit", {})
    if (row.get("family") != "walk" or row.get("rejected_ticks", 1) != 0
            or audit.get("version") != WALKING_RL_ADMISSION_VERSION
            or audit.get("ground_limits") != WALKING_RL_GROUND_LIMITS
            or not audit.get("accepted")):
        return False
    verified = walking_reference_audit(row.get("recovery_audit", {}))
    return verified == audit and verified["accepted"]


def grounded_reference_audit(geometry, family):
    """Versioned extension of the bounded sole-error policy, not a strict pass."""
    audit = walking_reference_audit(geometry)
    reasons = [reason.replace("walking_ground_", "grounded_ground_")
               for reason in audit["rejection_reasons"]]
    if family not in GROUNDED_RL_FAMILIES:
        reasons.append("unsupported_ground_reference_family")
    return {**audit, "version": GROUNDED_RL_ADMISSION_VERSION, "family": family,
            "accepted": not reasons, "rejection_reasons": reasons}


def ground_rl_accepted(row):
    """Verify either the immutable walking-v1 or the grounded-family contract."""
    if walking_rl_accepted(row):
        return True
    audit = row.get("rl_reference_audit", {})
    family = row.get("family")
    if (family not in GROUNDED_RL_FAMILIES or row.get("rejected_ticks", 1) != 0
            or audit.get("version") != GROUNDED_RL_ADMISSION_VERSION
            or not audit.get("accepted")):
        return False
    verified = grounded_reference_audit(row.get("recovery_audit", {}), family)
    return verified == audit and verified["accepted"]


def take_family(capture_group):
    """Group BONES take numbers while retaining the original split labels."""
    if capture_group.startswith("bones_seed/"):
        return re.sub(r"_\d+$", "", capture_group)
    return capture_group


def reference_rejections(row, held_out_families=()):
    reasons = []
    if row.get("is_mirror"):
        reasons.append("mirror_deferred")
    if not (row.get("kinematics_accepted") and row.get("recovery_audit", {}).get("accepted")
            or ground_rl_accepted(row)):
        reasons.append("retargeting_audit_failed")
    if row.get("rejected_ticks", 1) != 0:
        reasons.append("invalid_reference_ticks")
    if row.get("frames", 0) < 2 or not row.get("reference_path"):
        reasons.append("missing_reference")
    if row.get("split") not in ("train", "validation", "test"):
        reasons.append("unknown_split")
    if row.get("split") == "train" and take_family(row["capture_group"]) in held_out_families:
        reasons.append("related_held_out_take")
    if row.get("family") in DEFERRED_CONTACT_FAMILIES:
        reasons.append("contact_task_not_configured")
    if row.get("family") == "kneel" or "low_pose_reference_audit" in row:
        from .low_pose_contract import low_pose_accepted
        if not low_pose_accepted(row):
            reasons.append("low_support_task_not_audited")
    description = " ".join([row.get("take_name", ""), *row.get("annotations", [])]).lower()
    if re.search(r"crutch|ladder|wheelchair|\b(?:sit|sitting|seated)\b.*\b(?:chair|bench|sofa)\b"
                 r"|\b(?:lean|leaning)\b.*\b(?:wall|table|rail)\b", description) or (
            re.search(r"\bsupport\w*\b", description)
            and re.search(r"\b(?:wall|table|rail|chair|bench|sofa)\b", description)):
        reasons.append("external_support_not_configured")
    return reasons


def admit_reference(row):
    """Called only after the independent retargeting and payload checks pass."""
    low_support = False
    if "low_pose_reference_audit" in row:
        from .low_pose_contract import low_pose_accepted
        low_support = low_pose_accepted(row)
    return {
        **row,
        "training_eligible": row["split"] == "train",
        "rl_training_eligible": row["split"] == "train",
        "training_stage": "motion_tracking_rl",
        "physics_qualified": False,
        "reference_admission": {
            "version": ADMISSION_VERSION,
            "reference_kind": "retargeted_kinematic_goal",
            "controller_success_required": False,
            "simulation_task": ("flat_ground_knee_and_foot_support_tracking_v1" if low_support
                                else "flat_ground_unloaded_motion_tracking"),
            "balance_is_learned": True,
            "qualified_robot_demonstration": False,
        },
    }
