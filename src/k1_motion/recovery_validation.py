"""Independent pose/path checks for recovery; these never imply physics success."""

from collections import Counter

import mujoco
import numpy as np

from .contracts import HumanFrame, MotionClip
from .reference_scaling import stretch_motion_clip
from .retarget_recovery import RETARGET_VERSION, RecoveryRetargeter
from .recovery_geometry import geometry_forward, recovery_model
from .streaming import SAMPLING_VERSION, control_schedule
from .retarget_speed import retarget_speed_contract, retarget_speed_metadata

RECOVERY_GATES = {
    "self_penetration_tolerance_m": 0.0001,
    "max_ground_penetration_m": 0.005,
    "stance_slip_p95_m_s": 0.2,
    "max_mean_landmark_error_increase_m": 0.015,
    "max_joint_rms_change_rad": 0.2,
    "minimum_motion_amplitude_ratio": 0.7,
    "maximum_motion_amplitude_ratio": 1.3,
}


def retarget_recovery(robot, human, metadata, *, speed_profile=None, motion_profile=None, control_tick_hold=False):
    times = np.asarray(human["times"], dtype=float)
    effective_times = times
    ticks, indices = control_schedule(effective_times, robot.control_dt)
    if indices[0] != 0 or len(ticks) < 2:
        raise ValueError("Recovery needs an initial frame and at least two control ticks")
    retargeter = RecoveryRetargeter(robot, speed_profile=speed_profile, motion_profile=motion_profile,
                                   control_tick_hold=control_tick_hold)
    references, reports = [], []
    previous = -1
    for control_time, index in zip(ticks, indices):
        if index != previous:
            frame = HumanFrame(
                float(effective_times[index]),
                float(effective_times[index] - effective_times[0]), int(index),
                human["positions"][index], human["orientations"][index],
                metadata["source_motion_id"],
            )
            if previous < 0:
                retargeter.calibrate(frame)
            reference = retargeter.process(frame, control_time=float(control_time))
            previous = index
        references.append(reference)
        reports.append(dict(retargeter.last_report))
    clip = MotionClip.from_references(references, {
        **metadata,
        **retarget_speed_metadata(robot, RETARGET_VERSION, speed_profile),
        **({"retarget_version": retargeter.retarget_version} if motion_profile is not None or control_tick_hold else {}),
        **({'command_hold_contract': 'control-tick-hold-v1; every pose delta bounded over one output tick'}
           if control_tick_hold else {}),
        **({"motion_scale": retargeter.calibration.metadata()["motion_scale"]}
           if motion_profile is not None else {}),
        "model_signature": robot.signature,
        "calibration": retargeter.calibration.metadata(),
        "sampling": {
            "version": SAMPLING_VERSION,
            "control_dt": robot.control_dt,
            "source_frames": len(times),
            "processed_source_frames": len(set(indices)),
            "sample_clock": ("retimed_seconds_since_first_canonical_human_frame"
                             if motion_profile is not None else "seconds_since_first_canonical_human_frame"),
            "source_clock": ("interpolated_original_canonical_human_phase_seconds"
                             if motion_profile is not None else "original_canonical_human_seconds"),
            "received_clock": ("retimed_seconds_since_first_canonical_human_frame; clean replay"
                               if motion_profile is not None else
                               "seconds_since_first_canonical_human_frame; clean replay"),
            "velocity_bounds_clock": "control_clock",
        },
        "physics_qualified": False,
        "training_eligible": False,
        **({"experimental_reference": True, "kinematics_accepted": False}
           if motion_profile is not None else {}),
    }, sample_times=ticks)
    if control_tick_hold:
        from .reference_velocity_clock import playback_derivatives
        joint_velocity, root_velocity = playback_derivatives(clip)
        clip = MotionClip(clip.times,{**clip.values,'joint_velocity':joint_velocity,'root_velocity':root_velocity},
                          clip.metadata,clip.source_times,clip.received_times)
        for report in reports:
            report['retarget_version'] = retargeter.retarget_version
            report['velocity_clock'] = 'sample_clock; zero derivative during source holds'
    if motion_profile is not None:
        clip = stretch_motion_clip(clip, robot, clip.metadata["motion_scale"]["clock_scale"])
        for report in reports:
            report["retarget_version"] = clip.metadata["retarget_version"]
            report["report_clock"] = "pre_stretch_source_control_clock"
    return clip, reports


def audit_recovery(clip, baseline, reports, baseline_row, *, ground_profile=False, speed_profile=None):
    """Reloaded poses on an untouched model, including every 2 ms path sample."""
    robot = recovery_model("audit")
    model, data = robot.model, robot.data
    if clip.metadata.get("model_signature") != robot.signature:
        raise ValueError("Recovery/model signature mismatch")
    if (not np.array_equal(clip.times, baseline.times)
            or not np.array_equal(clip.source_times, baseline.source_times)
            or not np.array_equal(clip.received_times, baseline.received_times)):
        raise ValueError("Recovery changed motion duration or source/receive clocks")
    dt = np.diff(clip.times)
    if not np.allclose(dt, robot.control_dt, atol=1e-12, rtol=0):
        raise ValueError("Recovery audit requires the fixed control clock")
    qpos = np.c_[clip.values["root_position"], clip.values["root_orientation"],
                 clip.values["joint_position"]]
    velocity = np.zeros(model.nv)
    self_depth = ground_depth = 0.0
    self_bad_samples = ground_bad_samples = 0
    pairs = Counter()
    samples = 0
    ground_depths = []
    ground_run = longest_ground_run = 0
    nonfoot_ground_depth = 0.0
    feet = {model.body(f"{side}_ankle_roll_link").id for side in ("left", "right")}
    for index, pose in enumerate(qpos):
        fractions = [1.0] if index == 0 else np.arange(1, robot.substeps + 1) / robot.substeps
        if index:
            mujoco.mj_differentiatePos(model, velocity, 1.0, qpos[index - 1], pose)
        for fraction in fractions:
            data.qpos[:] = qpos[index - 1] if index else pose
            if index:
                mujoco.mj_integratePos(model, data.qpos, velocity, float(fraction))
            geometry_forward(model, data)
            max_self = max_ground = 0.0
            for contact in data.contact[:data.ncon]:
                bodies = model.geom_bodyid[[contact.geom1, contact.geom2]]
                depth = max(0.0, -float(contact.dist))
                if 0 in bodies:
                    max_ground = max(max_ground, depth)
                    if ground_profile and int(max(bodies)) not in feet:
                        nonfoot_ground_depth = max(nonfoot_ground_depth, depth)
                else:
                    max_self = max(max_self, depth)
                    if depth > RECOVERY_GATES["self_penetration_tolerance_m"]:
                        pairs[" / ".join(sorted(model.body(int(b)).name for b in bodies))] += 1
            self_depth = max(self_depth, max_self)
            ground_depth = max(ground_depth, max_ground)
            self_bad_samples += max_self > RECOVERY_GATES["self_penetration_tolerance_m"]
            ground_bad_samples += max_ground > RECOVERY_GATES["max_ground_penetration_m"]
            samples += 1
            if ground_profile:
                ground_depths.append(max_ground)
                ground_run = ground_run + 1 if max_ground > RECOVERY_GATES["max_ground_penetration_m"] else 0
                longest_ground_run = max(longest_ground_run, ground_run)
    speed = np.abs(np.diff(clip.values["joint_position"], axis=0) / dt[:, None])
    speed_contract = retarget_speed_contract(robot, speed_profile)
    speed_limits = np.asarray(speed_contract["joint_velocity_limits_rad_s"])
    stance = (clip.values["contacts"][1:] > 0.5) & (clip.values["contacts"][:-1] > 0.5)
    foot_speed = np.linalg.norm(
        np.diff(clip.values["landmarks"][:, [11, 15], :2], axis=0) / dt[:, None, None], axis=-1)
    slip = float(np.percentile(foot_speed[stance], 95)) if stance.any() else 0.0
    joint_change = float(np.sqrt(np.mean((clip.values["joint_position"]
                                        - baseline.values["joint_position"]) ** 2)))
    old_amplitude = np.linalg.norm(np.std(baseline.values["joint_position"], axis=0))
    new_amplitude = np.linalg.norm(np.std(clip.values["joint_position"], axis=0))
    amplitude_ratio = float(new_amplitude / old_amplitude) if old_amplitude > 1e-6 else None
    mean_error = float(np.mean([r["rms_landmark_error_m"] for r in reports]))
    reasons = []
    if not np.all(clip.values["valid"]):
        reasons.append("retarget_invalid_ticks")
    if self_bad_samples:
        reasons.append("self_collision_on_command_path")
    if ground_bad_samples:
        reasons.append("ground_penetration_on_command_path")
    if np.any(speed > speed_limits + 1e-6):
        reasons.append("control_clock_velocity_limit")
    if slip > RECOVERY_GATES["stance_slip_p95_m_s"]:
        reasons.append("stance_foot_slip")
    if joint_change > RECOVERY_GATES["max_joint_rms_change_rad"]:
        reasons.append("motion_joint_distortion")
    if mean_error > baseline_row["rms_landmark_error_m"] + RECOVERY_GATES["max_mean_landmark_error_increase_m"]:
        reasons.append("human_tracking_regression")
    if amplitude_ratio is not None and not (RECOVERY_GATES["minimum_motion_amplitude_ratio"]
                                           <= amplitude_ratio <= RECOVERY_GATES["maximum_motion_amplitude_ratio"]):
        reasons.append("motion_amplitude_changed")
    result = {
        "version": "recovery-independent-audit-v1",
        "geometry_audited": True,
        "accepted": not reasons,
        "rejection_reasons": reasons,
        "audit_hz": 1.0 / model.opt.timestep,
        "samples": samples,
        "self_collision_samples": int(self_bad_samples),
        "ground_penetration_samples": int(ground_bad_samples),
        "max_self_penetration_m": self_depth,
        "max_ground_penetration_m": ground_depth,
        "collision_pairs": dict(pairs),
        "joint_speed_max_rad_s": float(speed.max(initial=0)),
        "stance_slip_p95_m_s": slip,
        "joint_rms_change_rad": joint_change,
        "motion_amplitude_ratio": amplitude_ratio,
        "rms_landmark_error_m": mean_error,
        "gates": RECOVERY_GATES,
        "physics_qualified": False,
        "training_eligible": False,
    }
    if ground_profile:
        result["ground_profile"] = {
            "max_penetration_m": ground_depth,
            "mean_penetration_m": float(np.mean(ground_depths)),
            "p95_penetration_m": float(np.percentile(ground_depths, 95)),
            "fraction_over_5mm": float(ground_bad_samples / samples),
            "longest_over_5mm_s": float(longest_ground_run * model.opt.timestep),
            "max_nonfoot_penetration_m": nonfoot_ground_depth,
        }
    if speed_profile not in (None, "legacy-command-v1"):
        result.update(version=f"recovery-independent-audit-v1+{speed_contract['profile']}",
                      retarget_speed_contract=speed_contract)
    return result
