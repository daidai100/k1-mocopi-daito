"""Replay the newest arrived human frame on the controller clock, without lookahead."""

import numpy as np

from .contracts import HumanFrame, MotionClip
from .reference_scaling import stretch_motion_clip
from .retarget import Retargeter
from .retarget_speed import retarget_speed_metadata
from .reference_velocity_clock import PLAYBACK_RETARGET_VERSION, playback_derivatives

SAMPLING_VERSION = "latest-human-at-control-rate-v1"


def control_schedule(source_times, control_dt, arrivals=None, dropped=None):
    times = np.asarray(source_times, dtype=float)
    if (
        times.ndim != 1
        or len(times) < 2
        or not np.isfinite(times).all()
        or np.any(np.diff(times) <= 0)
        or not np.isfinite(control_dt)
        or control_dt <= 0
    ):
        raise ValueError("Replay requires increasing finite source times and a positive control period")
    arrivals = times - times[0] if arrivals is None else np.asarray(arrivals, dtype=float)
    dropped = np.zeros(len(times), bool) if dropped is None else np.asarray(dropped, dtype=bool)
    if arrivals.shape != times.shape or not np.isfinite(arrivals).all() or dropped.shape != times.shape:
        raise ValueError("Invalid arrival or loss schedule")
    count = int(np.floor(np.nextafter((times[-1] - times[0]) / control_dt, np.inf))) + 1
    ticks = np.arange(count) * control_dt
    available = np.flatnonzero(~dropped)
    order = available[np.argsort(arrivals[available], kind="stable")]
    indices = np.full(count, -1, dtype=int)
    if len(order):
        # Late older packets must not replace a newer accepted source frame.
        newest = np.maximum.accumulate(order)
        positions = np.searchsorted(arrivals[order], ticks, side="right") - 1
        present = positions >= 0
        indices[present] = newest[positions[present]]
    return ticks, indices


def retarget_at_control_rate(robot, human, metadata, *, speed_profile=None, motion_profile=None):
    times = np.asarray(human["times"], dtype=float)
    effective_times = times
    ticks, indices = control_schedule(effective_times, robot.control_dt)
    if indices[0] != 0 or len(ticks) < 2:
        raise ValueError("Control-rate replay needs an initial frame and at least two ticks")
    retarget = Retargeter(robot, speed_profile=speed_profile, motion_profile=motion_profile)
    references, reports = [], []
    previous = -1
    for index in indices:
        if index != previous:
            frame = HumanFrame(
                float(effective_times[index]),
                float(effective_times[index] - effective_times[0]),
                int(index),
                human["positions"][index],
                human["orientations"][index],
                metadata["source_motion_id"],
            )
            if previous < 0:
                retarget.calibrate(frame)
            reference = retarget.process(frame)
            previous = index
        references.append(reference)
        reports.append(dict(retarget.last_report))
    speed_metadata = retarget_speed_metadata(robot, PLAYBACK_RETARGET_VERSION, speed_profile)
    if motion_profile is not None:
        speed_metadata["retarget_version"] += f"+{motion_profile}"
    sampling = {
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
        "velocity_clock": "sample_clock",
        "first_velocity_frame": "zero causal derivative",
    }
    clip = MotionClip.from_references(
        references,
        {
            **metadata,
            **speed_metadata,
            **({"motion_scale": retarget.calibration.metadata()["motion_scale"]}
               if motion_profile is not None else {}),
            **({"model_signature": robot.signature, "kinematics_accepted": False,
                "physics_qualified": False, "training_eligible": False,
                "experimental_reference": True} if motion_profile is not None else {}),
            "sampling": sampling,
            "calibration": retarget.calibration.metadata(),
        },
        sample_times=ticks,
    )
    if motion_profile is not None:
        clip = MotionClip(clip.times, clip.values, clip.metadata, times[indices],
                          effective_times[indices] - effective_times[0])
    # Retargeting observes source timestamps, while this artifact plays on the
    # controller clock. Derive from the saved pose sequence, including zero
    # velocity on held frames, without changing IK, poses, or source clocks.
    joint_velocity, root_velocity = playback_derivatives(clip)
    clip = MotionClip(
        clip.times,
        {**clip.values, "joint_velocity": joint_velocity, "root_velocity": root_velocity},
        clip.metadata,
        clip.source_times,
        clip.received_times,
    )
    if motion_profile is not None:
        clip = stretch_motion_clip(clip, robot, clip.metadata["motion_scale"]["clock_scale"])
    for report in reports:
        report["retarget_version"] = clip.metadata["retarget_version"]
        report["velocity_clock"] = "sample_clock"
        if motion_profile is not None:
            report["report_clock"] = "pre_stretch_source_control_clock"
    return clip, reports
