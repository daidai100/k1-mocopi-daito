"""Causal, bounded sole-clearance correction for walking RL references.

Only world height changes. Joint motion, horizontal travel and source clocks
remain exact; an independent audit still decides whether a clip is usable.
"""

import mujoco
import numpy as np

from .contracts import MotionClip
from .recovery_geometry import geometry_forward, recovery_model

GROUND_REFERENCE_VERSION = "causal-sole-clearance-v1"
GROUND_REFERENCE_SETTINGS = {
    "max_lift_m": 0.05,
    "max_rise_m_s": 0.5,
    "release_m_s": 0.1,
    "clearance_m": 0.001,
}


def correct_walking_ground(clip, settings=None):
    """Lift each current pose using only the previous and current commands.

    For a flat plane, a vertical translation changes signed ground distance by
    exactly that translation. At fraction f of a command interval the shift is
    (1-f)*previous_lift + f*current_lift. This gives the required current lift
    from the same 2 ms path samples used by the independent auditor. Correction
    bounds can leave residual penetration; those candidates remain rejectable.
    """
    settings = dict(GROUND_REFERENCE_SETTINGS if settings is None else settings)
    return _correct_ground(clip, settings, ("left_ankle_roll_link", "right_ankle_roll_link"),
                           GROUND_REFERENCE_VERSION)


def _correct_ground(clip, settings, support_bodies, version):
    """Shared translation math; each public task supplies its own audited support set."""
    if (set(settings) != set(GROUND_REFERENCE_SETTINGS)
            or not all(np.isfinite(v) and v > 0 for v in settings.values())):
        raise ValueError("Invalid walking ground correction settings")
    robot = recovery_model("retarget")
    model, data = robot.model, robot.data
    if clip.metadata.get("model_signature") != robot.signature:
        raise ValueError("Walking reference/model mismatch")
    if not np.allclose(np.diff(clip.times), robot.control_dt, rtol=0, atol=1e-12):
        raise ValueError("Walking correction requires the original control clock")
    ground = model.geom("ground").id
    if (model.geom_type[ground] != mujoco.mjtGeom.mjGEOM_PLANE
            or model.geom_bodyid[ground] != 0
            or not np.allclose(model.geom_quat[ground], [1, 0, 0, 0])):
        raise ValueError("Walking correction requires a horizontal world ground plane")
    feet = {model.body(name).id for name in support_bodies}
    poses = np.c_[clip.values["root_position"], clip.values["root_orientation"],
                  clip.values["joint_position"]]
    velocity = np.zeros(model.nv)
    lifts = np.zeros(len(poses))
    nonfoot_bad = height_limited = rise_limited = 0
    for index, pose in enumerate(poses):
        previous = lifts[index - 1] if index else 0.0
        required = 0.0
        if index:
            mujoco.mj_differentiatePos(model, velocity, 1.0, poses[index - 1], pose)
        fractions = np.arange(1, robot.substeps + 1) / robot.substeps if index else [1.0]
        for fraction in fractions:
            data.qpos[:] = poses[index - 1] if index else pose
            if index:
                mujoco.mj_integratePos(model, data.qpos, velocity, float(fraction))
            geometry_forward(model, data)
            for contact in data.contact[:data.ncon]:
                if ground not in (contact.geom1, contact.geom2) or contact.dist >= 0:
                    continue
                other = contact.geom2 if contact.geom1 == ground else contact.geom1
                if model.geom_bodyid[other] not in feet and contact.dist < -0.005:
                    nonfoot_bad += 1
                depth = -float(contact.dist) + settings["clearance_m"]
                required = max(required, (depth - (1 - fraction) * previous) / fraction)
        desired = max(required, previous - settings["release_m_s"] * robot.control_dt, 0.0)
        height_limited += desired > settings["max_lift_m"]
        upper = settings["max_lift_m"]
        if index:
            rise_limit = previous + settings["max_rise_m_s"] * robot.control_dt
            rise_limited += desired > rise_limit
            upper = min(upper, rise_limit)
        lifts[index] = min(desired, upper)
    values = {key: value.copy() for key, value in clip.values.items()}
    values["root_position"][:, 2] += lifts
    values["landmarks"][:, :, 2] += lifts[:, None]
    values["root_velocity"][1:, 2] += np.diff(lifts) / np.diff(clip.times)
    report = {
        "version": version,
        "settings": settings,
        "max_lift_m": float(lifts.max()),
        "mean_lift_m": float(lifts.mean()),
        "rms_lift_m": float(np.sqrt(np.mean(lifts**2))),
        "max_rise_m_s": float(np.maximum(np.diff(lifts), 0).max(initial=0) / robot.control_dt),
        "mean_landmark_error_increase_bound_m": float(lifts.mean() / np.sqrt(3)),
        "height_limited_ticks": int(height_limited),
        "rise_limited_ticks": int(rise_limited),
        "nonfoot_ground_contact_samples": int(nonfoot_bad),
        "causal": True,
        "joint_motion_unchanged": True,
        "horizontal_motion_unchanged": True,
        "clocks_unchanged": True,
        "physics_qualified": False,
    }
    result = MotionClip(clip.times.copy(), values,
                        {**clip.metadata, "ground_correction": report},
                        None if clip.source_times is None else clip.source_times.copy(),
                        None if clip.received_times is None else clip.received_times.copy())
    return result, report
