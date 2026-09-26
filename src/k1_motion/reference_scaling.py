"""Offline playback stretching of an already retargeted K1 trajectory."""

import copy

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .contracts import MotionClip
from .reference_velocity_clock import playback_derivatives


def stretch_motion_clip(clip, robot, factor):
    """Interpolate K1 poses on a slower 50 Hz clock, then rebuild all derived fields.

    This offline operation uses the next saved pose. It must not be used as a
    live mocopi processor, and its output still needs independent path audit.
    """
    if not np.isfinite(factor) or factor < 1:
        raise ValueError("Offline motion stretch factor must be finite and at least one")
    if clip.metadata.get("model_signature") != robot.signature:
        raise ValueError("Motion/model signature mismatch")
    if not np.allclose(np.diff(clip.times), robot.control_dt, atol=1e-12, rtol=0):
        raise ValueError("Offline stretch requires the K1 control clock")
    old_times = clip.times - clip.times[0]
    duration = old_times[-1] * factor
    count = int(np.floor(np.nextafter(duration / robot.control_dt, np.inf))) + 1
    new_times = np.arange(count) * robot.control_dt
    phase = np.minimum(new_times / factor, old_times[-1])

    def linear(key):
        value = clip.values[key]
        flat = value.reshape(len(old_times), -1)
        result = np.stack([np.interp(phase, old_times, flat[:, i])
                           for i in range(flat.shape[1])], axis=1)
        return result.reshape((len(phase), *value.shape[1:]))

    orientation = clip.values["root_orientation"]
    rotations = Rotation.from_quat(orientation[:, [1, 2, 3, 0]])
    xyzw = Slerp(old_times, rotations)(phase).as_quat()
    wxyz = xyzw[:, [3, 0, 1, 2]]
    root = linear("root_position")
    joints = linear("joint_position")
    data = mujoco.MjData(robot.model)
    landmarks = np.empty((len(phase), 17, 3))
    for i in range(len(phase)):
        data.qpos[:] = np.r_[root[i], wxyz[i], joints[i]]
        mujoco.mj_forward(robot.model, data)
        landmarks[i] = robot.landmarks(data)
    previous = np.searchsorted(old_times, phase, side="right") - 1
    following = np.minimum(previous + 1, len(old_times) - 1)
    values = {
        "root_position": root,
        "root_orientation": wxyz,
        "root_velocity": np.zeros((len(phase), 6)),
        "joint_position": joints,
        "joint_velocity": np.zeros((len(phase), 22)),
        "landmarks": landmarks,
        "contacts": clip.values["contacts"][previous],
        "contact_confidence": clip.values["contact_confidence"][previous],
        "valid": np.minimum(clip.values["valid"][previous], clip.values["valid"][following]),
    }
    metadata = copy.deepcopy(clip.metadata)
    metadata["retarget_version"] += f"+offline-stretch-{factor:g}-v1"
    metadata["offline_stretch"] = {
        "version": "k1-offline-pose-stretch-v1", "factor": factor,
        "source_duration_s": float(old_times[-1]),
        "output_duration_s": float(new_times[-1]),
        "method": "linear K1 root/joint pose, quaternion slerp, model-forward landmarks",
        "causal_live_execution": False,
        "received_clock": "offline artifact available at playback time",
        "physics_qualified": False,
        "training_eligible": False,
    }
    metadata["sampling"] = {**metadata.get("sampling", {}),
                            "sample_clock": "offline_stretched_K1_playback_seconds",
                            "received_clock": "offline artifact available at playback time",
                            "velocity_clock": "sample_clock"}
    source_times = (None if clip.source_times is None else
                    np.interp(phase, old_times, clip.source_times))
    received_times = None if source_times is None else new_times
    result = MotionClip(new_times, values, metadata, source_times, received_times)
    joint_velocity, root_velocity = playback_derivatives(result)
    values.update(joint_velocity=joint_velocity, root_velocity=root_velocity)
    return MotionClip(new_times, values, metadata, source_times, received_times)
