"""Versioned contracts shared by adapters, simulation, and deployment inference."""

from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np

from .math3d import unit_quat

LANDMARKS = (
    "pelvis",
    "chest",
    "head",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "left_hip",
    "left_knee",
    "left_ankle",
    "left_toe",
    "right_hip",
    "right_knee",
    "right_ankle",
    "right_toe",
)
PARENTS = (-1, 0, 1, 1, 3, 4, 1, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15)
JOINT_NAMES = (
    "aahead_yaw_joint",
    "aahead_pitch_joint",
    "aaleft_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_elbow_pitch_joint",
    "left_elbow_yaw_joint",
    "aaright_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_elbow_pitch_joint",
    "right_elbow_yaw_joint",
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_pitch_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_pitch_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)
CONVENTIONS = {
    "version": 1,
    "world_axes": "right_handed_x_forward_y_left_z_up",
    "length": "metres",
    "angle": "radians",
    "quaternion": "wxyz",
    "poses": "world",
    "root_velocity": "world_linear_and_world_angular",
    "joint_velocity": "causal_backward_difference_radians_per_second",
    "source_time": "seconds_in_source_clock",
    "receive_time": "local_monotonic_seconds",
}


def array(value, shape, name):
    a = np.array(value, dtype=np.float64, copy=True)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f"{name}: expected {shape} finite values, got {a.shape}")
    a.setflags(write=False)
    return a


@dataclass(frozen=True)
class HumanFrame:
    source_time: float
    received_time: float
    frame_number: int
    positions: np.ndarray
    orientations: np.ndarray
    source_id: str
    session: int = 0

    def __post_init__(self):
        if not np.isfinite([self.source_time, self.received_time]).all():
            raise ValueError("Invalid frame clocks")
        object.__setattr__(self, "positions", array(self.positions, (len(LANDMARKS), 3), "positions"))
        q = unit_quat(array(self.orientations, (len(LANDMARKS), 4), "orientations"))
        q.setflags(write=False)
        object.__setattr__(self, "orientations", q)


@dataclass(frozen=True)
class Reference:
    source_time: float
    received_time: float
    root_position: np.ndarray
    root_orientation: np.ndarray
    root_velocity: np.ndarray
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    landmarks: np.ndarray
    contacts: np.ndarray
    contact_confidence: np.ndarray
    valid: bool = True
    session: int = 0

    def __post_init__(self):
        if not np.isfinite([self.source_time, self.received_time]).all():
            raise ValueError("Invalid reference clocks")
        for name, shape in (
            ("root_position", (3,)),
            ("root_velocity", (6,)),
            ("joint_position", (22,)),
            ("joint_velocity", (22,)),
            ("landmarks", (len(LANDMARKS), 3)),
            ("contacts", (2,)),
            ("contact_confidence", (2,)),
        ):
            object.__setattr__(self, name, array(getattr(self, name), shape, name))
        object.__setattr__(
            self, "root_orientation", array(unit_quat(self.root_orientation), (4,), "root_orientation")
        )
        if np.any((self.contacts != 0) & (self.contacts != 1)):
            raise ValueError("Contacts must be binary")
        if np.any((self.contact_confidence < 0) | (self.contact_confidence > 1)):
            raise ValueError("Contact confidence outside [0, 1]")


@dataclass(frozen=True)
class RobotState:
    """Feedback; planar policies additionally require world odometry."""

    time: float
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    orientation: np.ndarray
    angular_velocity: np.ndarray  # body-frame gyro, rad/s
    healthy: bool = True
    root_position: np.ndarray | None = None
    linear_velocity: np.ndarray | None = None

    def __post_init__(self):
        if not np.isfinite(self.time):
            raise ValueError("Invalid robot time")
        for name, shape in (("joint_position", (22,)), ("joint_velocity", (22,)), ("angular_velocity", (3,))):
            object.__setattr__(self, name, array(getattr(self, name), shape, name))
        object.__setattr__(self, "orientation", array(unit_quat(self.orientation), (4,), "orientation"))
        for name in ("root_position", "linear_velocity"):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, array(getattr(self, name), (3,), name))


REFERENCE_ARRAYS = (
    "root_position",
    "root_orientation",
    "root_velocity",
    "joint_position",
    "joint_velocity",
    "landmarks",
    "contacts",
    "contact_confidence",
    "valid",
)


@dataclass
class MotionClip:
    times: np.ndarray
    values: dict
    metadata: dict = field(default_factory=dict)
    source_times: np.ndarray | None = None
    received_times: np.ndarray | None = None

    def __post_init__(self):
        self.times = np.asarray(self.times, dtype=float)
        if (
            self.times.ndim != 1
            or len(self.times) < 2
            or not np.isfinite(self.times).all()
            or np.any(np.diff(self.times) <= 0)
        ):
            raise ValueError("Motion requires at least two strictly increasing finite timestamps")
        if self.metadata.get("joint_names") != list(JOINT_NAMES):
            raise ValueError("Motion joint order does not match the K1 contract")
        if self.metadata.get("conventions") != CONVENTIONS:
            raise ValueError("Motion convention/version mismatch")
        shapes = ((3,), (4,), (6,), (22,), (22,), (len(LANDMARKS), 3), (2,), (2,), ())
        for key, shape in zip(REFERENCE_ARRAYS, shapes):
            self.values[key] = array(self.values[key], (len(self.times), *shape), key)
        norms = np.linalg.norm(self.values["root_orientation"], axis=-1)
        if not np.allclose(norms, 1.0, atol=1e-5):
            raise ValueError("Motion contains non-unit quaternions")
        if np.any((self.values["contacts"] != 0) & (self.values["contacts"] != 1)):
            raise ValueError("Invalid motion contacts")
        if np.any((self.values["valid"] != 0) & (self.values["valid"] != 1)):
            raise ValueError("Invalid motion validity mask")
        if (self.source_times is None) != (self.received_times is None):
            raise ValueError("Sampled motion must retain both source and receive clocks")
        if self.source_times is not None:
            self.source_times = array(self.source_times, self.times.shape, "source_times")
            self.received_times = array(self.received_times, self.times.shape, "received_times")
            if np.any(np.diff(self.source_times) < 0) or np.any(self.received_times > self.times):
                raise ValueError("Sampled motion uses a future or out-of-order human frame")

    @classmethod
    def from_references(cls, frames, metadata, *, sample_times=None):
        metadata = {**metadata, "joint_names": list(JOINT_NAMES), "conventions": CONVENTIONS}
        return cls(
            np.array([f.source_time for f in frames]) if sample_times is None else sample_times,
            {key: np.stack([getattr(f, key) for f in frames]) for key in REFERENCE_ARRAYS},
            metadata,
            None if sample_times is None else np.array([f.source_time for f in frames]),
            None if sample_times is None else np.array([f.received_time for f in frames]),
        )

    def frame(self, index, received_time=None):
        t = float(self.times[index])
        return Reference(
            t if self.source_times is None else float(self.source_times[index]),
            (t if self.received_times is None else float(self.received_times[index]))
            if received_time is None
            else received_time,
            **{key: value[index] for key, value in self.values.items()},
        )

    def causal_index(self, source_time):
        if source_time < self.times[0]:
            raise ValueError("Reference has not started")
        return min(len(self.times) - 1, int(np.searchsorted(self.times, source_time, side="right") - 1))

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".partial")
        clocks = (
            {}
            if self.source_times is None
            else {"source_times": self.source_times, "received_times": self.received_times}
        )
        with tmp.open("wb") as f:
            np.savez_compressed(
                f, times=self.times, metadata=json.dumps(self.metadata), **self.values, **clocks
            )
        tmp.replace(path)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            return cls(
                data["times"],
                {key: data[key] for key in REFERENCE_ARRAYS},
                json.loads(str(data["metadata"])),
                data["source_times"] if "source_times" in data else None,
                data["received_times"] if "received_times" in data else None,
            )
