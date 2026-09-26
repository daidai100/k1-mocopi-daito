"""Explicit neutral/floor/facing calibration; no future-frame floor estimate."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from .contracts import HumanFrame
from .math3d import heading, quaternion, rotation


def motion_scale_contract(profile):
    """Extra root-motion scaling after the neutral human-to-K1 size match."""
    if profile is None:
        return None
    if profile != "robot-fit-70-v1":
        raise ValueError(f"Unknown motion scale profile: {profile}")
    return {
        "version": "k1-root-motion-scale-v1",
        "profile": profile,
        "horizontal_travel_scale": 0.7,
        "upward_excursion_scale": 0.7,
        "downward_excursion_scale": 1.0,
        "clock_scale": 1.25,
        "scope": "Candidate spatial scaling after neutral-height calibration; offline K1 playback stretch",
    }


@dataclass(frozen=True)
class Calibration:
    origin: np.ndarray
    yaw: float
    scale: float
    robot_origin: np.ndarray
    floor: float
    session: int
    neutral_root: np.ndarray | None = None
    motion_profile: str | None = None

    @classmethod
    def from_neutral(cls, frame, robot, floor=None, robot_origin=None, *, motion_profile=None):
        motion_scale_contract(motion_profile)
        p = frame.positions
        floor = float(min(p[12, 2], p[16, 2]) - 0.02) if floor is None else float(floor)
        height = p[0, 2] - floor
        if not np.isfinite(height) or not 0.45 < height < 1.5:
            raise ValueError("Calibration needs an upright neutral human pose and correct metres/z-up")
        origin = p[0].copy()
        origin[2] = floor
        target = np.zeros(3) if robot_origin is None else np.asarray(robot_origin, float).copy()
        target[2] = 0.0
        return cls(
            origin,
            heading(frame.orientations[0]),
            robot.neutral_qpos[2] / height,
            target,
            floor,
            frame.session,
            target + np.array([0.0, 0.0, robot.neutral_qpos[2]]),
            motion_profile,
        )

    def apply(self, frame):
        if frame.session != self.session:
            raise ValueError("Input session changed; pause and recalibrate")
        r = Rotation.from_euler("z", -self.yaw)
        p = r.apply(frame.positions - self.origin) * self.scale + self.robot_origin
        contract = motion_scale_contract(self.motion_profile)
        if contract is not None:
            travel = p[0] - self.neutral_root
            shift = np.array([
                (contract["horizontal_travel_scale"] - 1.0) * travel[0],
                (contract["horizontal_travel_scale"] - 1.0) * travel[1],
                (contract["upward_excursion_scale"] - 1.0) * max(travel[2], 0.0),
            ])
            p += shift
        q = quaternion(r * rotation(frame.orientations))
        return HumanFrame(
            frame.source_time, frame.received_time, frame.frame_number, p, q, frame.source_id, frame.session
        )

    def metadata(self):
        metadata = {
            "origin": self.origin.tolist(),
            "yaw": self.yaw,
            "scale": self.scale,
            "robot_origin": self.robot_origin.tolist(),
            "floor": self.floor,
            "session": self.session,
            "method": "explicit_first_neutral_pose_no_future_floor_scan",
        }
        if self.motion_profile is not None:
            metadata["motion_scale"] = motion_scale_contract(self.motion_profile)
        return metadata
