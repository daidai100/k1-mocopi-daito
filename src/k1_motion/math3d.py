"""Right handed, x forward/y left/z up; all public quaternions are wxyz."""

import numpy as np
from scipy.spatial.transform import Rotation


def unit_quat(q):
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if q.shape[-1] != 4 or not np.isfinite(q).all() or np.any(norm < 1e-8):
        raise ValueError("Expected finite nonzero wxyz quaternion")
    return q / norm


def rotation(q):
    return Rotation.from_quat(np.roll(unit_quat(q), -1, axis=-1))


def quaternion(r):
    return np.roll(r.as_quat(), 1, axis=-1)


def anatomical_rotation(p, names):
    """Derive pelvis axes from landmarks, independent of source bone-local axes."""
    left = p[names.index("left_hip")] - p[names.index("right_hip")]
    up = p[names.index("chest")] - p[names.index("pelvis")]
    forward = np.cross(left, up)
    if min(np.linalg.norm(left), np.linalg.norm(up), np.linalg.norm(forward)) < 1e-6:
        raise ValueError("Degenerate anatomical axes")
    forward /= np.linalg.norm(forward)
    left = np.cross(up, forward)
    left /= np.linalg.norm(left)
    up = np.cross(forward, left)
    return Rotation.from_matrix(np.column_stack([forward, left, up]))


def heading(q):
    forward = rotation(q).apply([1.0, 0.0, 0.0])
    return float(np.arctan2(forward[1], forward[0]))


def backward_velocity(value, previous, dt):
    if previous is None:
        return np.zeros_like(value)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Nonpositive derivative interval")
    return (value - previous) / dt
