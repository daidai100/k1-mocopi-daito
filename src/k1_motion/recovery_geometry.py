"""Pose-only MuJoCo work for offline recovery, never for controller simulation."""

from functools import lru_cache

import mujoco
import numpy as np

from .robot import K1Model


def geometry_forward(model, data):
    """Refresh rigid-body poses, Jacobian inputs and contacts, without dynamics.

    Recovery reads site/body poses and geometric contacts only. It does not read
    constraint forces, accelerations or sensors. Keep mj_forward/mj_step in all
    controller code; this is not a replacement for a physics step.
    """
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    mujoco.mj_collision(model, data)


def has_self_penetration(model, data, tolerance):
    penetrating = data.contact.dist[:data.ncon] < -tolerance
    if not np.any(penetrating):
        return False
    bodies = model.geom_bodyid[data.contact.geom[:data.ncon][penetrating]]
    return bool(np.any(np.all(bodies != 0, axis=1)))


@lru_cache(maxsize=2)
def _worker_model(purpose):
    # Distinct immutable models keep the independent auditor separate from IK.
    if purpose not in {"retarget", "audit"}:
        raise ValueError("Unknown recovery model purpose")
    return K1Model()


def recovery_model(purpose):
    """Reuse compilation per process, resetting all simulator state per motion."""
    robot = _worker_model(purpose)
    robot.reset()
    return robot
