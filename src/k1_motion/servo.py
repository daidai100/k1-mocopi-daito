"""Native K1 position/velocity servo, with policy-bound actuation settings.

The robot/data model stays pinned while an explicit controller experiment may
change its desired velocities and PD gains. Legacy position-only callers retain
the original integration path.
"""

import mujoco
import numpy as np


def gains(robot, settings=None):
    settings = settings or {}
    return (
        robot.kp * np.asarray(settings.get("kp_scale", 1.0)),
        robot.kd * np.asarray(settings.get("kd_scale", 1.0)),
    )


def step_pd(robot, targets, velocities=None, *, damping_only=False, settings=None, trace=None):
    if velocities is None and not settings and trace is None:
        return robot.step(targets, damping_only)
    from .actuation import action_settings, command_velocity_limits
    from .actuators import (SAFETY_FIELDS, actuator_parameters, configure_robot_actuators,
                            torque_envelope, speed_guard_torque, safety_sample, accumulate_safety)
    resolved = action_settings(robot, settings)
    configure_robot_actuators(robot, resolved)
    parameters = actuator_parameters(robot, resolved)
    dynamic = 'actuator_profile' in resolved
    operating_speed = command_velocity_limits(robot, resolved)
    targets = np.asarray(targets, dtype=float)
    velocities = np.zeros(22) if velocities is None else np.asarray(velocities, dtype=float)
    if (
        targets.shape != (22,)
        or velocities.shape != (22,)
        or not np.isfinite(targets).all()
        or not np.isfinite(velocities).all()
    ):
        raise ValueError("Invalid K1 position/velocity command")
    targets = np.clip(targets, robot.limits[:, 0], robot.limits[:, 1])
    velocity_cap = operating_speed if dynamic else robot.velocity_limit
    velocities = np.clip(velocities, -velocity_cap, velocity_cap)
    kp, kd = gains(robot, settings)
    if damping_only:
        velocities = np.zeros(22)
    saturation = effort = 0.0
    collision = False
    safety = np.zeros(len(SAFETY_FIELDS))
    for _ in range(robot.substeps):
        torque = 0.0 if damping_only else kp * (targets - robot.data.qpos[7:])
        torque = torque + kd * (velocities - robot.data.qvel[6:])
        available = (torque_envelope(robot.data.qvel[6:], parameters['effort'],
                                    parameters['velocity'], parameters['knee'])
                     if dynamic else robot.effort)
        saturation += np.mean(np.abs(torque) >= available)
        robot.data.ctrl[:] = np.clip(torque, -available, available)
        if resolved.get('operating_speed_guard', False):
            robot.data.ctrl[:] = speed_guard_torque(robot.data.ctrl, robot.data.qvel[6:], operating_speed)
        if trace is not None:
            trace.append(dict(time_s=float(robot.data.time), q=robot.data.qpos[7:].copy(),
                dq=robot.data.qvel[6:].copy(), requested_torque=np.asarray(torque).copy(),
                available_torque=np.asarray(available).copy(), applied_torque=robot.data.ctrl.copy()))
        effort += float(np.mean((robot.data.ctrl / robot.effort) ** 2))
        mujoco.mj_step(robot.model, robot.data)
        accumulate_safety(safety, safety_sample(robot.data.qpos[7:], robot.data.qvel[6:], torque,
                          available, robot.limits, operating_speed, parameters['velocity']), robot.substeps)
        sample_collision = any(
            c.dist <= 0 and 0 not in robot.model.geom_bodyid[[c.geom1, c.geom2]]
            for c in robot.data.contact[: robot.data.ncon]
        )
        collision |= sample_collision
        if trace is not None:
            trace[-1].update(post_q=robot.data.qpos[7:].copy(), post_dq=robot.data.qvel[6:].copy(),
                collision=sample_collision, safety=safety_sample(robot.data.qpos[7:], robot.data.qvel[6:], torque,
                    available, robot.limits, operating_speed, parameters['velocity']))
    if not np.isfinite(robot.data.qpos).all() or not np.isfinite(robot.data.qvel).all():
        raise FloatingPointError("Nonfinite simulated robot state")
    return {
        "effort_saturation": saturation / robot.substeps,
        "effort": effort / robot.substeps,
        "self_collision": collision,
        "safety": dict(zip(SAFETY_FIELDS, safety)),
    }


def step_command(robot, command, settings=None, *, trace=None):
    return step_pd(
        robot, command.targets, command.velocities, damping_only=command.damping_only, settings=settings, trace=trace
    )
