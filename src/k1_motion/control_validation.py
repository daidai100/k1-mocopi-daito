"""Uninterrupted native replay of a saved reference with a pinned controller."""

from dataclasses import replace
import time

import mujoco
import numpy as np

from .evaluation import THRESHOLDS
from .evaluation_panel import _contact_slip
from .math3d import rotation
from .runtime import Controller, Mode
from .servo import step_command

MEASUREMENT_VERSION = "k1-controller-replay-v2-motion-fidelity"
CONTROL_THRESHOLDS = {
    **THRESHOLDS,
    "max_root_velocity_rmse_m_s": 0.3,
    "max_mean_root_orientation_error_rad": 0.35,
    "max_root_orientation_error_rad": 1.0,
    "progress_check_min_displacement_m": 0.5,
    "min_root_progress_ratio": 0.7,
    "max_root_progress_ratio": 1.3,
}


def _low_support_contact_slip(robot):
    """Tangential speed at actual foot/knee/shin contact points, not ankle origins."""
    supports = {robot.model.body(f"{side}_{part}").id for side in ("left", "right")
                for part in ("ankle_roll_link", "knee_pitch_link")}
    speeds = []
    jac = np.zeros((3, robot.model.nv))
    for contact in robot.data.contact[:robot.data.ncon]:
        bodies = robot.model.geom_bodyid[[contact.geom1, contact.geom2]]
        if 0 not in bodies or int(max(bodies)) not in supports or contact.dist > 0:
            continue
        mujoco.mj_jac(robot.model, robot.data, jac, None, contact.pos, int(max(bodies)))
        velocity = jac @ robot.data.qvel
        normal = contact.frame[:3]
        speeds.append(float(np.linalg.norm(velocity-normal*(normal @ velocity))))
    return speeds


def motion_fidelity(clip, qpos, qvel):
    """Evaluation-only world motion; never exposed as controller feedback."""
    count = len(qpos)
    actual = np.asarray(qpos)
    velocity = np.asarray(qvel)[:, :3]
    desired_velocity = clip.values["root_velocity"][1 : count + 1, :3]
    angle = (
        rotation(clip.values["root_orientation"][1 : count + 1]).inv() * rotation(actual[:, 3:7])
    ).magnitude()
    desired_delta = clip.values["root_position"][count, :2] - clip.values["root_position"][0, :2]
    actual_delta = actual[-1, :2] - clip.values["root_position"][0, :2]
    distance = float(np.linalg.norm(desired_delta))
    ratio = (
        float(actual_delta @ desired_delta / distance**2)
        if distance >= CONTROL_THRESHOLDS["progress_check_min_displacement_m"]
        else None
    )
    result = {
        "root_velocity_rmse_m_s": float(
            np.sqrt(np.mean(np.sum((velocity - desired_velocity) ** 2, axis=-1)))
        ),
        "mean_root_orientation_error_rad": float(np.mean(angle)),
        "max_root_orientation_error_rad": float(np.max(angle)),
        "reference_horizontal_displacement_m": distance,
        "actual_horizontal_displacement_m": float(np.linalg.norm(actual_delta)),
        "root_progress_ratio": ratio,
    }
    result["motion_fidelity_passed"] = bool(
        result["root_velocity_rmse_m_s"] <= CONTROL_THRESHOLDS["max_root_velocity_rmse_m_s"]
        and result["mean_root_orientation_error_rad"]
        <= CONTROL_THRESHOLDS["max_mean_root_orientation_error_rad"]
        and result["max_root_orientation_error_rad"] <= CONTROL_THRESHOLDS["max_root_orientation_error_rad"]
        and (
            ratio is None
            or CONTROL_THRESHOLDS["min_root_progress_ratio"]
            <= ratio
            <= CONTROL_THRESHOLDS["max_root_progress_ratio"]
        )
    )
    return result


def replay_clip(robot, policy, clip, trace_path=None, *, authority_trace=False):
    from .low_pose_contract import low_pose_accepted, minimum_tracking_height, minimum_tracking_upright
    if clip.metadata["model_signature"] != robot.signature:
        raise ValueError("Reference/model mismatch")
    if not np.all(clip.values["valid"]):
        raise ValueError("Dynamic replay requires zero invalid reference ticks")
    if not np.allclose(np.diff(clip.times), robot.control_dt, atol=1e-9, rtol=0):
        raise ValueError("Dynamic replay requires controller-rate references")
    reference_scale = getattr(policy, "metadata", {}).get("reference_scale")
    if reference_scale is not None:
        from .reference_scale import scale_clip
        clip = scale_clip(clip, robot, reference_scale)
    elif clip.metadata.get("reference_scale") is not None:
        raise ValueError("Scaled reference requires matching policy scaling metadata")
    low_support = low_pose_accepted(clip.metadata) if (clip.metadata.get("family") == "kneel"
                       or "low_pose_reference_audit" in clip.metadata) else False
    if (clip.metadata.get("family") == "kneel" or "low_pose_reference_audit" in clip.metadata) and not low_support:
        raise ValueError("Low support replay requires an audited task")
    minimum_heights = minimum_tracking_height(clip.values["root_position"][:, 2], low_support=low_support,
                                              standing_minimum=THRESHOLDS["min_height_m"])
    minimum_upright = minimum_tracking_upright(rotation(clip.values["root_orientation"]).apply([0,0,1])[:,2],
                                               low_support=low_support,
                                               standing_minimum=THRESHOLDS["min_upright_cos"])
    from .actuators import SAFETY_FIELDS, actuator_contract, configure_robot_actuators
    controller = Controller(robot, policy)
    configure_robot_actuators(robot, controller.action_settings)
    robot.reset(clip.frame(0))
    buffered = controller.preview_buffer is not None
    arrival_origin = float(clip.times[0])
    def packet(index):
        original = clip.frame(index)
        return replace(original, received_time=original.received_time-arrival_origin)
    initial = packet(0) if buffered else replace(clip.frame(0), received_time=0.0)
    robot.data.qvel[:3] = initial.root_velocity[:3]
    robot.data.qvel[3:6] = rotation(initial.root_orientation).inv().apply(initial.root_velocity[3:])
    robot.data.qvel[6:] = initial.joint_velocity
    mujoco.mj_forward(robot.model, robot.data)
    controller.calibrate(initial, robot.state(0), True, 0, playback_time=float(clip.times[0]))
    pushed = 0
    if buffered:
        for pushed in range(1, min(16, len(clip.times))):
            controller.set_reference(packet(pushed), playback_time=float(clip.times[pushed]))
        if pushed == len(clip.times)-1:
            controller.finish_reference()
    delay = .3 if buffered else 0.
    controller.arm(robot.state(delay), delay)
    qerrors, bodyerrors, saturation, velocity_saturation, slips, latency = [], [], [], [], [], []
    collision_ticks = 0
    trace, velocities, targets, target_velocities = [], [], [], []
    fallen = stopped = False
    world_errors, substep_safety = [], []
    authority_commands, authority_substeps = [], []
    for index in range(len(clip.times) - 1):
        now = delay + index * robot.control_dt
        if buffered:
            available = min(index+15, len(clip.times)-1)
            for next_index in range(pushed+1, available+1):
                controller.set_reference(packet(next_index), playback_time=float(clip.times[next_index]))
                pushed = next_index
                if pushed == len(clip.times)-1:
                    controller.finish_reference()
        elif index:
            controller.set_reference(replace(clip.frame(index), received_time=now))
        start = time.perf_counter()
        command = controller.tick(robot.state(now), now)
        latency.append(time.perf_counter() - start)
        if authority_trace:
            authority_commands.append({k: v.copy() for k, v in controller.last_command.items()})
            metrics = step_command(robot, command, controller.action_settings, trace=authority_substeps)
        else:
            metrics = step_command(robot, command, controller.action_settings)
        mujoco.mj_forward(robot.model, robot.data)
        actual = clip.frame(index + 1)
        qerrors.append(np.mean((robot.data.qpos[7:] - actual.joint_position) ** 2))
        relative = (robot.landmarks() - robot.data.qpos[:3]) - (actual.landmarks - actual.root_position)
        bodyerrors.append(np.mean(np.sum(relative**2, axis=-1)))
        world_errors.append(np.linalg.norm(robot.landmarks()-actual.landmarks, axis=-1))
        substep_safety.append(metrics['safety'])
        saturation.append(metrics["effort_saturation"])
        velocity_saturation.append(np.mean(abs(robot.data.qvel[6:]) > robot.velocity_limit))
        slips.extend(_low_support_contact_slip(robot) if low_support else _contact_slip(robot))
        collision_ticks += bool(metrics["self_collision"] or robot.contact_metrics()[1])
        trace.append(robot.data.qpos.copy())
        velocities.append(robot.data.qvel.copy())
        targets.append(command.targets)
        target_velocities.append(np.zeros(22) if command.velocities is None else command.velocities)
        fallen = bool(
            robot.data.qpos[2] < minimum_heights[index+1]
            or rotation(robot.data.qpos[3:7]).apply([0, 0, 1])[2] < minimum_upright[index+1]
        )
        stopped = command.mode != Mode.ACTIVE
        if fallen or stopped:
            break
    result = {
        "reference_scale": reference_scale,
        "measurement_version": "k1-low-support-controller-replay-v1" if low_support else MEASUREMENT_VERSION,
        "completed": not fallen and not stopped,
        "fell": fallen,
        "controller_stopped": stopped,
        "controller_reason": controller.reason,
        "duration_s": float(clip.times[-1] - clip.times[0]),
        "simulated_s": float(robot.data.time),
        "joint_rmse_rad": float(np.sqrt(np.mean(qerrors))),
        "relative_body_rmse_m": float(np.sqrt(np.mean(bodyerrors))),
        "world_body_rmse_m": float(np.sqrt(np.mean(np.square(world_errors)))),
        "world_body_p95_m": float(np.percentile(world_errors, 95)),
        "world_body_max_error_m": float(np.max(world_errors)),
        "world_body_per_point_rmse_m": np.sqrt(np.mean(np.square(world_errors), axis=0)).tolist(),
        "full_reference_duration_world_score": float(np.sum(1/(1+np.square(world_errors)/.15**2))
            / ((len(clip.times)-1)*len(world_errors[0]))),
        "effort_saturation_fraction": float(np.mean(saturation)),
        "velocity_saturation_fraction": float(np.mean(velocity_saturation)),
        "contact_point_slip_mean_m_s": float(np.mean(slips)) if slips else 0.0,
        "self_collision_ticks": collision_ticks,
        "command_p95_ms": float(np.percentile(latency, 95) * 1000),
        "command_p99_ms": float(np.percentile(latency, 99) * 1000),
        "resets_during_trial": 0,
        "initialization": "reference_pose_and_velocity",
        "hardware_verified": False,
        "low_support_task": low_support,
    }
    result['actuator_safety'] = {
        key: float((np.max if key.endswith(('max_ratio','max_error')) else np.mean)
                   ([step[key] for step in substep_safety])) for key in SAFETY_FIELDS}
    result['actuator_safety'].update(sample_period_s=float(robot.model.opt.timestep),
        sample_count=len(trace)*robot.substeps, contract=actuator_contract(robot, controller.action_settings))
    result['absolute_motion_v1'] = dict(version='world-position-safety-v1',
        world_rmse_limit_m=.15, world_p95_limit_m=.30,
        full_duration_completed=result['completed'],
        clean=bool(result['completed'] and collision_ticks == 0
                   and result['world_body_rmse_m'] <= .15 and result['world_body_p95_m'] <= .30
                   and result['actuator_safety']['operating_speed_fraction'] == 0
                   and result['actuator_safety']['joint_limit_fraction'] == 0))
    if buffered:
        result['preview'] = dict(playback_delay_s=.3,
            preview_horizon_s=controller.preview_buffer.horizon_s,
            captured_clock_preserved=True, explicit_end_of_stream=True,
            scored_ticks=len(trace), reference_ticks=len(clip.times)-1,
            feedback_and_actuator_delay_s=0.)
    result["pose_balance_collision_passed"] = bool(
        result["completed"]
        and collision_ticks == 0
        and result["joint_rmse_rad"] <= THRESHOLDS["max_joint_rmse_rad"]
        and result["relative_body_rmse_m"] <= THRESHOLDS["max_relative_body_rmse_m"]
        and result["effort_saturation_fraction"] <= THRESHOLDS["max_effort_saturation_fraction"]
        and result["velocity_saturation_fraction"] == 0
        and result["contact_point_slip_mean_m_s"] <= THRESHOLDS["max_contact_slip_m_s"]
        and result["command_p95_ms"] <= THRESHOLDS["command_p95_ms"]
    )
    result.update(motion_fidelity(clip, trace, velocities))
    result["clean_success"] = result["pose_balance_collision_passed"] and result["motion_fidelity_passed"]
    if authority_trace and authority_commands and authority_commands[0]:
        actions = np.asarray([r['action'][10:] for r in authority_commands])
        requested = np.asarray([r['requested_torque'][10:] for r in authority_substeps])
        available = np.asarray([r['available_torque'][10:] for r in authority_substeps])
        result['command_authority'] = dict(control_ticks=len(actions),
            leg_action_bound_fraction=float(np.mean(np.abs(actions) > .95)),
            any_leg_action_bound_ticks=int(np.sum(np.any(np.abs(actions) > .95, axis=-1))),
            leg_torque_saturation_fraction=float(np.mean(np.abs(requested) > available)),
            position_guard_ticks=int(sum(np.any(np.abs(r['executed_target']-r['arm_target']) > 1e-6)
                                         for r in authority_commands)),
            torque_clock='pre-integration requested/applied torque; post_q/post_dq/safety after same substep')
    from .trajectory_metrics import trajectory_fidelity
    result["trajectory_v3"] = trajectory_fidelity(
        clip, trace, completed=result["completed"], old_clean_success=result["clean_success"])
    if trace_path is not None:
        authority = {}
        if authority_trace and authority_commands and authority_commands[0]:
            authority.update({f'command_{k}': np.asarray([r[k] for r in authority_commands])
                              for k in authority_commands[0]})
            authority.update({f'substep_{k}': np.asarray([r[k] for r in authority_substeps])
                              for k in authority_substeps[0]})
        np.savez_compressed(
            trace_path,
            qpos=trace,
            qvel=velocities,
            targets=targets,
            target_velocities=target_velocities,
            control_dt=robot.control_dt,
            world_body_error_m=world_errors,
            **authority,
        )
    return result
