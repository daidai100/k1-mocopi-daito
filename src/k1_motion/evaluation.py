"""Closed-loop evaluation with fixed thresholds, retained failures, and no fall resets."""

from collections import defaultdict
from dataclasses import replace
import json
from pathlib import Path
import time

import mujoco
import numpy as np
import torch

from .contracts import MotionClip
from .learning import Policy
from .math3d import rotation
from .robot import K1Model
from .servo import step_command
from .runtime import Controller, Mode
from .actuation import action_settings

THRESHOLDS = {
    "min_height_m": 0.22,
    "min_upright_cos": 0.2,
    "max_joint_rmse_rad": 0.35,
    "max_relative_body_rmse_m": 0.15,
    "max_effort_saturation_fraction": 0.05,
    "max_contact_slip_m_s": 0.2,
    "min_trials_per_family": 20,
    "min_families": 8,
    "min_family_completion": 0.95,
    "command_p95_ms": 40.0,
}


def evaluate(library, output, policy_path=None, split="test", limit=None):
    from .evaluation_panel import _contact_slip
    torch.set_num_threads(1)
    directory, output = Path(library), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "thresholds.json").write_text(json.dumps(THRESHOLDS, indent=2) + "\n")
    rows = [json.loads(line) for line in (directory / "index.jsonl").read_text().splitlines()]
    rows = [r for r in rows if r["split"] == split]
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError(f"No {split} clips")
    robot = K1Model()
    policy = Policy(policy_path, robot.signature) if policy_path else None
    if policy and policy.metadata.get("reference_scale") is not None:
        raise ValueError("Legacy evaluator cannot score scaled policies; use control_validation.replay_clip")
    if policy and split in ("validation", "test"):
        overlap = set(policy.metadata["train_parents"]) & {r["capture_group"] for r in rows}
        if overlap:
            raise ValueError(f"Train/evaluation recording overlap: {overlap}")
    trials = []
    for row in rows:
        if not row["kinematics_accepted"]:
            trials.append(
                {
                    "id": row["id"],
                    "source_motion_id": row["source_motion_id"],
                    "capture_group": row["capture_group"],
                    "family": row["family"],
                    "split": split,
                    "completed": False,
                    "tracking_passed": False,
                    "fell": False,
                    "reference_rejected": True,
                    "rejection_counts": row.get("rejection_counts", {}),
                    "error": row.get("error"),
                    "simulated_s": 0.0,
                    "resets_during_trial": 0,
                }
            )
            continue
        clip = MotionClip.load(directory / row["reference_path"])
        if clip.metadata["model_signature"] != robot.signature:
            raise ValueError("Evaluation reference uses different robot contract")
        robot.reset(clip.frame(0))
        controller = Controller(robot, policy)
        initial = replace(clip.frame(0), received_time=0.0)
        robot.data.qvel[:3] = initial.root_velocity[:3]
        robot.data.qvel[3:6] = rotation(initial.root_orientation).inv().apply(initial.root_velocity[3:])
        robot.data.qvel[6:] = initial.joint_velocity
        mujoco.mj_forward(robot.model, robot.data)
        # Reference-state initialization at trial start is counted and disclosed.
        controller.calibrate(initial, robot.state(0.0), True, 0.0)
        controller.arm(robot.state(0.0), 0.0)
        trace, elapsed, errors, body_errors, sats, slips, velocity_sats, collisions = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        last_index = 0
        fallen, stopped = False, False
        start = clip.times[0]
        duration = clip.times[-1] - start
        for tick in range(max(1, round(duration / robot.control_dt))):
            now = tick * robot.control_dt
            index = clip.causal_index(start + now)
            if index != last_index:
                controller.set_reference(replace(clip.frame(index), received_time=now))
                last_index = index
            contacts, collision = robot.contact_metrics()
            stamp = time.perf_counter()
            command = controller.tick(robot.state(now), now, supported=bool(contacts.all()))
            elapsed.append(time.perf_counter() - stamp)
            metrics = step_command(robot, command, controller.action_settings)
            mujoco.mj_forward(robot.model, robot.data)
            landmarks = robot.landmarks()
            actual = clip.frame(clip.causal_index(start + now + robot.control_dt))
            err = np.mean((robot.data.qpos[7:] - actual.joint_position) ** 2)
            relative = (landmarks - robot.data.qpos[:3]) - (actual.landmarks - actual.root_position)
            body_err = np.mean(np.sum(relative ** 2, axis=-1))
            errors.append(err)
            body_errors.append(body_err)
            sats.append(metrics["effort_saturation"])
            velocity_sats.append(float(np.mean(np.abs(robot.data.qvel[6:]) > robot.velocity_limit)))
            slips.extend(_contact_slip(robot))
            collisions.append(metrics["self_collision"] or collision or robot.contact_metrics()[1])
            trace.append(robot.data.qpos.copy())
            upright = rotation(robot.data.qpos[3:7]).apply([0, 0, 1])[2]
            fallen = bool(
                robot.data.qpos[2] < THRESHOLDS["min_height_m"] or upright < THRESHOLDS["min_upright_cos"]
            )
            stopped = command.mode in (Mode.FAULT, Mode.STOPPED) or command.mode == Mode.PAUSED
            if fallen or stopped:
                break
        result = {
            "id": row["id"],
            "source_motion_id": row["source_motion_id"],
            "capture_group": row["capture_group"],
            "family": row["family"],
            "split": split,
            "reference_rejected": False,
            "duration_s": float(duration),
            "simulated_s": len(trace) * robot.control_dt,
            "completed": not fallen and not stopped,
            "fell": fallen,
            "controller_stopped": stopped,
            "controller_reason": controller.reason,
            "resets_during_trial": 0,
            "initialization": "reference_state",
            "measurement_version": "substep-collision-euclidean-body-contact-point-slip-v2",
            "joint_rmse_rad": float(np.sqrt(np.mean(errors))),
            "relative_body_rmse_m": float(np.sqrt(np.mean(body_errors))),
            "effort_saturation_fraction": float(np.mean(sats)),
            "velocity_saturation_fraction": float(np.mean(velocity_sats)),
            "foot_slip_mean_m_s": float(np.mean(slips)) if slips else 0.0,
            "self_collision_frames": int(np.count_nonzero(collisions)),
            "command_p95_ms": float(np.percentile(elapsed, 95) * 1000),
            "command_p99_ms": float(np.percentile(elapsed, 99) * 1000),
        }
        result["tracking_passed"] = (
            result["completed"]
            and result["joint_rmse_rad"] <= THRESHOLDS["max_joint_rmse_rad"]
            and result["relative_body_rmse_m"] <= THRESHOLDS["max_relative_body_rmse_m"]
            and result["effort_saturation_fraction"] <= THRESHOLDS["max_effort_saturation_fraction"]
            and result["velocity_saturation_fraction"] == 0
            and result["foot_slip_mean_m_s"] <= THRESHOLDS["max_contact_slip_m_s"]
            and result["self_collision_frames"] == 0
            and result["command_p95_ms"] <= THRESHOLDS["command_p95_ms"]
        )
        np.savez_compressed(output / f"{row['id']}.npz", qpos=np.array(trace), control_dt=robot.control_dt)
        trials.append(result)
    families = defaultdict(list)
    for trial in trials:
        families[trial["family"]].append(trial)
    summaries = {
        name: {
            "trials": len(items),
            "completion": sum(t["completed"] for t in items) / len(items),
            "tracking_pass_rate": sum(t["tracking_passed"] for t in items) / len(items),
            "falls": sum(t["fell"] for t in items),
        }
        for name, items in families.items()
    }
    covered = len(families) >= THRESHOLDS["min_families"] and all(
        len(v) >= THRESHOLDS["min_trials_per_family"] for v in families.values()
    )
    report = {
        "policy": str(policy_path) if policy_path else "untrained_zero_residual_with_imu_prior",
        "action_settings": action_settings(robot, policy.metadata.get("action_settings") if policy else None),
        "backend": f"mujoco-{mujoco.__version__}",
        "split": split,
        "trials": trials,
        "families": summaries,
        "coverage_target_met": covered,
        "reference_rejections": sum(t["reference_rejected"] for t in trials),
        "behaviorally_accepted": covered
        and all(s["tracking_pass_rate"] >= THRESHOLDS["min_family_completion"] for s in summaries.values()),
        "live_mocopi_tested": False,
        "hardware_verified": False,
        "scope": "Closed-loop references, CPU exported student; not raw sensor end-to-end latency.",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def standing_bouts(output, seconds=300.0, policy_path=None):
    """Two uninterrupted bouts with explicit supported pause/calibration, no teleport."""
    from .evaluation_panel import _contact_slip

    if not np.isfinite(seconds) or seconds < 0.02:
        raise ValueError("Bout duration must be finite and at least one control interval")
    torch.set_num_threads(1)
    robot, stats = K1Model(), []
    policy = Policy(policy_path, robot.signature) if policy_path else None
    controller = Controller(robot, policy)
    controller.calibrate(robot.neutral_reference(), robot.state(0.0), True, 0.0)
    controller.arm(robot.state(0.0), 0.0)
    now, falls, max_jump, previous, p95 = 0.0, 0, 0.0, robot.neutral.copy(), []
    max_step_fraction = 0.0
    trace, phases, qerr, bodyerr, saturation, slips, collisions = [], [], [], [], [], [], []
    failure = None

    def step(expected, phase):
        nonlocal now, falls, max_jump, previous, failure, max_step_fraction
        stamp = time.perf_counter()
        contacts, _ = robot.contact_metrics()
        command = controller.tick(robot.state(now), now, supported=bool(contacts.all()))
        p95.append(time.perf_counter() - stamp)
        metrics = step_command(robot, command, controller.action_settings)
        mujoco.mj_forward(robot.model, robot.data)
        max_jump = max(max_jump, float(np.max(np.abs(command.targets - previous))))
        max_step_fraction = max(
            max_step_fraction,
            float(np.max(np.abs(command.targets - previous) / (controller.command_velocity_limits * robot.control_dt))),
        )
        previous = command.targets
        upright = float(rotation(robot.data.qpos[3:7]).apply([0, 0, 1])[2])
        fallen = robot.data.qpos[2] < THRESHOLDS["min_height_m"] or upright < THRESHOLDS["min_upright_cos"]
        falls += int(fallen)
        trace.append(robot.data.qpos.copy())
        phases.append(phase)
        qerr.append(np.mean((robot.data.qpos[7:] - robot.neutral) ** 2))
        relative = (robot.landmarks() - robot.data.qpos[:3]) - (
            robot.neutral_landmarks - robot.neutral_qpos[:3]
        )
        bodyerr.append(np.mean(np.sum(relative**2, axis=-1)))
        saturation.append(metrics["effort_saturation"])
        slips.extend(_contact_slip(robot))
        collisions.append(metrics["self_collision"] or robot.contact_metrics()[1] > 0)
        now += robot.control_dt
        if fallen:
            failure = "fall"
        elif command.mode != expected:
            failure = controller.reason
        return upright

    requested_steps = round(seconds / robot.control_dt)
    for bout in range(2):
        minimum, minimum_upright, completed_steps = float("inf"), 1.0, 0
        for _ in range(requested_steps):
            if now > controller.reference.source_time + 1e-9:
                controller.set_reference(robot.neutral_reference(now))
            minimum_upright = min(minimum_upright, step(Mode.ACTIVE, bout + 1))
            minimum = min(minimum, robot.data.qpos[2])
            completed_steps += 1
            if failure:
                break
        stats.append(
            {
                "bout": bout + 1,
                "minimum_root_height_m": float(minimum),
                "minimum_upright_cos": minimum_upright,
                "seconds_requested": seconds,
                "seconds_completed": completed_steps * robot.control_dt,
                "completed": failure is None and completed_steps == requested_steps,
            }
        )
        if failure:
            break
        if bout == 0:
            controller.pause(now)
            for _ in range(50):
                step(Mode.PAUSED, 0)
                if failure:
                    break
            if failure:
                break
            contacts, _ = robot.contact_metrics()
            try:
                controller.calibrate(robot.neutral_reference(now), robot.state(now), bool(contacts.all()), now)
                controller.arm(robot.state(now), now)
            except RuntimeError as error:
                failure = str(error)
                break
    report = {
        "version": "standing-student-bouts-v2",
        "controller": str(policy_path) if policy else "untrained_double_support_imu_baseline",
        "policy_sha256": policy.metadata["sha256"] if policy else None,
        "model_signature": robot.signature,
        "source_kind": "synthetic_neutral_robot_reference",
        "bouts": stats,
        "falls": falls,
        "failure_reason": failure,
        "resets_after_initialization": 0,
        "max_target_step_rad": max_jump,
        "max_target_step_limit_fraction": max_step_fraction,
        "action_settings": controller.action_settings,
        "command_p95_ms": float(np.percentile(p95, 95) * 1000),
        "joint_rmse_rad": float(np.sqrt(np.mean(qerr))),
        "relative_body_rmse_m": float(np.sqrt(np.mean(bodyerr))),
        "effort_saturation_fraction": float(np.mean(saturation)),
        "contact_point_slip_mean_m_s": float(np.mean(slips)) if slips else 0.0,
        "self_collision_frames": int(np.count_nonzero(collisions)),
        "events": controller.events,
        "five_minute_duration_met": seconds >= 300,
        "behaviorally_accepted": False,
        "live_mocopi_tested": False,
        "hardware_verified": False,
        "scope": "Standing, supported pause, calibration and rearming with the selected actor. No human retargeting or dynamic-tracking acceptance.",
    }
    report["passed"] = bool(
        not failure
        and len(stats) == 2
        and all(s["completed"] for s in stats)
        and report["command_p95_ms"] <= THRESHOLDS["command_p95_ms"]
        and report["joint_rmse_rad"] <= THRESHOLDS["max_joint_rmse_rad"]
        and report["relative_body_rmse_m"] <= THRESHOLDS["max_relative_body_rmse_m"]
        and report["effort_saturation_fraction"] <= THRESHOLDS["max_effort_saturation_fraction"]
        and report["contact_point_slip_mean_m_s"] <= THRESHOLDS["max_contact_slip_m_s"]
        and report["self_collision_frames"] == 0
        and max_step_fraction <= 1 + 1e-5
    )
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(
        Path(output).with_suffix(".rollout.npz"),
        qpos=np.asarray(trace),
        phases=np.asarray(phases),
        control_dt=robot.control_dt,
    )
    return report
