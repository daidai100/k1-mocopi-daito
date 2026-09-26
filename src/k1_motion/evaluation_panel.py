"""Frozen, recording-grouped trials through causal human replay and the runtime.

Source recordings and their rejection outcomes stay in the denominator. Repeated
trials are explicitly labelled; they are never counted as independent captures.
Transport perturbations are engineering assumptions, not measured Sony errors.
"""

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import hashlib
import json
import multiprocessing
from pathlib import Path
import time

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import torch

from .contracts import HumanFrame
from .evaluation import THRESHOLDS
from .learning import Policy
from .math3d import quaternion, rotation
from .retarget import RETARGET_VERSION, Retargeter
from .robot import K1Model
from .servo import step_command
from .runtime import Controller, Mode
from .streaming import control_schedule
from .actuation import action_settings

SCENARIOS = ("clean", "initial_pose", "delay_jitter", "brief_loss", "push", "calibration")
PANEL_VERSION = "human-streaming-v2"


def build_panel(library, output, split="test", trials_per_family=20, seed=42):
    directory, output = Path(library), Path(output)
    index = (directory / "index.jsonl").read_bytes()
    rows = [r for r in map(json.loads, index.splitlines()) if r["split"] == split]
    if not rows or trials_per_family < 1:
        raise ValueError("Panel requires recordings and a positive trial count")
    grouped = defaultdict(list)
    fingerprints = {}
    for row in rows:
        grouped[row["family"]].append(row)
        if row["kinematics_accepted"]:
            fingerprints[row["id"]] = hashlib.sha256((directory / row["human_path"]).read_bytes()).hexdigest()
    rng = np.random.default_rng(seed)
    trials = []
    for family, records in sorted(grouped.items()):
        records.sort(key=lambda r: r["id"])
        for i in range(max(trials_per_family, len(records))):
            row = records[i % len(records)]
            trials.append(
                {
                    "trial_id": f"{family}-{i:03d}-{row['id']}",
                    "recording_id": row["id"],
                    "capture_group": row["capture_group"],
                    "family": family,
                    "scenario": SCENARIOS[(i + i // len(records)) % len(SCENARIOS)],
                    "repeat": i // len(records),
                    "seed": int(rng.integers(0, 2**31)),
                }
            )
    panel = {
        "version": PANEL_VERSION,
        "split": split,
        "seed": seed,
        "index_sha256": hashlib.sha256(index).hexdigest(),
        "trials": trials,
        "human_sha256": fingerprints,
        "retarget_version": RETARGET_VERSION,
        "independent_recordings": len(rows),
        "capture_groups": len({r["capture_group"] for r in rows}),
        "reference_rejections": sum(not r["kinematics_accepted"] for r in rows),
        "thresholds": THRESHOLDS,
        "metric_contract": {
            "body_error": "RMS Euclidean error of root-relative landmark positions",
            "slip": "mean tangential velocity at active foot-ground contact points",
            "latency": "human-frame validation, causal retargeting, and runtime command CPU time",
            "initialization": "one reference-state reset per trial; no resets after a fall",
        },
        "assumptions": {
            "initial_joint_noise_rad": 0.015,
            "initial_roll_pitch_rad": 0.02,
            "delay_seconds": 0.02,
            "jitter_seconds": 0.008,
            "loss_burst_source_frames": 3,
            "push_force_newtons": 10.0,
            "push_duration_seconds": 0.1,
            "calibration": "rigid yaw/translation and 3 percent human scale change",
            "raw_mocopi_tested": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(json.dumps(panel, indent=2) + "\n")
    return panel


def _contact_slip(robot):
    feet = {robot.model.body(f"{side}_ankle_roll_link").id for side in ("left", "right")}
    velocities = []
    jac = np.zeros((3, robot.model.nv))
    force = np.zeros(6)
    for i, contact in enumerate(robot.data.contact[: robot.data.ncon]):
        bodies = robot.model.geom_bodyid[[contact.geom1, contact.geom2]]
        body = next((int(b) for b in bodies if b in feet), None)
        if body is None or 0 not in bodies:
            continue
        mujoco.mj_contactForce(robot.model, robot.data, i, force)
        if force[0] <= 1.0:
            continue
        mujoco.mj_jac(robot.model, robot.data, jac, None, contact.pos, body)
        velocities.append(float(np.linalg.norm((jac @ robot.data.qvel)[:2])))
    return velocities


def _trial(robot, policy, row, case, human, assumptions, thresholds, output):
    if getattr(policy, "metadata", {}).get("reference_scale") is not None:
        raise ValueError("Human streaming evaluator cannot score scaled policies; use control_validation.replay_clip")
    base = {
        **case,
        "source_motion_id": row["source_motion_id"],
        "reference_rejected": not row["kinematics_accepted"],
        "resets_during_trial": 0,
    }
    if base["reference_rejected"]:
        return {
            **base,
            "completed": False,
            "tracking_passed": False,
            "fell": False,
            "reason": "offline_reference_rejected",
            "rejection_counts": row.get("rejection_counts", {}),
            "error": row.get("error"),
            "simulated_s": 0.0,
        }
    rng = np.random.default_rng(case["seed"])
    times = human["times"] - human["times"][0]
    if len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Canonical human replay requires increasing finite timestamps")
    arrivals = times.copy()
    dropped = np.zeros(len(times), bool)
    scenario = case["scenario"]
    if scenario == "delay_jitter":
        arrivals += assumptions["delay_seconds"] + rng.uniform(
            -assumptions["jitter_seconds"], assumptions["jitter_seconds"], len(times)
        )
        arrivals[0] = 0
    elif scenario == "brief_loss":
        for start in (len(times) // 3, 2 * len(times) // 3):
            dropped[start : start + assumptions["loss_burst_source_frames"]] = True
        dropped[0] = False
    transform = Rotation.from_euler("z", 1.2) if scenario == "calibration" else Rotation.identity()
    offset = np.array([2.0, -1.0, 0.0]) if scenario == "calibration" else np.zeros(3)
    scale = 1.03 if scenario == "calibration" else 1.0

    def frame(index):
        return HumanFrame(
            float(times[index]),
            float(arrivals[index]),
            int(index),
            transform.apply(human["positions"][index] * scale) + offset,
            quaternion(transform * rotation(human["orientations"][index])),
            row["source_motion_id"],
        )

    retarget = Retargeter(robot)
    retarget.calibrate(frame(0))
    initial = retarget.process(frame(0))
    if not initial.valid:
        return {
            **base,
            "completed": False,
            "tracking_passed": False,
            "fell": False,
            "reason": "initial_streaming_reference_rejected",
            "simulated_s": 0.0,
            "retarget_report": retarget.last_report,
        }
    robot.reset(initial)
    if scenario == "initial_pose":
        robot.data.qpos[7:] = np.clip(
            robot.data.qpos[7:]
            + rng.uniform(
                -assumptions["initial_joint_noise_rad"], assumptions["initial_joint_noise_rad"], 22
            ),
            robot.limits[:, 0],
            robot.limits[:, 1],
        )
        tilt = Rotation.from_euler(
            "xy",
            rng.uniform(-assumptions["initial_roll_pitch_rad"], assumptions["initial_roll_pitch_rad"], 2),
        )
        robot.data.qpos[3:7] = quaternion(rotation(initial.root_orientation) * tilt)
        mujoco.mj_forward(robot.model, robot.data)
    controller = Controller(robot, policy)
    controller.calibrate(replace(initial, received_time=0.0), robot.state(0), True, 0)
    controller.arm(robot.state(0), 0)
    last_index, failed, fallen = 0, False, False
    qerr, bodyerr, saturation, velocity_saturation, slips, latency, collisions = [], [], [], [], [], [], []
    trace, targets, indices, ages = [], [], [], []
    velocities, actions, torques, references, reference_roots, reference_orientations, reference_contacts = (
        [], [], [], [], [], [], []
    )
    duration = float(times[-1])
    ticks, source_indices = control_schedule(times, robot.control_dt, arrivals, dropped)
    for now, index in zip(ticks, source_indices):
        stamp = time.perf_counter()
        index = max(int(index), last_index)
        if index > last_index:
            controller.set_reference(retarget.process(frame(index)))
            last_index = index
        ref = controller.reference
        contacts, _ = robot.contact_metrics()
        command = controller.tick(robot.state(now), now, bool(contacts.all()))
        latency.append(time.perf_counter() - stamp)
        robot.data.xfrc_applied[:] = 0
        push_time = min(1.0, duration * 0.5)
        if scenario == "push" and push_time <= now < push_time + assumptions["push_duration_seconds"]:
            robot.data.xfrc_applied[robot.model.body("trunk").id, 0] = assumptions["push_force_newtons"]
        metrics = step_command(robot, command, controller.action_settings)
        # mj_step integrates qpos after computing derived positions/contacts.
        # Refresh them at the measured state without advancing the simulation.
        mujoco.mj_forward(robot.model, robot.data)
        landmarks = robot.landmarks()
        qerr.append(np.mean((robot.data.qpos[7:] - ref.joint_position) ** 2))
        relative = (landmarks - robot.data.qpos[:3]) - (ref.landmarks - ref.root_position)
        bodyerr.append(np.mean(np.sum(relative**2, axis=-1)))
        saturation.append(metrics["effort_saturation"])
        velocity_saturation.append(np.mean(np.abs(robot.data.qvel[6:]) > robot.velocity_limit))
        slips.extend(_contact_slip(robot))
        collisions.append(metrics["self_collision"] or robot.contact_metrics()[1])
        trace.append(robot.data.qpos.copy())
        targets.append(command.targets.copy())
        velocities.append(robot.data.qvel.copy())
        actions.append(controller.previous_action.copy())
        torques.append(robot.data.ctrl.copy())
        references.append(ref.joint_position.copy())
        reference_roots.append(ref.root_position.copy())
        reference_orientations.append(ref.root_orientation.copy())
        reference_contacts.append(ref.contacts.copy())
        indices.append(index)
        ages.append(now - ref.received_time)
        fallen = bool(
            robot.data.qpos[2] < thresholds["min_height_m"]
            or rotation(robot.data.qpos[3:7]).apply([0, 0, 1])[2] < thresholds["min_upright_cos"]
        )
        failed = command.mode != Mode.ACTIVE
        if fallen or failed:
            break
    result = {
        **base,
        "duration_s": duration,
        "simulated_s": len(trace) * robot.control_dt,
        "completed": not fallen and not failed,
        "fell": fallen,
        "reason": controller.reason if failed else "fall" if fallen else "finished",
        "joint_rmse_rad": float(np.sqrt(np.mean(qerr))),
        "relative_body_rmse_m": float(np.sqrt(np.mean(bodyerr))),
        "effort_saturation_fraction": float(np.mean(saturation)),
        "velocity_saturation_fraction": float(np.mean(velocity_saturation)),
        "contact_point_slip_mean_m_s": float(np.mean(slips)) if slips else 0.0,
        "self_collision_frames": int(np.count_nonzero(collisions)),
        "received_frame_to_command_p95_ms": float(np.percentile(latency, 95) * 1000),
        "received_frame_to_command_p99_ms": float(np.percentile(latency, 99) * 1000),
        "max_received_frame_age_s": float(max(ages)),
        "processed_source_frames": len(set(indices)),
        "source_frames": len(times),
        "measurement_version": "substep-collision-v2",
    }
    result["tracking_passed"] = bool(
        result["completed"]
        and result["joint_rmse_rad"] <= thresholds["max_joint_rmse_rad"]
        and result["relative_body_rmse_m"] <= thresholds["max_relative_body_rmse_m"]
        and result["effort_saturation_fraction"] <= thresholds["max_effort_saturation_fraction"]
        and result["velocity_saturation_fraction"] == 0
        and result["contact_point_slip_mean_m_s"] <= thresholds["max_contact_slip_m_s"]
        and result["self_collision_frames"] == 0
        and result["received_frame_to_command_p95_ms"] <= thresholds["command_p95_ms"]
    )
    np.savez_compressed(
        output / (case["trial_id"] + ".npz"),
        qpos=np.asarray(trace),
        targets=np.asarray(targets),
        qvel=np.asarray(velocities),
        actions=np.asarray(actions),
        torques=np.asarray(torques),
        reference_joints=np.asarray(references),
        reference_roots=np.asarray(reference_roots),
        reference_orientations=np.asarray(reference_orientations),
        reference_contacts=np.asarray(reference_contacts),
        source_indices=np.asarray(indices),
        received_frame_age=np.asarray(ages),
        control_dt=robot.control_dt,
    )
    return result


def _evaluate_case(robot, policy, rows, case, panel, directory, cache, output):
    row = rows[case["recording_id"]]
    if row["split"] != panel["split"] or row["capture_group"] != case["capture_group"]:
        raise ValueError("Panel/source identity mismatch")
    if row["kinematics_accepted"] and row["model_signature"] != robot.signature:
        raise ValueError("Panel robot model differs")
    if row["kinematics_accepted"] and row["id"] not in cache:
        path = directory / row["human_path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != panel["human_sha256"][row["id"]]:
            raise ValueError("Frozen panel canonical human data changed")
        with np.load(path) as clip:
            cache[row["id"]] = {k: clip[k] for k in ("times", "positions", "orientations")}
    return _trial(
        robot, policy, row, case, cache.get(row["id"]), panel["assumptions"], panel["thresholds"], output
    )


def _initialize_panel_worker(directory, policy_path, rows, panel, output):
    global _panel_worker
    torch.set_num_threads(1)
    robot = K1Model()
    policy = Policy(policy_path, robot.signature) if policy_path else None
    # Each worker keeps its model, policy and canonical recordings loaded. _trial
    # creates fresh controller/retarget histories and explicitly resets physics.
    _panel_worker = (robot, policy, rows, panel, directory, {}, output)


def _worker_trial(case):
    robot, policy, rows, panel, directory, cache, output = _panel_worker
    return _evaluate_case(robot, policy, rows, case, panel, directory, cache, output)


def evaluate_panel(library, panel_path, output, policy_path=None, limit=None, trial_ids=None, workers=1):
    torch.set_num_threads(1)
    directory, output = Path(library), Path(output)
    panel_bytes = Path(panel_path).read_bytes()
    panel = json.loads(panel_bytes)
    index = (directory / "index.jsonl").read_bytes()
    if (
        panel["version"] != PANEL_VERSION
        or panel["retarget_version"] != RETARGET_VERSION
        or hashlib.sha256(index).hexdigest() != panel["index_sha256"]
    ):
        raise ValueError("Frozen panel version/reference index mismatch")
    if limit is not None and limit < 1:
        raise ValueError("Trial limit must be positive")
    if workers < 1:
        raise ValueError("Replay worker count must be positive")
    if limit is not None and trial_ids:
        raise ValueError("Choose a trial limit or explicit trial identities")
    if trial_ids and (
        len(set(trial_ids)) != len(trial_ids)
        or set(trial_ids) - {c["trial_id"] for c in panel["trials"]}
    ):
        raise ValueError("Requested trial identities must be unique and present in the frozen panel")
    rows = {r["id"]: r for r in map(json.loads, index.splitlines())}
    robot = K1Model()
    policy = Policy(policy_path, robot.signature) if policy_path else None
    if (
        policy
        and panel["split"] in ("validation", "test")
        and set(policy.metadata["train_parents"]) & {r["capture_group"] for r in panel["trials"]}
    ):
        raise ValueError("Training parent overlaps frozen evaluation panel")
    output.mkdir(parents=True, exist_ok=False)
    (output / "panel.json").write_bytes(panel_bytes)
    cache, results = {}, []
    cases = panel["trials"] if limit is None else panel["trials"][:limit]
    if trial_ids:
        cases = [case for case in panel["trials"] if case["trial_id"] in trial_ids]
    pool = None
    try:
        if workers == 1:
            iterator = (
                _evaluate_case(robot, policy, rows, case, panel, directory, cache, output) for case in cases
            )
        else:
            pool = ProcessPoolExecutor(
                max_workers=min(workers, len(cases)),
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_initialize_panel_worker,
                initargs=(directory, policy_path, rows, panel, output),
            )
            iterator = pool.map(_worker_trial, cases)
        with (output / "trials.jsonl").open("w", buffering=1) as stream:
            for result in iterator:
                results.append(result)
                stream.write(json.dumps(result) + "\n")
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
    families = defaultdict(list)
    for result in results:
        families[result["family"]].append(result)
    summary = {
        name: {
            "trials": len(items),
            "independent_recordings": len({r["recording_id"] for r in items}),
            "capture_groups": len({r["capture_group"] for r in items}),
            "completed": sum(r["completed"] for r in items),
            "tracking_passed": sum(r["tracking_passed"] for r in items),
            "reference_rejections": sum(r["reference_rejected"] for r in items),
        }
        for name, items in families.items()
    }
    thresholds = panel["thresholds"]
    covered = len(families) >= thresholds["min_families"] and all(
        len(v) >= thresholds["min_trials_per_family"] for v in families.values()
    )
    report = {
        "version": PANEL_VERSION,
        "panel_sha256": hashlib.sha256(panel_bytes).hexdigest(),
        "retarget_version": RETARGET_VERSION,
        "model_signature": robot.signature,
        "backend": f"mujoco-{mujoco.__version__}",
        "split": panel["split"],
        "policy": str(policy_path) if policy_path else "untrained_imu_baseline",
        "action_settings": action_settings(robot, policy.metadata.get("action_settings") if policy else None),
        "control_ablation": policy.metadata.get("control_ablation") if policy else None,
        "replay_workers": workers,
        "families": summary,
        "trials": len(results),
        "coverage_target_met": covered,
        "partial_panel": len(results) != len(panel["trials"]),
        "behaviorally_accepted": bool(
            policy
            and covered
            and len(results) == len(panel["trials"])
            and all(
                v["tracking_passed"] / v["trials"] >= thresholds["min_family_completion"]
                for v in summary.values()
            )
        ),
        "scope": "Canonical human streaming to exported student; separate bout/transition/fault gates required",
        "live_mocopi_tested": False,
        "hardware_verified": False,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
