"""Interactive local simulation. This module never imports or writes a robot SDK."""

from contextlib import nullcontext
from dataclasses import replace
import json
from pathlib import Path
import select
import sys
import time

import mujoco
import numpy as np
import torch

from .contracts import MotionClip
from .learning import Policy
from .mocopi import UdpReceiver
from .recording import Recorder
from .retarget import Retargeter
from .robot import K1Model
from .servo import step_command
from .runtime import CommandOwner, Controller, Mode


def _viewer(robot, enabled):
    if enabled:
        import mujoco.viewer

        return mujoco.viewer.launch_passive(robot.model, robot.data)
    return nullcontext(None)


def demo(seconds, output, viewer=False):
    torch.set_num_threads(1)
    robot = K1Model()
    controller = Controller(robot)
    controller.calibrate(robot.neutral_reference(), robot.state(0), True, 0)
    controller.arm(robot.state(0), 0)
    reference_data = mujoco.MjData(robot.model)
    references, trace, min_height, max_error = [], [], 1.0, 0.0
    start = time.monotonic()
    with _viewer(robot, viewer) as window:
        for step in range(round(seconds / robot.control_dt)):
            now = step * robot.control_dt
            q = robot.neutral.copy()
            # Slow asymmetric reaching plus head motion; explicitly synthetic.
            q[0] = 0.2 * np.sin(now * 0.7)
            q[2] = -0.25 * (1 - np.cos(now * 0.6))
            q[3] = -1.3 + 0.2 * (1 - np.cos(now * 0.6))
            q[5] = -0.25 - 0.15 * (1 - np.cos(now * 0.6))
            q[6] = -0.15 * (1 - np.cos(now * 0.4))
            reference_data.qpos[:] = robot.neutral_qpos
            reference_data.qpos[7:] = q
            mujoco.mj_forward(robot.model, reference_data)
            dq = np.zeros(22) if not references else (q - references[-1].joint_position) / robot.control_dt
            ref = replace(
                robot.neutral_reference(now),
                joint_position=q,
                joint_velocity=dq,
                landmarks=robot.landmarks(reference_data),
            )
            references.append(ref)
            if step:
                controller.set_reference(ref)
            command = controller.tick(robot.state(now), now, supported=True)
            step_command(robot, command, controller.action_settings)
            trace.append(robot.data.qpos.copy())
            min_height = min(min_height, robot.data.qpos[2])
            max_error = max(max_error, float(np.sqrt(np.mean((robot.data.qpos[7:] - q) ** 2))))
            if robot.data.qpos[2] < 0.22:
                break
            if window:
                if not window.is_running():
                    break
                window.sync()
                wait = start + (step + 1) * robot.control_dt - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "source_kind": "synthetic_joint_reference",
        "controller": "zero_residual_with_imu_prior",
        "seconds_requested": seconds,
        "seconds_completed": len(trace) * robot.control_dt,
        "minimum_root_height_m": min_height,
        "max_joint_rmse_rad": max_error,
        "fell": bool(min_height < 0.22),
        "hardware_tested": False,
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(output.with_suffix(".rollout.npz"), qpos=np.array(trace))
    MotionClip.from_references(
        references,
        {
            "source_motion_id": "synthetic/reaching",
            "family": "reach",
            "model_signature": robot.signature,
            "physics_qualified": not report["fell"],
        },
    ).save(output.with_suffix(".reference.npz"))
    return report


def live_sim(args):
    torch.set_num_threads(1)
    robot = K1Model()
    policy = Policy(args.policy, robot.signature) if args.policy else None
    controller = Controller(robot, policy)
    from .actuators import configure_robot_actuators
    # Bind the viewer and live servo to the same effective model/data instance.
    configure_robot_actuators(robot, controller.action_settings)
    retargeter = Retargeter(robot)
    recorder = Recorder(args.record) if args.record else None
    receiver = UdpReceiver(args.host, args.port, recorder)
    # Supported simulation idle remains active while waiting for human input.
    start = time.monotonic()
    controller.calibrate(robot.neutral_reference(start), robot.state(start), True, start)
    last_frame = None
    reminder_sent = False
    displayed_mode = Mode.READY
    print(
        "Simulation idle. Commands: calibrate, arm, pause, recenter, restart-input, status, stop, quit.", flush=True
    )
    try:
        with (
            CommandOwner(Path("/tmp") / f"k1-motion-sim-{args.port}.lock"),
            _viewer(robot, args.viewer) as viewer,
        ):
            while args.seconds <= 0 or time.monotonic() - start < args.seconds:
                loop_start = time.monotonic()
                if not reminder_sent and loop_start - start >= 300:
                    print("Five-minute reminder: pause and recenter when convenient.", flush=True)
                    reminder_sent = True
                contacts, _ = robot.contact_metrics()
                supported = bool(contacts.all())
                state = robot.state(loop_start)
                frame = receiver.latest()
                if select.select([sys.stdin], [], [], 0)[0]:
                    command = sys.stdin.readline().strip()
                    try:
                        if command == "quit":
                            break
                        if command in ("calibrate", "recenter"):
                            if frame is None or loop_start - frame.received_time > 0.15:
                                raise ValueError("Fresh mocopi frame required")
                            if controller.mode not in (Mode.READY, Mode.PAUSED):
                                raise ValueError("Pause before calibrating")
                            retargeter.calibrate(frame)
                            ref = retargeter.process(frame)
                            controller.calibrate(ref, state, supported, loop_start)
                            last_frame = (frame.session, frame.frame_number)
                            if recorder:
                                recorder.event("calibration", loop_start, retargeter.calibration.metadata())
                        elif command == "arm":
                            if retargeter.calibration is None:
                                raise ValueError("Calibrate mocopi before arming")
                            controller.arm(state, loop_start)
                        elif command == "pause":
                            controller.pause(loop_start)
                        elif command == "restart-input":
                            controller.pause(loop_start)
                            receiver.reset_session()
                            retargeter.calibration = None
                            last_frame = None
                        elif command == "stop":
                            controller.stop(loop_start)
                        elif command == "status":
                            print(
                                json.dumps(
                                    {
                                        "mode": controller.mode.value,
                                        "frame_age_ms": None
                                        if frame is None
                                        else 1000 * (loop_start - frame.received_time),
                                        "input": dict(receiver.decoder.counts),
                                    }
                                ),
                                flush=True,
                            )
                        if command:
                            print(controller.mode.value, controller.reason, flush=True)
                            if recorder:
                                recorder.event("operator_" + command, loop_start)
                    except (ValueError, RuntimeError) as exc:
                        print(str(exc), flush=True)
                if frame is not None and retargeter.calibration is not None:
                    key = (frame.session, frame.frame_number)
                    if key != last_frame:
                        try:
                            controller.set_reference(retargeter.process(frame))
                            last_frame = key
                        except ValueError as exc:
                            controller.pause(loop_start)
                            retargeter.calibration = None
                            print(f"Input paused: {exc}", flush=True)
                command = controller.tick(state, loop_start, supported)
                if command.mode != displayed_mode:
                    print(f"{command.mode.value}: {command.reason}", flush=True)
                    displayed_mode = command.mode
                step_command(robot, command, controller.action_settings)
                if viewer:
                    if not viewer.is_running():
                        break
                    viewer.sync()
                delay = loop_start + robot.control_dt - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
    except KeyboardInterrupt:
        pass
    finally:
        receiver.close()
        if recorder:
            recorder.close()
    return {
        "mode": controller.mode.value,
        "events": controller.events,
        "input_counts": dict(receiver.decoder.counts),
    }
