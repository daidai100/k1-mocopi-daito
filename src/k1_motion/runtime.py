"""Simulation/deployment-independent controller state machine and command ownership."""

from dataclasses import dataclass, replace
from enum import Enum
import fcntl
from pathlib import Path

import numpy as np
import torch

from .observations import ObservationBuilder, reference_tensor, targets_tensor
from .actuation import (action_settings, command_velocity_limits, target_velocities_tensor,
                        position_guard_tensor)


class Mode(str, Enum):
    DISARMED = "disarmed"
    READY = "ready"
    ACTIVE = "active"
    PAUSED = "paused"
    FAULT = "fault"
    STOPPED = "stopped"


@dataclass(frozen=True)
class Command:
    targets: np.ndarray
    mode: Mode
    damping_only: bool
    reason: str
    velocities: np.ndarray | None = None


class CommandOwner:
    """Exclusive OS lock, held for the command publisher's entire lifetime."""

    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = path.open("a+")
        try:
            fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.stream.close()
            raise RuntimeError("Another process owns K1 commands") from None

    def close(self):
        self.stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class ZeroResidual:
    """Explicit untrained, standing/slow-motion baseline; not a universal policy."""

    def __call__(self, observation):
        return np.zeros(22)


class Controller:
    def __init__(self, robot, policy=None, history=None, input_timeout=0.15, state_timeout=0.06):
        self.robot, self.policy = robot, policy or ZeroResidual()
        from .action_chunks import checkpoint_chunk_size
        self.action_chunk_size = checkpoint_chunk_size(getattr(self.policy,'metadata',{}))
        self.pending_chunk = None
        self.chunk_cursor = 0
        self.action_settings = action_settings(robot, getattr(policy, "metadata", {}).get("action_settings"))
        from .actuation import active_action_mask
        self.action_mask = np.asarray(active_action_mask(self.action_settings))
        if self.action_settings.get('actuator_profile'):
            from .actuators import actuator_contract
            if getattr(policy, 'metadata', {}).get('actuator_contract') != actuator_contract(robot, self.action_settings):
                raise ValueError('Policy effective actuator contract differs from runtime')
        self.command_velocity_limits = command_velocity_limits(robot, self.action_settings)
        margin = np.asarray(self.action_settings.get('command_position_margin_rad', 0.))
        self.position_margin = (torch.tensor(margin, dtype=torch.float32)
                                if np.any(margin > 0) else None)
        self.arm_feedback = None
        if self.action_settings.get("arm_collision_clearance", 0):
            from .collision_feedback import ArmCollisionFeedback

            self.arm_feedback = ArmCollisionFeedback(robot, self.action_settings["arm_collision_clearance"])
        if history is None:
            history = getattr(policy, "metadata", {}).get("observation", {}).get("history", 4)
        from .observations import observation_profile
        profile = observation_profile(getattr(policy, "metadata", {}).get("observation", {}))
        self.builder = ObservationBuilder(robot.neutral, history, profile=profile,
            command_feedback='command_feedback' in getattr(policy, 'metadata', {}).get('observation', {}))
        self.preview_buffer = None
        if profile == 'preview':
            from .preview import PreviewBuffer
            from .observations import valid_observation_contract
            contract = policy.metadata['observation']
            if not valid_observation_contract(contract, history):
                raise ValueError('Incompatible buffered preview policy contract')
            self.preview_buffer = PreviewBuffer(delay_s=contract['playback_delay_s'],
                                                horizon_s=contract['preview_horizon_s'])
        self.input_timeout, self.state_timeout = input_timeout, state_timeout
        self.mode = Mode.DISARMED
        self.reason = "awaiting calibration"
        self.reference = None
        self.previous_action = np.zeros(22)
        self.previous_target = robot.neutral.copy()
        self.previous_velocity = np.zeros(22)
        self.previous_guard_delta = np.zeros(22)
        self.last_command = {}
        self.last_tick = None
        self.supported_since = None
        self.events = []

    def _mode(self, mode, reason, now):
        self.mode, self.reason = mode, reason
        self.pending_chunk = None
        self.chunk_cursor = 0
        if mode != Mode.ACTIVE:
            self.previous_velocity[:] = 0
            self.previous_guard_delta[:] = 0
        self.events.append({"time": now, "mode": mode.value, "reason": reason})

    def set_reference(self, reference, *, playback_time=None):
        if self.reference is not None:
            if reference.session != self.reference.session:
                self.builder.reset()
                self.previous_action[:] = 0
                if self.preview_buffer is not None:
                    self.preview_buffer.reset()
                self._mode(
                    Mode.PAUSED, "input session changed; recalibration required", reference.received_time
                )
                return
            if reference.source_time <= self.reference.source_time and not (
                    self.preview_buffer is not None and playback_time is not None
                    and reference.source_time == self.reference.source_time):
                if self.preview_buffer is not None:
                    return  # late packets do not revive the input-loss watchdog
                raise ValueError("Reference must advance")
        if self.preview_buffer is not None:
            self.preview_buffer.push(reference, playback_time)
        self.reference = reference

    def finish_reference(self):
        """Explicit finite-recording end; never infer end-of-stream from silence."""
        if self.preview_buffer is None:
            raise RuntimeError('Finite buffered drain requires a preview policy')
        self.preview_buffer.finish()

    def calibrate(self, reference, state, supported, now, *, playback_time=None):
        if self.mode not in (Mode.DISARMED, Mode.PAUSED, Mode.READY):
            raise RuntimeError("Pause in supported idle before recalibration")
        if not supported or not state.healthy or not reference.valid:
            raise RuntimeError("Calibration requires supported healthy idle and a valid reference")
        if self.mode == Mode.PAUSED and (self.supported_since is None or now - self.supported_since < 0.3):
            raise RuntimeError("Wait for supported idle before recalibration")
        self.reference = reference
        if self.preview_buffer is not None:
            self.preview_buffer.reset(anchor_time=now)
            self.preview_buffer.push(reference, playback_time)
        self.builder.reset()
        self.previous_action[:] = 0
        # Keep the last motor target to avoid a command jump at the new origin.
        if self.mode == Mode.DISARMED:
            self.previous_target = state.joint_position.copy()
        self._mode(Mode.READY, "calibrated; explicit arm required", now)

    def arm(self, state, now):
        ref = self.reference
        if self.mode != Mode.READY or ref is None or not ref.valid or not state.healthy:
            raise RuntimeError("Controller not ready to arm")
        fresh = ref.received_time
        if self.preview_buffer is not None:
            window = self.preview_buffer.sample(now)
            if window is None:
                raise RuntimeError('Reference buffer must fill before arming')
            if window.completed or not window.current.valid:
                raise RuntimeError('Valid unfinished buffered reference required to arm')
            fresh = window.latest_received_time
            fresh = min(fresh, now) if fresh-now <= 1e-8 else fresh
        if (
            not 0 <= now - fresh <= self.input_timeout
            or not 0 <= now - state.time <= self.state_timeout
        ):
            raise RuntimeError("Fresh input and state required to arm")
        self._mode(Mode.ACTIVE, "operator armed", now)

    def pause(self, now):
        if self.mode in (Mode.ACTIVE, Mode.READY):
            self.builder.reset()
            self.previous_action[:] = 0
            if self.preview_buffer is not None:
                self.preview_buffer.reset()
            self._mode(Mode.PAUSED, "operator paused", now)

    def stop(self, now):
        self._mode(Mode.STOPPED, "operator stop", now)

    def tick(self, state, now, supported=False):
        if not np.isfinite(now) or (self.last_tick is not None and now <= self.last_tick):
            raise ValueError("Control clock must advance monotonically")
        dt = self.robot.control_dt if self.last_tick is None else now - self.last_tick
        self.last_tick = now
        if supported:
            if self.supported_since is None:
                self.supported_since = now
        else:
            self.supported_since = None
        if not state.healthy or not 0 <= now - state.time <= self.state_timeout or dt > 0.1:
            self._mode(Mode.FAULT, "invalid/stale robot feedback or missed control deadline", now)
        window = None
        reference_age = None
        if self.mode == Mode.ACTIVE:
            ref = self.reference
            fresh = ref is not None and 0 <= now-ref.received_time <= self.input_timeout
            if self.preview_buffer is not None:
                window = self.preview_buffer.sample(now)
                if window is not None and window.completed:
                    self._mode(Mode.STOPPED, 'buffered reference completed', now)
                ref = None if window is None else window.current
                fresh = (window is not None
                         and (-1e-8 <= now-window.latest_received_time <= self.input_timeout or window.draining)
                         and window.current_sample_age <= self.input_timeout)
                reference_age = None if window is None else window.current_sample_age
            if self.mode == Mode.ACTIVE and (ref is None or not ref.valid or not fresh):
                self.builder.reset()
                self.previous_action[:] = 0
                self._mode(
                    Mode.PAUSED if supported else Mode.FAULT,
                    "input lost; supported idle"
                    if supported
                    else "input lost outside qualified idle envelope",
                    now,
                )
        if self.mode in (Mode.DISARMED, Mode.FAULT, Mode.STOPPED):
            return Command(state.joint_position.copy(), self.mode, True, self.reason)
        if self.mode in (Mode.PAUSED, Mode.READY):
            if not supported:
                self._mode(Mode.FAULT, "idle support unavailable", now)
                return Command(state.joint_position.copy(), self.mode, True, self.reason)
            ref = replace(self.robot.neutral_reference(now), received_time=now)
            action = np.zeros(22)
        else:
            ref = self.reference if window is None else window.current
            observation = self.builder.build(state, ref, self.previous_action, now, preview=window,
                command_state=(self.previous_target, self.previous_velocity, self.previous_guard_delta))
            if self.pending_chunk is None or self.chunk_cursor >= self.action_chunk_size:
                predicted = np.asarray(self.policy(observation))
                expected = (22,) if self.action_chunk_size == 1 else (self.action_chunk_size,22)
                if predicted.shape != expected or not np.isfinite(predicted).all():
                    self._mode(Mode.FAULT, "invalid policy output", now)
                    return Command(state.joint_position.copy(), self.mode, True, self.reason)
                self.pending_chunk = predicted.reshape(self.action_chunk_size,22).copy()
                self.chunk_cursor = 0
            action = self.pending_chunk[self.chunk_cursor]
            self.chunk_cursor += 1
            action = np.clip(action, -1, 1) * self.action_mask
        command_trace = {}
        target = targets_tensor(
            torch.tensor(action[None], dtype=torch.float32),
            reference_tensor(ref),
            torch.tensor(np.array(state.orientation)[None], dtype=torch.float32),
            torch.tensor(np.array(state.angular_velocity)[None], dtype=torch.float32),
            torch.tensor(self.robot.limits, dtype=torch.float32),
            self.action_settings["residual_scale"],
            self.action_settings.get("upper_body_residual_scale", 1.0),
            self.action_settings.get("imu_reference_rate_scale", 0.0),
            self.action_settings.get("ankle_prior_scale", 1.0), diagnostics=command_trace,
        )[0].numpy()
        self.last_command = {k: v[0].numpy().copy() for k, v in command_trace.items()}
        limit = min(dt, self.robot.control_dt * 2) * self.command_velocity_limits
        target = np.clip(target, self.previous_target - limit, self.previous_target + limit)
        self.last_command['slew_target'] = target.copy()
        if self.arm_feedback is not None and self.mode == Mode.ACTIVE:
            target = self.arm_feedback.project(state.joint_position, state.joint_velocity, target)
            target = np.clip(target, self.robot.limits[:, 0], self.robot.limits[:, 1])
        target = np.clip(target, self.previous_target - limit, self.previous_target + limit)
        self.previous_action = action.copy()
        self.last_command['arm_target'] = target.copy()
        velocity = target_velocities_tensor(
            reference_tensor(ref), torch.tensor(self.command_velocity_limits, dtype=torch.float32),
            self.action_settings, torch.tensor([max(0.0, now - ref.received_time)
                                               if reference_age is None else reference_age]),
        )[0].numpy()
        self.last_command['raw_velocity'] = velocity.copy()
        if self.position_margin is not None:
            guarded, guarded_velocity = position_guard_tensor(
                torch.tensor(target[None], dtype=torch.float32),
                torch.tensor(velocity[None], dtype=torch.float32),
                torch.tensor(self.previous_target[None], dtype=torch.float32),
                torch.tensor(np.array(state.joint_position)[None], dtype=torch.float32),
                torch.tensor(self.robot.limits, dtype=torch.float32),
                torch.tensor(limit, dtype=torch.float32), self.position_margin)
            enabled = self.position_margin.numpy() > 0
            # Preserve legacy double-precision slew arithmetic on disabled
            # joints in a mixed margin vector; their commands are unchanged.
            target = np.where(enabled, guarded[0].numpy(), target)
            velocity = np.where(enabled, guarded_velocity[0].numpy(), velocity)
        self.previous_target = target
        self.previous_velocity = velocity.copy()
        self.previous_guard_delta = target-self.last_command['raw_target']
        self.last_command.update(executed_target=target.copy(), executed_velocity=velocity.copy())
        return Command(self.previous_target.copy(), self.mode, False, self.reason, velocity)
