"""Causal observations; optional planar profile requires measured odometry."""

import numpy as np
import torch

from .contracts import JOINT_NAMES

FRAME_SIZE = 135
OBSERVATION_VERSION = "k1-causal-v1"
PLANAR_VERSION = "k1-causal-planar-v2"
PREVIEW_VERSION = "k1-buffered-preview-v1"
PREVIEW_FRAME_SIZE = 40


def observation_profile(contract):
    return {PLANAR_VERSION: 'planar', PREVIEW_VERSION: 'preview'}.get(contract.get('version'), 'causal')


def valid_observation_contract(contract, history=None):
    try:
        expected = observation_contract(contract['history'] if history is None else history,
            observation_profile(contract), preview_horizon_s=contract.get('preview_horizon_s', .3),
            command_feedback='command_feedback' in contract)
        return contract == expected
    except (KeyError, TypeError, ValueError):
        return False


def quat_mul(a, b):
    aw, av, bw, bv = a[..., :1], a[..., 1:], b[..., :1], b[..., 1:]
    return torch.cat(
        [aw * bw - (av * bv).sum(-1, keepdim=True), aw * bv + bw * av + torch.linalg.cross(av, bv)], -1
    )


def quat_inv(q):
    return torch.cat([q[..., :1], -q[..., 1:]], -1)


def quat_apply(q, v):
    uv = torch.linalg.cross(q[..., 1:], v.expand_as(q[..., 1:]))
    return v + 2 * (q[..., :1] * uv + torch.linalg.cross(q[..., 1:], uv))


def frame_tensor(
    q, dq, orientation, angular_velocity, previous_action, ref, frame_age, neutral, validate=True,
    profile="causal", position=None, velocity=None, command_state=None,
):
    """All inputs [N,...]; body gyro and projected gravity mirror deployment IMU."""
    inv = quat_inv(orientation)
    relative = quat_mul(inv, ref["root_orientation"])
    x = torch.zeros_like(angular_velocity)
    x[:, 0] = 1
    y = torch.zeros_like(x)
    y[:, 1] = 1
    gravity = torch.zeros_like(x)
    gravity[:, 2] = -1
    result = torch.cat(
        [
            q - neutral,
            dq * 0.05,
            quat_apply(inv, gravity),
            angular_velocity * 0.25,
            previous_action,
            ref["joint_position"] - q,
            ref["joint_velocity"] * 0.05,
            quat_apply(relative, x),
            quat_apply(relative, y),
            ref["root_position"][:, 2:3],
            quat_apply(inv, ref["root_velocity"][:, :3]) * 0.25,
            quat_apply(inv, ref["root_velocity"][:, 3:]) * 0.25,
            frame_age.reshape(-1, 1).clamp(0, 1),
            ref["valid"].reshape(-1, 1),
            ref["contacts"],
            ref["contact_confidence"],
        ],
        dim=-1,
    )
    if result.shape[-1] != FRAME_SIZE:
        raise ValueError(f"Observation schema size changed: {result.shape[-1]}")
    if profile in ("planar", "preview"):
        if position is None or velocity is None:
            raise ValueError("Planar observations require world position and velocity odometry")

        def heading(quaternion, world_omega):
            forward = quat_apply(quaternion, x)
            xy = forward[:, :2]
            yaw = xy / xy.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            derivative = torch.linalg.cross(world_omega, forward)
            rate = (forward[:, 0]*derivative[:, 1] - forward[:, 1]*derivative[:, 0]) / xy.square().sum(-1).clamp(min=1e-4)
            return yaw, rate[:, None]

        heading_actual, rate_actual = heading(orientation, quat_apply(orientation, angular_velocity))
        heading_ref, rate_ref = heading(ref["root_orientation"], ref["root_velocity"][:, 3:])
        heading_error = torch.stack(((heading_actual*heading_ref).sum(-1),
            heading_actual[:, 0]*heading_ref[:, 1]-heading_actual[:, 1]*heading_ref[:, 0]), -1)
        displacement = ref["root_position"] - position
        displacement = displacement.clone()
        displacement[:, 2] = 0
        velocity_error = ref["root_velocity"][:, :3] - velocity
        velocity_error = velocity_error.clone()
        velocity_error[:, 2] = 0
        result = torch.cat((result, position[:, :2], ref["root_position"][:, :2],
            quat_apply(inv, displacement)[:, :2], velocity[:, :2], ref["root_velocity"][:, :2],
            quat_apply(inv, velocity_error)[:, :2], heading_actual, heading_ref, heading_error,
            rate_actual, rate_ref), -1)
    elif profile != "causal":
        raise ValueError("Unknown observation profile")
    if command_state is not None:
        target, desired_velocity, guard_delta = command_state
        result = torch.cat((result, target-q, desired_velocity*.05, guard_delta), -1)
    if validate and not torch.isfinite(result).all():
        raise ValueError("Nonfinite actor observation")
    return result


def append_preview_tensor(causal, q, orientation, position, future_refs, mask):
    """Append three current preview slots; no robot-future state or history shift."""
    from .preview import PREVIEW_OFFSETS
    if len(future_refs) != len(PREVIEW_OFFSETS) or mask.shape != (len(q), len(PREVIEW_OFFSETS)):
        raise ValueError('Preview requires three future references and an[N,3] availability mask')
    inverse = quat_inv(orientation)
    x = torch.zeros_like(position)
    x[:, 0] = 1
    y = torch.zeros_like(position)
    y[:, 1] = 1
    slots = []
    for index, ref in enumerate(future_refs):
        relative = quat_mul(inverse, ref['root_orientation'])
        present = mask[:, index].bool() & ref['valid'].bool()
        features = torch.cat((ref['joint_position']-q,
            quat_apply(inverse, ref['root_position']-position),
            quat_apply(relative, x), quat_apply(relative, y),
            quat_apply(inverse, ref['root_velocity'][:, :3])*.25,
            quat_apply(inverse, ref['root_velocity'][:, 3:])*.25,
            ref['contacts']), -1)
        features = torch.where(present[:, None], features, torch.zeros_like(features))
        slots.append(torch.cat((features, present[:, None].to(q.dtype)), -1))
    return torch.cat((causal, *slots), -1)


class History:
    def __init__(self, batch, length=4, device="cpu", frame_size=FRAME_SIZE):
        self.length = length
        self.frames = torch.zeros((batch, length, frame_size + 1), device=device)

    def reset(self, indices=None):
        if indices is None:
            self.frames.zero_()
        else:
            self.frames[indices] = 0

    def push(self, frame):
        self.frames[:, :-1] = self.frames[:, 1:].clone()
        self.frames[:, -1, :-1] = frame
        self.frames[:, -1, -1] = 1
        return self.frames.flatten(1).clone()


def reference_tensor(reference, device="cpu"):
    result = {
        k: torch.as_tensor(np.array(getattr(reference, k)), dtype=torch.float32, device=device)[None]
        for k in (
            "root_position",
            "root_orientation",
            "root_velocity",
            "joint_position",
            "joint_velocity",
            "contacts",
            "contact_confidence",
        )
    }
    result["valid"] = torch.tensor([float(reference.valid)], device=device)
    return result


class ObservationBuilder:
    def __init__(self, neutral, history=4, profile="causal", command_feedback=False):
        self.neutral = torch.tensor(np.array(neutral), dtype=torch.float32)
        self.profile = profile
        self.command_feedback = command_feedback
        self.history = History(1, history, frame_size=observation_contract(
            history, profile, command_feedback=command_feedback)["frame_size"])

    def reset(self):
        self.history.reset()

    def build(self, state, reference, previous_action, now, *, preview=None, frame_age=None, command_state=None):
        def tensor(x):
            return torch.tensor(np.array(x), dtype=torch.float32)[None]

        if self.profile == 'preview':
            if preview is None or preview.current is not reference:
                raise ValueError('Buffered preview window matching the current reference is required')
            frame_age = preview.current_sample_age
        if frame_age is None:
            frame_age = max(0., now-reference.received_time)
        if self.command_feedback and command_state is None:
            raise ValueError('Command feedback observations require prior executed commands')
        frame = frame_tensor(
            tensor(state.joint_position),
            tensor(state.joint_velocity),
            tensor(state.orientation),
            tensor(state.angular_velocity),
            tensor(previous_action),
            reference_tensor(reference),
            torch.tensor([frame_age]),
            self.neutral,
            profile=self.profile,
            position=None if state.root_position is None else tensor(state.root_position),
            velocity=None if state.linear_velocity is None else tensor(state.linear_velocity),
            command_state=tuple(tensor(v) for v in command_state) if self.command_feedback else None,
        )
        result = self.history.push(frame)
        if self.profile == 'preview':
            result = append_preview_tensor(result, tensor(state.joint_position), tensor(state.orientation),
                tensor(state.root_position), [reference_tensor(r or reference) for r in preview.future],
                torch.tensor([preview.mask]))
        if not torch.isfinite(result).all():
            raise ValueError('Nonfinite actor observation')
        return result[0].numpy()


def targets_tensor(action, ref, orientation, angular_velocity, limits, residual_scale=0.25,
                   upper_body_residual_scale=1.0, imu_reference_rate_scale=0.0,
                   ankle_prior_scale=1.0, diagnostics=None):
    """All 22 residuals act jointly; a modest double-support ankle feedback prior.

    The prior is shared by training and runtime and disabled for single support or
    flight. Its standing qualification does not establish dynamic capability.
    """
    relative = quat_mul(quat_inv(ref["root_orientation"]), orientation)
    w, x, y, z = relative.unbind(-1)
    roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = torch.asin((2 * (w * y - z * x)).clamp(-1, 1))
    supported = (ref["contacts"].min(-1).values > 0.5).float()
    angular_velocity = angular_velocity - imu_reference_rate_scale * quat_apply(
        quat_inv(orientation), ref["root_velocity"][:, 3:])
    correction_pitch = ankle_prior_scale * supported * (1.0 * pitch + 0.15 * angular_velocity[:, 1]).clamp(-0.25, 0.25)
    correction_roll = ankle_prior_scale * supported * (1.0 * roll + 0.15 * angular_velocity[:, 0]).clamp(-0.2, 0.2)
    residual = action.clamp(-1, 1).clone()
    residual[:, :10] *= upper_body_residual_scale
    result = ref["joint_position"] + residual_scale * residual
    result = result.clone()
    for i in (14, 20):
        result[:, i] += correction_pitch
    for i in (15, 21):
        result[:, i] += correction_roll
    clamped = result.clamp(limits[:, 0], limits[:, 1])
    if diagnostics is not None:
        diagnostics.update(action=action, reference_joint=ref['joint_position'],
            reference_velocity=ref['joint_velocity'], residual=residual_scale*residual,
            ankle_prior=result-(ref['joint_position']+residual_scale*residual),
            raw_target=result, joint_limited_target=clamped)
    return clamped


def observation_contract(history=4, profile="causal", preview_horizon_s=.3, command_feedback=False):
    if profile not in ("causal", "planar", "preview"):
        raise ValueError("Unknown observation profile")
    if not isinstance(history, int) or history < 1:
        raise ValueError('Observation history must be a positive integer')
    if not isinstance(command_feedback, bool):
        raise ValueError('Command feedback must be boolean')
    frame_size = FRAME_SIZE + (20 if profile in ("planar", "preview") else 0) + (66 if command_feedback else 0)
    result = {
        "version": {'causal': OBSERVATION_VERSION, 'planar': PLANAR_VERSION, 'preview': PREVIEW_VERSION}[profile],
        "frame_size": frame_size,
        "history": history,
        "size": history * (frame_size + 1),
        "joint_names": list(JOINT_NAMES),
        "future_frames": 0,
        "action": "joint_residual_normalized_plus_double_support_imu_prior",
        "robot_feedback": ["joint_position", "joint_velocity", "orientation", "body_angular_velocity"],
    }
    if command_feedback:
        result['command_feedback'] = dict(version='executed-command-v1', size=66,
            layout=['previous_target_minus_measured_q22', 'previous_desired_velocity22_scaled_.05',
                    'previous_executed_minus_raw_target22'],
            reset='measured/reference joint position, zero velocity and guard delta')
    if profile in ("planar", "preview"):
        result["robot_feedback"] += ["world_root_position", "world_linear_velocity"]
        result["planar_features"] = ["world_xy", "reference_world_xy", "body_xy_error",
            "world_xy_velocity", "reference_world_xy_velocity", "body_xy_velocity_error",
            "yaw_cos_sin", "reference_yaw_cos_sin", "yaw_error_cos_sin", "yaw_rate", "reference_yaw_rate"]
        result["odometry_frame"] = "same_world_frame_and_origin_as_reference"
    if profile == 'preview':
        from .preview import MAX_PREVIEW_HOLD_S, PLAYBACK_DELAY_S, PREVIEW_OFFSETS
        if preview_horizon_s not in (0., .3):
            raise ValueError('Preview horizon must be0 or300ms')
        result.update(size=result['size']+len(PREVIEW_OFFSETS)*PREVIEW_FRAME_SIZE,
            future_frames=len(PREVIEW_OFFSETS), preview_offsets=list(PREVIEW_OFFSETS),
            preview_frame_size=PREVIEW_FRAME_SIZE, preview_horizon_s=float(preview_horizon_s),
            playback_delay_s=PLAYBACK_DELAY_S, preview_max_hold_s=MAX_PREVIEW_HOLD_S,
            preview_layout='planar history followed by current100/200/300ms slots',
            preview_features=['joint_position_error22', 'body_root_displacement3', 'relative_orientation_axes6',
                              'body_root_linear_angular_velocity6_scaled_.25', 'contacts2', 'available1'],
            preview_availability='received by control time; causal hold<=40ms; no extrapolation beyond latest; missing/masked slots zero',
            frame_age='walltime minus playback delay minus actual sample arrival; packet freshness checked separately')
    return result
