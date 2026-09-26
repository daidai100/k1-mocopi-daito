"""Explicit policy action settings, separate from the pinned robot/data model.

Legacy exports use the robot configuration. Experiments may change controller
authority without changing reference retargeting or physical joint/effort limits.
These settings are simulation assumptions, not qualified hardware capabilities.
"""

import numpy as np


def action_settings(robot, settings=None):
    values = {
        "residual_scale": float(robot.config["residual_scale"]),
        "command_velocity_limit": float(robot.config["command_velocity_limit"]),
    }
    optional = {"target_velocity_scale", "kp_scale", "kd_scale",
                "upper_body_residual_scale", "imu_reference_rate_scale", "arm_collision_clearance",
                "actuator_profile", "command_position_margin_rad", "operating_speed_guard",
                "mask_inactive_actions", "ankle_prior_scale"}
    if settings is not None:
        if not isinstance(settings, dict) or set(settings) - (set(values) | optional):
            raise ValueError("Unknown policy action settings")
        values.update(settings)
    for name, value in values.items():
        if name in ("operating_speed_guard", "mask_inactive_actions"):
            if not isinstance(value, bool):
                raise ValueError(f"Policy {name} must be a boolean")
            continue
        if name == "command_position_margin_rad":
            margin = np.asarray(value, dtype=float)
            if (margin.shape not in ((), (22,)) or not np.isfinite(margin).all()
                    or np.any(margin < 0)
                    or np.any(2*margin >= robot.limits[:, 1]-robot.limits[:, 0])):
                raise ValueError("Policy command_position_margin_rad must be a finite nonnegative scalar "
                                 "or 22-vector smaller than each joint half-range")
            values[name] = float(margin) if margin.ndim == 0 else margin.tolist()
            continue
        if name == "actuator_profile":
            from .actuators import ACTUATOR_PROFILE
            if value != ACTUATOR_PROFILE:
                raise ValueError(f"Unknown actuator profile: {value}")
            continue
        if name == "arm_collision_clearance":
            if not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value <= .03:
                raise ValueError("Arm collision clearance must be in [0, 0.03] metres")
            values[name] = float(value)
            continue
        if name in ("target_velocity_scale", "kp_scale", "kd_scale", "command_velocity_limit"):
            a = np.asarray(value, dtype=float)
            if (a.shape not in ((), (22,)) or not np.isfinite(a).all()
                    or np.any(a < 0 if name == "target_velocity_scale" else a <= 0)):
                raise ValueError(f"Policy {name} must be a finite valid scalar or 22-vector")
            values[name] = float(a) if a.ndim == 0 else a.tolist()
            continue
        if name in ("upper_body_residual_scale", "imu_reference_rate_scale", "ankle_prior_scale"):
            if not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Policy {name} must be in [0, 1]")
            values[name] = float(value)
            continue
        if not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"Policy {name} must be finite and positive")
        values[name] = float(value)
    return values


def active_action_mask(settings):
    """Opt-in, checkpointed exclusion of joints with zero learned authority."""
    disabled = settings.get('mask_inactive_actions', False) and settings.get('upper_body_residual_scale', 1.) == 0
    return [0. if disabled else 1.]*10+[1.]*12


def action_statistics(distribution, action, mask=None):
    logprob, entropy = distribution.log_prob(action), distribution.entropy()
    if mask is not None:
        logprob, entropy = logprob*mask, entropy*mask
    return logprob.sum(-1), entropy.sum(-1)


def command_metrics(trace):
    """Small tensor reductions only; detailed arrays stay in last_command/replay."""
    return {
        'command/leg_action_bound_fraction': (trace['action'][..., 10:].abs() > .95).float().mean(),
        'command/leg_any_action_bound_fraction': (trace['action'][..., 10:].abs() > .95).any(-1).float().mean(),
        'command/guard_delta_rms_rad': (trace['executed_target']-trace['raw_target']).square().mean().sqrt(),
        'command/joint_clamp_fraction': ((trace['joint_limited_target']-trace['raw_target']).abs() > 1e-6).float().mean(),
        'command/slew_fraction': ((trace['slew_target']-trace['joint_limited_target']).abs() > 1e-6).float().mean(),
        'command/arm_projection_fraction': ((trace['arm_target']-trace['slew_target']).abs() > 1e-6).float().mean(),
        'command/position_guard_fraction': ((trace['executed_target']-trace['arm_target']).abs() > 1e-6).float().mean(),
        'command/velocity_guard_fraction': ((trace['executed_velocity']-trace['raw_velocity']).abs() > 1e-6).float().mean(),
        'command/ankle_prior_rms_rad': trace['ankle_prior'].square().mean().sqrt(),
    }


def command_velocity_limits(robot, settings):
    return np.minimum(robot.velocity_limit, settings["command_velocity_limit"])


def target_velocities_tensor(ref, limits, settings, age=None):
    """Causal K1 joint velocities; stale held packets never keep feedforward alive."""
    import torch

    scale = torch.as_tensor(settings.get("target_velocity_scale", 0.0),
                            dtype=ref["joint_velocity"].dtype, device=ref["joint_velocity"].device)
    velocity = (ref["joint_velocity"] * scale).clamp(-limits, limits)
    if age is not None:
        velocity = torch.where((age <= 0.04)[:, None], velocity, torch.zeros_like(velocity))
    return velocity


def position_guard_tensor(targets, velocities, previous_target, current_q, limits, step_limit, margin):
    """Keep desired positions inside a declared interior range without a jump.

    This acts on commands only. If initialization starts outside the interior
    envelope, enter it at the existing target slew cap. Outward feedforward
    tapers across one margin-width before the interior boundary, considering
    both measured position and the resulting target; inward recovery remains
    available. Zero-margin joints preserve their original commands exactly.
    This preventive command guard is not a guarantee on measured joint state.
    """
    import torch

    lower, upper = limits[:, 0]+margin, limits[:, 1]-margin
    interior = torch.maximum(torch.minimum(targets, upper), lower)
    guarded = torch.maximum(torch.minimum(interior, previous_target+step_limit),
                            previous_target-step_limit)
    enabled = margin > 0
    guarded = torch.where(enabled, guarded, targets)
    width = margin.clamp_min(torch.finfo(targets.dtype).eps)
    positive = ((upper-torch.maximum(current_q, guarded))/width).clamp(0., 1.)
    negative = ((torch.minimum(current_q, guarded)-lower)/width).clamp(0., 1.)
    factor = torch.where(velocities > 0, positive, negative)
    return guarded, torch.where(enabled, velocities*factor, velocities)
