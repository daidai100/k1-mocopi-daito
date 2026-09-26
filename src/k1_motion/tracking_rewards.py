"""Versioned BeyondMimic-inspired *tracking-only* reward ablation for causal K1.

Not a reproduction: root height replaces global anchor XYZ; K1 landmark sites
replace upstream body origins; velocities are backward differences. The six
upstream weights/sigmas retain their ratios, with peak tracking reward scaled
from 5 to 7.5 to match our current baseline. Safety costs and terminations stay
outside this module and are identical across the ablation.
"""
import torch

from .observations import quat_apply, quat_inv, quat_mul
from .spatial_rewards import SERIES_PROFILES
from .simple_rewards import SCREEN_PROFILES
from .world_objective import WORLD_PROFILES

BEYONDMIMIC_CAUSAL = "beyondmimic-causal-v1"
REWARD_PROFILES = ("legacy", BEYONDMIMIC_CAUSAL, *SERIES_PROFILES, *SCREEN_PROFILES, *WORLD_PROFILES)
TERMS = {
    "anchor_height": (0.5, 0.3),
    "anchor_orientation": (0.5, 0.4),
    "relative_body_position": (1.0, 0.3),
    "relative_body_orientation": (1.0, 0.4),
    "global_body_linear_velocity": (1.0, 1.0),
    "global_body_angular_velocity": (1.0, 3.14),
}
TRACKING_SCALE = 1.5


def angular_velocity(current, previous, dt):
    """World-frame shortest-arc rotation vector / dt, using only t and t-1."""
    delta = quat_mul(current, quat_inv(previous))
    delta = delta / delta.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    delta = torch.where(delta[..., :1] < 0, -delta, delta)
    vector = delta[..., 1:]
    length = vector.norm(dim=-1, keepdim=True)
    angle = 2 * torch.atan2(length, delta[..., :1].clamp(min=0))
    scale = torch.where(length > 1e-7, angle / length.clamp(min=1e-7), 2.0)
    return vector * scale / dt


def angle_squared(first, second):
    return angular_velocity(first, second, 1.0).square().sum(-1)


def beyondmimic_causal_tracking(state, reference, previous_orientation,
                              previous_reference_orientation, landmark_velocity, dt):
    # Same relative-yaw operation as upstream commands.py. Translate only XY;
    # preserve the reference height, including genuine low support postures.
    delta = quat_mul(state["orientation"], quat_inv(reference["root_orientation"]))
    w, x, y, z = delta.unbind(-1)
    yaw = torch.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    heading = torch.stack(((yaw/2).cos(), torch.zeros_like(yaw), torch.zeros_like(yaw), (yaw/2).sin()), -1)
    heading = heading[:, None].expand_as(reference["body_orientation"])
    anchor = torch.cat((state["position"][:, :2], reference["root_position"][:, 2:3]), -1)
    aligned_position = anchor[:, None] + quat_apply(
        heading, reference["landmarks"]-reference["root_position"][:, None])
    aligned_orientation = quat_mul(heading, reference["body_orientation"])
    angular_error = (angular_velocity(state["body_orientation"], previous_orientation, dt)
                     - angular_velocity(reference["body_orientation"], previous_reference_orientation, dt))
    errors = {
        "anchor_height": (state["position"][:, 2]-reference["root_position"][:, 2]).square(),
        "anchor_orientation": angle_squared(state["orientation"], reference["root_orientation"]),
        "relative_body_position": (state["landmarks"]-aligned_position).square().sum(-1).mean(-1),
        "relative_body_orientation": angle_squared(state["body_orientation"], aligned_orientation).mean(-1),
        "global_body_linear_velocity": (landmark_velocity-reference["landmark_velocity"]).square().sum(-1).mean(-1),
        "global_body_angular_velocity": angular_error.square().sum(-1).mean(-1),
    }
    components = {name: (-errors[name]/sigma**2).exp() for name, (_, sigma) in TERMS.items()}
    tracking = TRACKING_SCALE * sum(weight*components[name] for name, (weight, _) in TERMS.items())
    return tracking, components
