"""Versioned, bounded S2--S5 spatial/temporal tracking rewards."""
import copy
import math

import torch

SERIES_PROFILES = tuple(f"spatial-s{i}-v1" for i in range(2, 6))
BASE_WEIGHTS = dict(joint=.5, height=1., orientation=1., body_position=2., feet_position=2.,
                    body_orientation=.5, velocity=4., body_velocity=.5, root_xy=2., progress=0.,
                    support_score=0.)
SCALES = dict(joint=.3, height=.08, orientation=.4, body_position=.12, feet_position=.08,
              body_orientation=.5, velocity=.5, body_velocity=1.5, root_xy=.25, progress=.15)
SUPPORT_CONTRACT = dict(version="ground-foot-contact-patch-v1", normal_force_threshold_n=1.,
    confidence_threshold=.8, swing_support_score_max=.2, slip_scale_m_s=.1,
    slip_measurement="normal-force-weighted RMS tangential velocity at active ground contact points",
    velocity="point Jacobian times qvel, world coordinates, final physics-substep contact solve",
    contact_detector="sum positive solved foot-ground normal forces > 1 N",
    feet="left and right ankle-roll bodies", confidence_source="retained reference support score",
    confidence_semantics="expected support: score >= .8; expected swing: score <= .2; ambiguous labels masked")


class SpatialTrackingReward:
    def __init__(self, profile, num_envs, device, dt):
        if profile not in SERIES_PROFILES:
            raise ValueError("Unknown spatial reward profile")
        if not math.isfinite(dt) or dt <= 0 or abs(round(.5/dt)*dt-.5) > 1e-8:
            raise ValueError("Spatial progress requires an integral 0.5-second control window")
        self.profile, self.dt, self.window = profile, dt, round(.5/dt)
        self.weights = dict(BASE_WEIGHTS)
        if profile != SERIES_PROFILES[0]:
            self.weights.update(body_velocity=1.5, progress=1.)
        if profile == 'spatial-s4-v1':
            self.weights['support_score'] = 1.
        if profile == 'spatial-s5-v1':
            self.weights['joint'] = 0.
        self.contract = dict(version=profile, weights=copy.deepcopy(self.weights), scales=copy.deepcopy(SCALES),
            maximum_tracking_per_second=9.5, window_seconds=.5,
            normalization="9.5 * weighted sum / enabled weight sum",
            root_frame="shared reference world frame; no per-step recentering",
            timestamp="state and target at t; backward actual point velocities; no future target",
            support=copy.deepcopy(SUPPORT_CONTRACT) if self.weights['support_score'] else None)
        self.actual = torch.zeros((self.window, num_envs, 2), device=device)
        self.reference = torch.zeros_like(self.actual)
        self.steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.cursor = 0

    def reset(self, ids, position, reference_position):
        self.actual[:, ids] = position[:, :2]
        self.reference[:, ids] = reference_position[:, :2]
        self.steps[ids] = 0

    def step(self, state, reference, landmark_velocity, support=None):
        relative = ((state['landmarks'] - state['position'][:, None])
                    - (reference['landmarks'] - reference['root_position'][:, None]))
        root_angle = 2*(state['orientation']*reference['root_orientation']).sum(-1).abs().clamp(max=1).acos()
        body_angle = 2*(state['body_orientation']*reference['body_orientation']).sum(-1).abs().clamp(max=1).acos()
        errors = dict(joint=(state['q']-reference['joint_position']).square().mean(-1),
            height=(state['position'][:, 2]-reference['root_position'][:, 2]).square(),
            orientation=root_angle.square(), body_position=relative.square().sum(-1).mean(-1),
            feet_position=relative[:, [11, 15]].square().sum(-1).mean(-1),
            body_orientation=body_angle.square().mean(-1),
            velocity=(state['velocity']-reference['root_velocity'][:, :3]).square().sum(-1),
            body_velocity=(landmark_velocity-reference['landmark_velocity']).square().sum(-1).mean(-1),
            root_xy=(state['position'][:, :2]-reference['root_position'][:, :2]).square().sum(-1))
        parts = {key: (-value/SCALES[key]**2).exp() for key, value in errors.items()}
        zero = torch.zeros_like(root_angle)
        enabled = {key: torch.ones_like(zero) for key in errors}
        parts.update(progress=zero, progress_enabled=zero, support_score=zero, support_enabled=zero)
        enabled.update(progress=zero, support_score=zero)
        if self.weights['progress']:
            self.steps += 1
            delta = ((state['position'][:, :2]-self.actual[self.cursor])
                     - (reference['root_position'][:, :2]-self.reference[self.cursor]))
            errors['progress'] = delta.square().sum(-1)
            parts['progress'] = (-errors['progress']/SCALES['progress']**2).exp()
            enabled['progress'] = parts['progress_enabled'] = (self.steps >= self.window).float()
            self.actual[self.cursor] = state['position'][:, :2]
            self.reference[self.cursor] = reference['root_position'][:, :2]
            self.cursor = (self.cursor+1) % self.window
        if self.weights['support_score']:
            if support is None:
                raise ValueError("S4 requires measured ground contact forces and contact-point slip")
            wanted = reference['contacts'] >= .5
            eligible = torch.where(wanted,
                reference['contact_confidence'] >= SUPPORT_CONTRACT['confidence_threshold'],
                reference['contact_confidence'] <= SUPPORT_CONTRACT['swing_support_score_max'])
            measured = support['contact'].bool()
            slip = (-support['slip_speed'].square()/SUPPORT_CONTRACT['slip_scale_m_s']**2).exp()
            foot_score = torch.where(wanted, measured.float()*slip, (~measured).float())
            count = eligible.sum(-1)
            parts['support_score'] = (foot_score*eligible).sum(-1)/count.clamp(min=1)
            enabled['support_score'] = parts['support_enabled'] = (count > 0).float()
            parts['support_agreement'] = ((wanted == measured)*eligible).sum(-1)/count.clamp(min=1)
            parts['support_slip_m_s'] = (support['slip_speed']*measured).sum(-1)/measured.sum(-1).clamp(min=1)
            parts['support_normal_force_n'] = support['normal_force'].sum(-1)
        denominator = sum(weight*enabled[key] for key, weight in self.weights.items())
        tracking = torch.zeros_like(zero)
        for key, weight in self.weights.items():
            weighted = 9.5*weight*enabled[key]*parts[key]/denominator
            parts['weighted/'+key] = weighted
            tracking = tracking + weighted
        parts['enabled_weight_sum'] = denominator
        parts['tracking_per_second'] = tracking
        parts['root_xy_error_m'] = errors['root_xy'].sqrt()
        parts['root_speed_error_m_s'] = errors['velocity'].sqrt()
        parts['world_body_error_m'] = (state['landmarks']-reference['landmarks']).square().sum(-1).mean(-1).sqrt()
        return tracking, parts
