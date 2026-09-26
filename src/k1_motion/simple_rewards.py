"""Small, explicitly weighted causal tracking objectives for bounded screens."""
import copy

import torch

from .spatial_rewards import SpatialTrackingReward

V1_PROFILES = ('screen-s3-tail-v1', 'simple-track-v1', 'simple-track-fast-v1',
                   'simple-track-wide-v1', 'simple-track-no-joint-v1',
                   'simple-track-body-velocity-v1', 'simple-track-gaussian-v1')
SCREEN_PROFILES = V1_PROFILES + tuple(name.replace('-v1', '-v2')
                                    for name in V1_PROFILES if name.startswith('simple-'))


class SimpleTrackingReward:
    def __init__(self, profile, num_envs, device, dt):
        if profile not in SCREEN_PROFILES:
            raise ValueError('Unknown simple reward profile')
        self.parent = None
        if profile == 'screen-s3-tail-v1':
            self.parent = SpatialTrackingReward('spatial-s3-v1', num_envs, device, dt)
            self.contract = copy.deepcopy(self.parent.contract)
            self.contract.update(version=profile, root_xy_kernel='1/(1+error_squared/scale_squared)')
            return
        self.weights = dict(velocity=4., anchor=2., body_position=1.5, orientation=1.,
                            joint=.5, body_velocity=.5)
        self.scales = dict(velocity=.5, anchor_xy=.5, anchor_z=.08, body_position=.12,
                           orientation=.4, joint=.3, body_velocity=1.5)
        self.independent_height = profile.endswith('-v2')
        variant = profile.replace('-v2', '-v1')
        if self.independent_height:
            self.weights.update(anchor=1., height=1.)
            self.scales['height'] = .08
        self.anchor_kernel = 'inverse_quadratic'
        if variant == 'simple-track-fast-v1':
            self.weights.update(velocity=5., body_position=.5)
        elif variant == 'simple-track-wide-v1':
            self.scales['velocity'] = .75
        elif variant == 'simple-track-no-joint-v1':
            self.weights['joint'] = 0.
        elif variant == 'simple-track-body-velocity-v1':
            self.weights.update(body_velocity=1.5, body_position=.5)
        elif variant == 'simple-track-gaussian-v1':
            self.anchor_kernel = 'gaussian'
        self.contract = dict(version=profile, weights=copy.deepcopy(self.weights),
            scales=copy.deepcopy(self.scales), maximum_tracking_per_second=sum(self.weights.values()),
            normalization='none; coefficients are per-second, angle removal does not reweight other terms',
            anchor_kernel=self.anchor_kernel,
            independent_height=self.independent_height,
            anchor_error=('XY squared norm/xy_scale^2 in shared calibrated frame; separate Gaussian height'
                          if self.independent_height else
                          'XY squared norm/xy_scale^2 + z_error^2/z_scale^2 in shared calibrated frame'),
            timestamp='state/reference at t; backward body-point velocity; no future target',
            progress_history=False, support_reward=False)

    def reset(self, ids, position, reference_position):
        if self.parent is not None:
            self.parent.reset(ids, position, reference_position)

    def step(self, state, reference, landmark_velocity, support=None):
        if self.parent is not None:
            total, parts = self.parent.step(state, reference, landmark_velocity, support)
            old = parts['weighted/root_xy']
            error = (state['position'][:, :2]-reference['root_position'][:, :2]).square().sum(-1)
            parts['root_xy'] = 1/(1+error/.25**2)
            parts['weighted/root_xy'] = 9.5*2*parts['root_xy']/parts['enabled_weight_sum']
            total = total-old+parts['weighted/root_xy']
            parts['tracking_per_second'] = total
            return total, parts
        relative = ((state['landmarks']-state['position'][:, None])
                    -(reference['landmarks']-reference['root_position'][:, None]))
        angle = 2*(state['orientation']*reference['root_orientation']).sum(-1).abs().clamp(max=1).acos()
        difference = state['position']-reference['root_position']
        anchor = difference[:, :2].square().sum(-1)/self.scales['anchor_xy']**2
        if not self.independent_height:
            anchor = anchor+difference[:, 2].square()/self.scales['anchor_z']**2
        errors = dict(velocity=(state['velocity']-reference['root_velocity'][:, :3]).square().sum(-1),
            body_position=relative.square().sum(-1).mean(-1), orientation=angle.square(),
            joint=(state['q']-reference['joint_position']).square().mean(-1),
            body_velocity=(landmark_velocity-reference['landmark_velocity']).square().sum(-1).mean(-1))
        if self.independent_height:
            errors['height'] = difference[:, 2].square()
        parts = {key: (-value/self.scales[key]**2).exp() for key, value in errors.items()}
        parts['anchor'] = 1/(1+anchor) if self.anchor_kernel == 'inverse_quadratic' else (-anchor).exp()
        total = torch.zeros_like(angle)
        for key, weight in self.weights.items():
            parts['weighted/'+key] = weight*parts[key]
            total += parts['weighted/'+key]
        parts.update(tracking_per_second=total,
                     root_xy_error_m=difference[:, :2].norm(dim=-1),
                     root_speed_error_m_s=errors['velocity'].sqrt(),
                     world_body_error_m=(state['landmarks']-reference['landmarks']).square().sum(-1).mean(-1).sqrt())
        return total, parts
