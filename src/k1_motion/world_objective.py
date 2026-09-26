"""World-space imitation with small posture regularizers and measured safety."""
import copy
import math

import torch

from .observations import quat_apply

WORLD_PROFILES = ('world-body-v1', 'world-velocity-v1', 'world-decomposed-v1',
                  'world-pointwise-root-v1', 'world-capped-root-v1', 'causal-balanced-v1',
                  'survival-position-v1', 'survival-position-v2')


class WorldBodyTracking:
    def __init__(self, profile='world-body-v1', settings=None):
        if profile not in WORLD_PROFILES:
            raise ValueError('Unknown world tracking profile')
        if settings is not None and profile not in ('world-decomposed-v1', 'world-pointwise-root-v1', 'world-capped-root-v1', 'survival-position-v2'):
            raise ValueError('Settings require a configurable world reward profile')
        if profile == 'causal-balanced-v1':
            self.contract = causal_balanced_contract()
            return
        if profile in ('survival-position-v1', 'survival-position-v2'):
            self.contract = survival_position_contract(profile)
            if settings is not None:
                for group, overrides in settings.items():
                    if group not in ('weights', 'scales') or not isinstance(overrides, dict):
                        raise ValueError('Unknown survival reward settings')
                    target = self.contract[group]
                    for key, value in overrides.items():
                        if (key not in target or not isinstance(value, (int, float))
                                or isinstance(value, bool) or not math.isfinite(value)
                                or value < 0 or (group == 'scales' and value == 0)):
                            raise ValueError('Invalid survival reward weight or scale')
                        target[key] = float(value)
                maximum = sum(v for k, v in self.contract['weights'].items()
                              if k not in ('joint_error', 'root_xy_error'))
                if maximum <= 0:
                    raise ValueError('Positive tracking reward required')
                self.contract['maximum_tracking_per_second'] = maximum
            return
        if profile in ('world-decomposed-v1', 'world-pointwise-root-v1', 'world-capped-root-v1'):
            self.contract = decomposed_contract(settings, version=profile)
            return
        self.contract = dict(version='world-body-v1',
            weights=dict(world_position=9., joint_posture=.25, root_orientation=.25),
            scales=dict(world_position_m=.15, joint_posture_rad=.3, root_orientation_rad=.4),
            normalization='none; explicit per-second coefficients',
            maximum_tracking_per_second=9.5,
            position='mean per-point inverse quadratic; all 17 points in shared calibrated world frame',
            timestamp='current playback reference phase; no independent root recentering',
            posture='small regularizers resolve positional orientation ambiguities',
            progress_history=False)
        if profile == 'world-velocity-v1':
            self.contract['version'] = profile
            self.contract['weights']['root_velocity'] = 2.
            self.contract['scales']['root_velocity_m_s'] = .5
            self.contract['maximum_tracking_per_second'] = 11.5
            self.contract['root_velocity'] = (
                'Gaussian of current world XYZ root velocity squared error; '
                'independent of world position error; no coefficient renormalization')

    def reset(self, ids, position, reference_position):
        pass

    def step(self, state, reference, landmark_velocity, support=None):
        if self.contract['version'] in ('survival-position-v1', 'survival-position-v2'):
            return survival_position_step(self.contract, state, reference, landmark_velocity)
        if self.contract['version'] == 'causal-balanced-v1':
            return causal_balanced_step(self.contract, state, reference, landmark_velocity, support)
        if self.contract['version'] in ('world-decomposed-v1', 'world-pointwise-root-v1', 'world-capped-root-v1'):
            return decomposed_step(self.contract, state, reference)
        error = (state['landmarks']-reference['landmarks']).square().sum(-1)
        qerror = (state['q']-reference['joint_position']).square().mean(-1)
        angle = 2*(state['orientation']*reference['root_orientation']).sum(-1).abs().clamp(max=1).acos()
        parts = dict(world_position=(1/(1+error/.15**2)).mean(-1),
                     joint_posture=(-qerror/.3**2).exp(),
                     root_orientation=(-angle.square()/.4**2).exp(),
                     world_body_rmse_m=error.mean(-1).sqrt(),
                     world_body_max_error_m=error.max(-1).values.sqrt())
        if 'root_velocity' in self.contract['weights']:
            velocity_error = (state['velocity']-reference['root_velocity'][:, :3]).square().sum(-1)
            parts['root_velocity'] = (-velocity_error/.5**2).exp()
        total = 0.
        for name, weight in self.contract['weights'].items():
            parts['weighted/'+name] = weight*parts[name]
            total = total+parts['weighted/'+name]
        parts['tracking_per_second'] = total
        return total, parts


def survival_position_contract(profile='survival-position-v1'):
    weights = dict(root_xy_position=8., root_velocity=3., root_height=.5,
                   root_orientation=.5, body_relative=.25, body_velocity=.1, joint_error=.25)
    contract = dict(version=profile, weights=weights,
        scales=dict(root_xy_m=.5, root_velocity_m_s=.4, root_height_m=.08,
                    root_orientation_rad=.4, body_relative_m=.15, body_velocity_m_s=.75),
        maximum_tracking_per_second=sum(v for k, v in weights.items() if k != 'joint_error'),
        survival_per_second=2., fall_penalty=10., tracking_failure_penalty=5.,
        joint_error='negative arithmetic mean absolute joint-position error in radians over all 22 joints',
        integration='reward = dt * per-second terms; joint AUC = sum(dt * joint MAE); never feed accumulated AUC back as a reward',
        survival='constant credit per nonfailed physical tick, including supported pauses; no credit on failure tick',
        termination='one event penalty for fall or nonfall tracking failure; no event penalty at time limit',
        curriculum=False, normalization='fixed coefficients; no training-progress or episode-age reweighting',
        position='world root XY without recentering; independent velocity tracking',
        timestamp='post-step state and reference at the same phase', physics_qualified=False)
    if profile == 'survival-position-v2':
        weights['root_xy_error'] = .5
        contract['scales']['root_xy_softness_m'] = .5
        contract.update(tracking_termination=False, tracking_failure_penalty=0.,
            termination='falls/nonfinite state, finite scene limit or reference exhaustion; no pose or drift timeout',
            root_xy_error='negative sqrt(world XY error squared + softness squared) + softness; no dead zone, clipping, saturation or recentering',
            consistency='continuous time-integrated absolute-position cost; persistent offsets keep costing and recovery remains possible',
            progress_history=False)
    return contract


def survival_position_step(contract, state, reference, landmark_velocity):
    s = contract['scales']
    delta = state['position']-reference['root_position']
    relative = ((state['landmarks']-state['position'][:, None])
                -(reference['landmarks']-reference['root_position'][:, None]))
    velocity = (state['velocity']-reference['root_velocity'][:, :3]).square().sum(-1)
    body_velocity = (landmark_velocity-reference['landmark_velocity']).square().sum(-1)
    angle = 2*(state['orientation']*reference['root_orientation']).sum(-1).abs().clamp(max=1).acos()
    joint_mae = (state['q']-reference['joint_position']).abs().mean(-1)
    parts = dict(root_xy_position=1/(1+delta[:, :2].square().sum(-1)/s['root_xy_m']**2),
        root_velocity=1/(1+velocity/s['root_velocity_m_s']**2),
        root_height=1/(1+delta[:, 2].square()/s['root_height_m']**2),
        root_orientation=(-angle.square()/s['root_orientation_rad']**2).exp(),
        body_relative=(1/(1+relative.square().sum(-1)/s['body_relative_m']**2)).mean(-1),
        body_velocity=(1/(1+body_velocity/s['body_velocity_m_s']**2)).mean(-1),
        joint_error=-joint_mae, joint_mae_rad=joint_mae,
        root_xy_error_m=delta[:, :2].norm(dim=-1), root_speed_error_m_s=velocity.sqrt())
    if contract['version'] == 'survival-position-v2':
        softness = s['root_xy_softness_m']
        parts['root_xy_error'] = softness-(delta[:, :2].square().sum(-1)+softness**2).sqrt()
    total = torch.zeros_like(angle)
    for key, weight in contract['weights'].items():
        parts['weighted/'+key] = weight*parts[key]
        total += parts['weighted/'+key]
    parts['tracking_per_second'] = total
    if contract['version'] == 'survival-position-v2':
        parts['negative_tracking_fraction'] = (total < 0).float()
    return total, parts


def causal_balanced_contract():
    weights = dict(world_position=4., body_relative=2., root_velocity=3., body_velocity=1.,
        root_orientation=.5, root_height=.5, stance_still=.5, hold_still=.5)
    return dict(version='causal-balanced-v1', weights=weights,
        scales=dict(world_position_m=.20, body_relative_m=.15, root_velocity_m_s=.4,
                    body_velocity_m_s=.75, root_orientation_rad=.4, root_height_m=.08,
                    stance_slip_m_s=.1, hold_joint_speed_rad_s=.2, hold_root_speed_m_s=.05,
                    hold_root_angular_rad_s=.1),
        maximum_tracking_per_second=sum(weights.values()),
        normalization='none; bounded nonnegative per-second components, no negative tracking tail',
        world_position='mean per-point inverse quadratic in shared calibrated world frame',
        velocity='independent world root and landmark velocity tracking; no generic slow-motion bonus',
        stance='only confident reference stance feet with reference speed <=0.08 m/s; requires measured contact',
        hold='only reference root speed <=0.02 m/s, angular speed <=0.05 rad/s, all joint speeds <=0.05 rad/s and body-point speed <=0.03 m/s',
        timestamp='post-step state and reference at the same phase; backward derivatives only',
        observation='separate input contract caps preview at 300 ms; reward supplies no actor inputs',
        physics_qualified=False)


def causal_balanced_step(contract, state, reference, landmark_velocity, support):
    if support is None or any(k not in support for k in ('contact', 'slip_speed')):
        raise ValueError('Causal balanced reward requires measured foot support')
    s = contract['scales']
    delta = state['position']-reference['root_position']
    world = (state['landmarks']-reference['landmarks']).square().sum(-1)
    relative = ((state['landmarks']-state['position'][:, None])
                -(reference['landmarks']-reference['root_position'][:, None])).square().sum(-1)
    velocity = (state['velocity']-reference['root_velocity'][:, :3]).square().sum(-1)
    body_velocity = (landmark_velocity-reference['landmark_velocity']).square().sum(-1)
    angle = 2*(state['orientation']*reference['root_orientation']).sum(-1).abs().clamp(max=1).acos()
    stance = ((reference['contacts'] > .5) & (reference['contact_confidence'] >= .5)
              & (reference['landmark_velocity'][:, [11, 15]].norm(dim=-1) <= .08))
    stance_score = support['contact'].float()/(1+(support['slip_speed']/s['stance_slip_m_s']).square())
    hold = ((reference['root_velocity'][:, :3].norm(dim=-1) <= .02)
            & (reference['root_velocity'][:, 3:].norm(dim=-1) <= .05)
            & (reference['joint_velocity'].abs().max(-1).values <= .05)
            & (reference['landmark_velocity'].norm(dim=-1).max(-1).values <= .03))
    hold_error = (state['dq'].square().mean(-1)/s['hold_joint_speed_rad_s']**2
                  + state['velocity'].square().sum(-1)/s['hold_root_speed_m_s']**2
                  + state['omega'].square().sum(-1)/s['hold_root_angular_rad_s']**2)
    parts = dict(world_position=(1/(1+world/s['world_position_m']**2)).mean(-1),
        body_relative=(1/(1+relative/s['body_relative_m']**2)).mean(-1),
        root_velocity=1/(1+velocity/s['root_velocity_m_s']**2),
        body_velocity=(1/(1+body_velocity/s['body_velocity_m_s']**2)).mean(-1),
        root_orientation=(-angle.square()/s['root_orientation_rad']**2).exp(),
        root_height=1/(1+delta[:, 2].square()/s['root_height_m']**2),
        stance_still=(stance_score*stance).sum(-1)/stance.sum(-1).clamp(min=1),
        hold_still=hold.float()/(1+hold_error), hold_active_fraction=hold.float(),
        world_body_rmse_m=world.mean(-1).sqrt(), root_xy_error_m=delta[:, :2].norm(dim=-1),
        root_speed_error_m_s=velocity.sqrt())
    total = torch.zeros_like(angle)
    for key, weight in contract['weights'].items():
        parts['weighted/'+key] = weight*parts[key]
        total += parts['weighted/'+key]
    parts['tracking_per_second'] = total
    return total, parts


def decomposed_contract(settings=None, *, version='world-decomposed-v1'):
    if version not in ('world-decomposed-v1', 'world-pointwise-root-v1', 'world-capped-root-v1'):
        raise ValueError('Unknown decomposed world reward version')
    weights = dict(body_relative=3., root_velocity=3., yaw_rate=.5, heading=.25,
                   root_orientation=.5, root_height=.5, world_position=1.5,
                   joint_posture=.25, root_xy_cost=.25)
    scales = dict(body_relative_m=.12, root_velocity_m_s=.5, yaw_rate_rad_s=.5,
                  heading_rad=.4, root_orientation_rad=.4, root_height_m=.08,
                  world_position_m=.15, joint_posture_rad=.3, root_xy_m=.5)
    if version == 'world-capped-root-v1':
        weights.update(body_relative=3., root_xy_position=6., world_position=0., root_xy_cost=0.)
        scales['body_distance_cap_m'] = .30
    for group, overrides in (settings or {}).items():
        if group not in ('weights', 'scales') or not isinstance(overrides, dict):
            raise ValueError('Unknown decomposed reward settings')
        target = weights if group == 'weights' else scales
        for key, value in overrides.items():
            if (key not in target or not isinstance(value, (int, float))
                    or isinstance(value, bool) or not math.isfinite(value)
                    or value < 0 or (group == 'scales' and value == 0)):
                raise ValueError('Invalid decomposed reward weight or scale')
            target[key] = float(value)
    maximum = sum(v for k, v in weights.items() if k != 'root_xy_cost')
    if maximum <= 0:
        raise ValueError('Positive tracking reward required')
    return dict(version=version, weights=weights, scales=scales,
        body_distance_cap_m=scales.get('body_distance_cap_m'),
        maximum_tracking_per_second=maximum, normalization='none; explicit per-second coefficients',
        body_frame='Subtract each root; compare vectors in the same calibrated WORLD frame; no heading alignment',
        point_aggregation=('mean_of_capped_linear_distance_scores' if version == 'world-capped-root-v1' else 'mean_of_per_point_exponentials' if version == 'world-pointwise-root-v1'
                           else 'exponential_of_mean_squared_error'),
        root_velocity='world XYZ linear velocity only; no angular component',
        yaw_rate='world Z component of angular velocity; simulator body omega rotated into world',
        root_xy='negative Huber(norm(actual_xy-reference_xy)/scale), delta=1; no recentering',
        timestamp='physical state and reference at the same post-step phase; causal playback derivatives',
        early_termination_risk='Unbounded negative XY cost can favor termination at large lag; log negative share and do not promote without matched termination audit',
        progress_history=False)


def decomposed_step(contract, state, reference):
    scale = contract['scales']
    relative = ((state['landmarks']-state['position'][:, None])
                - (reference['landmarks']-reference['root_position'][:, None]))
    world_error = (state['landmarks']-reference['landmarks']).square().sum(-1)
    angle = 2*(state['orientation']*reference['root_orientation']).sum(-1).abs().clamp(max=1).acos()

    def yaw(q):
        w, x, y, z = q.unbind(-1)
        return torch.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))

    heading = yaw(state['orientation'])-yaw(reference['root_orientation'])
    heading = torch.atan2(heading.sin(), heading.cos())
    omega_world = quat_apply(state['orientation'], state['omega'])
    xy = (state['position'][:, :2]-reference['root_position'][:, :2]).norm(dim=-1)
    normalized_xy = xy/scale['root_xy_m']
    huber = torch.where(normalized_xy <= 1, .5*normalized_xy.square(), normalized_xy-.5)
    point_error = relative.square().sum(-1)
    errors = dict(body_relative=(point_error.mean(-1), 'body_relative_m'),
        root_velocity=((state['velocity']-reference['root_velocity'][:, :3]).square().sum(-1), 'root_velocity_m_s'),
        yaw_rate=((omega_world[:, 2]-reference['root_velocity'][:, 5]).square(), 'yaw_rate_rad_s'),
        heading=(heading.square(), 'heading_rad'), root_orientation=(angle.square(), 'root_orientation_rad'),
        root_height=((state['position'][:, 2]-reference['root_position'][:, 2]).square(), 'root_height_m'),
        joint_posture=((state['q']-reference['joint_position']).square().mean(-1), 'joint_posture_rad'))
    parts = {k: (-error/scale[name]**2).exp() for k, (error, name) in errors.items()}
    if contract['version'] == 'world-pointwise-root-v1':
        parts['body_relative'] = (-point_error/scale['body_relative_m']**2).exp().mean(-1)
    if contract['version'] == 'world-capped-root-v1':
        parts['body_relative'] = (1-point_error.sqrt().clamp(max=scale['body_distance_cap_m'])/scale['body_distance_cap_m']).mean(-1)
        parts['root_xy_position'] = 1/(1+normalized_xy.square())
    parts.update(world_position=(1/(1+world_error/scale['world_position_m']**2)).mean(-1),
                 root_xy_cost=-huber, root_xy_error_m=xy, body_relative_rmse_m=relative.square().sum(-1).mean(-1).sqrt(),
                 world_body_rmse_m=world_error.mean(-1).sqrt(), world_body_max_error_m=world_error.max(-1).values.sqrt())
    total = torch.zeros_like(xy)
    for key, weight in contract['weights'].items():
        parts['weighted/'+key] = weight*parts[key]
        total += parts['weighted/'+key]
    parts['tracking_per_second'] = total
    parts['negative_tracking_fraction'] = (total < 0).float()
    return total, parts


class SafetyObjective:
    """No slow-motion bonus: only measured operating-limit violations cost."""
    contract = dict(version='casual-safe-v1', per_second_weight=4., failure_penalty=1.,
        terms=['self_collision_control_tick', 'operating_speed_fraction', 'joint_limit_fraction'],
        speed_scope='actual joint speeds at every physics substep; configured operating limits',
        joint_scope='actual position beyond official joint range at every physics substep',
        regularizers=dict(action_change=.1, normalized_effort=.02),
        selection='Safety diagnostics remain separate from tracking; reward is not certification')

    def __init__(self):
        self.contract = copy.deepcopy(type(self).contract)

    def step(self, collision, metrics):
        required = ('operating_speed_fraction', 'joint_limit_fraction')
        if metrics is None or any(k not in metrics for k in required):
            raise ValueError('Safety objective requires measured physics-substep diagnostics')
        parts = {'safety/'+k: metrics[k] for k in required}
        parts['safety/self_collision'] = collision
        total = 4*(collision+metrics['operating_speed_fraction']+metrics['joint_limit_fraction'])
        parts['safety/cost_per_second'] = total
        return total, parts
