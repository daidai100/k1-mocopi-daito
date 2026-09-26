"""Bounded, train-only scene composition without resetting physical state.

Original reference buffers remain immutable. A negative destination frame is a
synthetic bridge, followed by the complete destination recording from frame zero.
Bridges are checked at the physics clock and cached by ordered source pair.
These checks establish kinematic compatibility, not dynamic feasibility.
"""
from collections import Counter, OrderedDict
import math

import mujoco
import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.spatial.transform import Rotation, RotationSpline
import torch

from .math3d import heading, quaternion, rotation
from .observations import quat_apply, quat_inv, quat_mul
from .recovery_geometry import geometry_forward


def transition_contract(settings, dt):
    if isinstance(settings, dict) and settings.get('mode') == 'shuffle':
        from .shuffled_scenes import sequence_contract
        return sequence_contract(settings, dt)
    defaults = dict(blend_seconds=.6, episode_seconds=30., candidate_count=8, cache_size=512)
    if not isinstance(settings, dict) or set(settings)-set(defaults):
        raise ValueError('Scene transition settings contain unknown fields')
    values = {**defaults, **settings}
    for key, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError('Scene transition settings must be finite and positive')
        if key in ('candidate_count', 'cache_size') and int(value) != value:
            raise ValueError('Scene transition candidate/cache sizes must be integers')
    for key in ('blend_seconds', 'episode_seconds'):
        if not math.isclose(values[key]/dt, round(values[key]/dt), abs_tol=1e-8):
            raise ValueError('Scene transition durations must use the control clock')
    if values['blend_seconds'] < 2*dt or values['episode_seconds'] <= values['blend_seconds']+dt:
        raise ValueError('Scene transition episode must leave room for a bridge and source motion')
    return dict(version='scene-transitions-v1', **values, control_dt=dt,
        alignment='reference-to-reference yaw and XY only; never reanchor to the simulated body',
        bridge='Hermite position/joints and rotation spline; FK bodies; backward control-clock velocities',
        successors='weighted admitted training recordings; distinct source id; bounded candidate search',
        support='matching foot support; no knee support; upright endpoints',
        gates=dict(max_height_difference_m=.08, max_joint_difference_rad=.6,
            max_orientation_difference_rad=.35, self_penetration_tolerance_m=.0001,
            max_ground_penetration_m=.005, stance_slip_p95_m_s=.2),
        termination='existing failures, no compatible successor, or finite episode duration; no grace refresh',
        sampling='source segment lengths/exposure only; bridge ticks logged separately',
        physics_qualified=False)


def transform_reference(ref, yaw, shift):
    """Rigid world transform, including world velocities and body orientations."""
    result = dict(ref)
    for key in ('root_position', 'landmarks', 'landmark_velocity'):
        if key in ref:
            v = ref[key]
            q = yaw if v.ndim == 2 else yaw[:, None].expand(-1, v.shape[1], -1)
            result[key] = quat_apply(q, v)
            if key != 'landmark_velocity':
                result[key] += shift if v.ndim == 2 else shift[:, None]
    for key in ('root_orientation', 'body_orientation'):
        if key in ref:
            q = ref[key]
            result[key] = quat_mul(yaw if q.ndim == 2 else yaw[:, None].expand_as(q), q)
    if 'root_velocity' in ref:
        v = ref['root_velocity']
        result['root_velocity'] = torch.cat((quat_apply(yaw, v[:, :3]), quat_apply(yaw, v[:, 3:])), -1)
    return result


class SceneTransitions:
    def __init__(self, env, contract):
        if env.corruption:
            raise ValueError('Scene transitions require a buffer-aware corruption model; corruption is unsupported')
        if any(row.get('split') != 'train' for row in env.library.rows):
            raise ValueError('Scene transitions require an admitted training library')
        self.source_ids = [row.get('id', row['reference_path']) for row in env.library.rows]
        if len(set(self.source_ids)) != len(self.source_ids):
            raise ValueError('Scene transitions require unique recording identities')
        self.env, self.contract = env, contract
        self.steps = round(contract['blend_seconds']/env.spec.control_dt)
        self.episode_steps = round(contract['episode_seconds']/env.spec.control_dt)
        self.yaw = torch.zeros((env.num_envs, 4), device=env.device)
        self.yaw[:, 0] = 1
        self.shift = torch.zeros((env.num_envs, 3), device=env.device)
        self.has_bridge = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.episode_bridge_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.buffers = {key: torch.zeros((env.num_envs, self.steps+1, *v.shape[1:]),
                                       dtype=v.dtype, device=env.device)
                        for key, v in env.library.frames(torch.tensor([0], device=env.device),
                                                        torch.tensor([0], device=env.device)).items()}
        self.cache = OrderedDict()
        self.rejections = Counter()
        self.cache_hits = self.cache_misses = 0
        # Gather endpoint tensors once, rather than reading packed GPU data for
        # every candidate. All source trajectories remain in their shared cache.
        ids = torch.arange(len(env.library.rows), device=env.device)
        self.endpoints = [{k: v.cpu().numpy() for k, v in env.library.frames(ids, f).items()}
                          for f in (torch.zeros_like(ids), torch.ones_like(ids),
                                    env.library.lengths-2, env.library.lengths-1)]
        self.audit_data = mujoco.MjData(env.spec.model)

    def reset(self, ids):
        self.yaw[ids] = 0
        self.yaw[ids, 0] = 1
        self.shift[ids] = 0
        self.has_bridge[ids] = False
        self.episode_bridge_steps[ids] = 0

    def frames(self, frames, keys=None):
        env = self.env
        ref = env.library.frames(env.clips, frames, keys=keys)
        # Branch-free gather for all worlds avoids a GPU synchronization in each
        # command/reward/preview query. Buffers are bounded by worlds x bridge.
        index = (frames+self.steps).clamp(0, self.steps)
        ids = torch.arange(env.num_envs, device=env.device)
        for key, value in ref.items():
            # Frame zero after a bridge has a real predecessor. Its derivatives
            # must describe that last bridge tick, rather than a recording reset.
            mask = ((frames <= 0) & self.has_bridge).reshape(-1, *([1]*(value.ndim-1)))
            ref[key] = torch.where(mask, self.buffers[key][ids, index], value)
        return transform_reference(ref, self.yaw, self.shift)

    def _bridge(self, source, destination):
        key = source, destination
        if key in self.cache:
            self.cache_hits += 1
            self.cache.move_to_end(key)
            return self.cache[key]
        self.cache_misses += 1
        result, reason = self._build(source, destination)
        self.cache[key] = result, reason
        if len(self.cache) > self.contract['cache_size']:
            self.cache.popitem(last=False)
        return result, reason

    def _build(self, source, destination):
        env, spec = self.env, self.env.spec
        a = {k: v[source].astype(np.float64) for k, v in self.endpoints[3].items()}
        b = {k: v[destination].astype(np.float64) for k, v in self.endpoints[0].items()}
        gates = self.contract['gates']
        support = a['contacts'] > .5
        if (not a['valid'] or not b['valid'] or not support.any()
                or not np.array_equal(support, b['contacts'] > .5)
                or a['knee_contacts'].any() or b['knee_contacts'].any()
                or min(rotation(a['root_orientation']).as_matrix()[2,2],
                       rotation(b['root_orientation']).as_matrix()[2,2]) < .7):
            return None, 'support'
        if abs(a['root_position'][2]-b['root_position'][2]) > gates['max_height_difference_m']:
            return None, 'height'
        if np.max(abs(a['joint_position']-b['joint_position'])) > gates['max_joint_difference_rad']:
            return None, 'pose'
        yaw = Rotation.from_euler('z', heading(a['root_orientation'])-heading(b['root_orientation']))
        shift = a['root_position']-yaw.apply(b['root_position'])
        shift[2] = 0  # Floor and gravity are shared, never vertically realigned.
        q = torch.tensor(quaternion(yaw)[None], dtype=torch.float64)
        translation = torch.tensor(shift[None], dtype=torch.float64)
        transformed = transform_reference({k: torch.tensor(v[None]) for k,v in b.items()}, q, translation)
        b = {k: v[0].numpy() for k,v in transformed.items()}
        if (rotation(b['root_orientation'])*rotation(a['root_orientation']).inv()).magnitude() > gates['max_orientation_difference_rad']:
            return None, 'orientation'
        n, dt, duration = self.steps+1, spec.control_dt, self.contract['blend_seconds']
        times = np.arange(n)*dt
        values = {k: np.repeat(v[None], n, axis=0) for k,v in a.items()}
        for position, velocity in (('root_position', 'root_velocity'), ('joint_position', 'joint_velocity')):
            width = len(a[position])
            values[position] = CubicHermiteSpline([0, duration], [a[position], b[position]],
                [a[velocity][:width], b[velocity][:width]])(times)
        before = self.endpoints[2]['root_orientation'][source]
        after = quaternion(yaw*rotation(self.endpoints[1]['root_orientation'][destination]))
        orientations = rotation(np.stack((before, a['root_orientation'], b['root_orientation'], after)))
        values['root_orientation'] = quaternion(RotationSpline([-dt, 0, duration, duration+dt], orientations)(times))
        values['joint_velocity'][1:] = np.diff(values['joint_position'], axis=0)/dt
        values['root_velocity'][1:, :3] = np.diff(values['root_position'], axis=0)/dt
        values['root_velocity'][1:, 3:] = (rotation(values['root_orientation'][1:]) *
            rotation(values['root_orientation'][:-1]).inv()).as_rotvec()/dt
        from .actuation import command_velocity_limits
        speed = command_velocity_limits(spec, env.action_settings)
        if (not all(np.isfinite(v).all() for v in values.values())
                or np.any(values['joint_position'] < spec.limits[:,0]-1e-6)
                or np.any(values['joint_position'] > spec.limits[:,1]+1e-6)
                or np.any(abs(values['joint_velocity']) > speed+1e-6)):
            return None, 'joint_limits_or_speed'
        model, data = spec.model, self.audit_data
        feet = {model.body(f'{side}_ankle_roll_link').id for side in ('left','right')}
        poses = np.c_[values['root_position'], values['root_orientation'], values['joint_position']]
        delta = np.zeros(model.nv)
        # Audit the executed piecewise command path at every physics tick, not
        # merely the interpolation knots. Early exits are retained as rejects.
        for i, pose in enumerate(poses):
            if i:
                mujoco.mj_differentiatePos(model, delta, 1., poses[i-1], pose)
            for fraction in ([1.] if i == 0 else np.arange(1, spec.substeps+1)/spec.substeps):
                data.qpos[:] = poses[i-1] if i else pose
                if i:
                    mujoco.mj_integratePos(model, data.qpos, delta, float(fraction))
                geometry_forward(model, data)
                for contact in data.contact[:data.ncon]:
                    bodies = model.geom_bodyid[[contact.geom1, contact.geom2]]
                    if 0 in bodies:
                        if contact.dist < -gates['max_ground_penetration_m']:
                            return None, 'ground_penetration'
                        if contact.dist < -.0001 and int(max(bodies)) not in feet:
                            return None, 'nonfoot_support'
                    elif contact.dist < -gates['self_penetration_tolerance_m']:
                        return None, 'self_collision'
            values['landmarks'][i] = data.site_xpos[spec.site_ids]
            values['body_orientation'][i] = data.xquat[model.site_bodyid[spec.site_ids]]
        values['landmark_velocity'][1:] = np.diff(values['landmarks'], axis=0)/dt
        slip = np.linalg.norm(np.diff(values['landmarks'][:, [11,15], :2], axis=0)/dt, axis=-1)[:, support]
        if np.percentile(slip, 95) > gates['stance_slip_p95_m_s']:
            return None, 'stance_slip'
        values['age'][:] = 0
        values['valid'][:] = 1
        values['contact_confidence'][:] = np.minimum(a['contact_confidence'], b['contact_confidence'])
        for key in ('minimum_tracking_height', 'minimum_tracking_upright'):
            values[key][:] = max(a[key], b[key])
        # Store bridges in destination coordinates so one accumulated transform
        # applies both to negative bridge frames and the original next scene.
        local = transform_reference({k: torch.tensor(v, dtype=torch.float32) for k,v in values.items()},
            quat_inv(q.float()).expand(n,-1), torch.tensor(-yaw.inv().apply(shift), dtype=torch.float32).expand(n,-1))
        return (local, q[0].float(), translation[0].float()), None

    def prepare(self, eligible):
        ids = eligible.nonzero(as_tuple=False).flatten().cpu().tolist()
        if not ids:
            return []
        weights = self.env.library.weights.detach().cpu()
        sources = self.env.clips[ids].cpu().tolist()
        pending = []
        for index, source in zip(ids, sources):
            candidate_weights = weights.clone()
            candidate_weights[source] = 0
            count = min(int(self.contract['candidate_count']), int((candidate_weights > 0).sum()))
            if not count:
                continue
            candidates = torch.multinomial(candidate_weights, count, replacement=False).tolist()
            for destination in candidates:
                bridge, reason = self._bridge(source, destination)
                if bridge is not None:
                    pending.append((index, destination, bridge))
                    break
                self.rejections[reason] += 1
        return pending

    def commit(self, pending):
        env = self.env
        for index, destination, (bridge, yaw, shift) in pending:
            self.has_bridge[index] = True
            old_yaw = self.yaw[index].clone()
            self.shift[index] += quat_apply(old_yaw, shift.to(env.device))
            self.yaw[index] = quat_mul(old_yaw, yaw.to(env.device))
            self.yaw[index] /= self.yaw[index].norm()
            for key, value in bridge.items():
                self.buffers[key][index] = value.to(env.device)
            env.clips[index] = destination
            env.frames[index] = env.command_frames[index] = -self.steps
            env.scene_steps[index] = 0
        if pending and env.curriculum is not None:
            env.curriculum.reset_environments(torch.tensor([p[0] for p in pending], device=env.device))

    def pending_mask(self, pending):
        mask = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        if pending:
            mask[[p[0] for p in pending]] = True
        return mask
