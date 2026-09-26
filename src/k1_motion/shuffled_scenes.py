"""Device-batched, train-only concatenation with bounded input discontinuities.

This composes command streams, not new geometry-qualified demonstrations.
Admission and actuator limits remain in force. No FK, disk reads, full-library
copy, or per-world Python candidate search occurs at a handoff.
"""
from collections import Counter
import math

import torch

from .observations import quat_apply, quat_inv, quat_mul
from .scene_transitions import transform_reference


FAMILIES = dict(motion=['gesture', 'other', 'bow', 'transition', 'object_interaction',
    'squat', 'dance', 'jump', 'kick', 'kneel', 'punch', 'avoidance', 'step_over', 'turn'],
    walk=['walk'], easy=['idle_stance', 'gesture', 'bow'], run=['run'])


def sequence_contract(settings, dt):
    defaults = dict(mode='shuffle', episode_seconds=120.,
        pattern=['motion', 'walk', 'motion', 'walk', 'easy', 'motion', 'walk', 'run'],
        pause_seconds=[.3, .8], candidate_count=128, max_joint_jump_rad=.6,
        max_height_jump_m=.08, max_heading_change_rad=.25, max_tilt_jump_rad=.35,
        random_initial_heading=True, random_start_slot=True)
    if not isinstance(settings, dict) or set(settings)-set(defaults):
        raise ValueError('Unknown shuffled scene settings')
    c = {**defaults, **settings}
    for key in ('episode_seconds', 'candidate_count', 'max_joint_jump_rad',
                'max_height_jump_m', 'max_heading_change_rad', 'max_tilt_jump_rad'):
        v = c[key]
        if isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or v < 0:
            raise ValueError('Shuffled scene values must be finite and nonnegative')
    if c['candidate_count'] < 1 or int(c['candidate_count']) != c['candidate_count']:
        raise ValueError('Positive integer candidate count required')
    if c['episode_seconds'] <= dt or c['max_joint_jump_rad'] <= 0:
        raise ValueError('Positive episode and jump bound required')
    if c['max_heading_change_rad'] > math.pi or c['max_tilt_jump_rad'] > math.pi:
        raise ValueError('Heading/tilt jump cannot exceed pi')
    pause = c['pause_seconds']
    if (not isinstance(pause, (list, tuple)) or len(pause) != 2
            or any(isinstance(v, bool) or not isinstance(v, (int, float))
                   or not math.isfinite(v) or v < 0 for v in pause)
            or pause[0] > pause[1] or pause[1]+dt >= c['episode_seconds']):
        raise ValueError('Invalid pause interval')
    if any(not math.isclose(v/dt, round(v/dt), abs_tol=1e-8)
           for v in (c['episode_seconds'], *pause)):
        raise ValueError('Scene durations must use the control clock')
    if (not isinstance(c['pattern'], list) or not c['pattern']
            or any(not isinstance(role, str) or role not in FAMILIES for role in c['pattern'])
            or not isinstance(c['random_initial_heading'], bool)
            or not isinstance(c['random_start_slot'], bool)):
        raise ValueError('Invalid scene pattern or heading flag')
    return dict(version='shuffled-scenes-v1', **c, control_dt=dt, families=FAMILIES,
        alignment='reference endpoint XY and heading, plus bounded random yaw increment; no body reanchor or vertical shift',
        sampling='fixed role pattern with random starting slot, independent uniform original draws per world; bounded batched candidate search',
        pauses='only endpoints with saved static-endpoint support feasibility; zero velocities while held',
        seams='bounded discontinuities in inputs; causal boundary derivatives; actuator slew limits still apply',
        termination='failure, finite episode cap, or no compatible distinct source in candidate batch',
        curriculum=False, physics_qualified=False)


def heading(q):
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))


def yaw_quaternion(angle):
    q = torch.zeros((*angle.shape, 4), dtype=angle.dtype, device=angle.device)
    q[..., 0], q[..., 3] = (angle/2).cos(), (angle/2).sin()
    return q


def angular_velocity(before, after, dt):
    delta = quat_mul(after, quat_inv(before))
    delta = torch.where(delta[:, :1] < 0, -delta, delta)
    length = delta[:, 1:].norm(dim=-1, keepdim=True)
    angle = 2*torch.atan2(length, delta[:, :1].clamp(min=0))
    return delta[:, 1:] * (angle/length.clamp(min=1e-8))/dt


class ShuffledScenes:
    def __init__(self, env, contract):
        rows, device = env.library.rows, env.device
        if (env.corruption or any(r.get('split') != 'train' or r.get('is_mirror', False) for r in rows)
                or len({r['id'] for r in rows}) != len(rows)):
            raise ValueError('Shuffled scenes require unique admitted training originals and no corruption')
        self.env, self.contract = env, contract
        self.episode_steps = round(contract['episode_seconds']/env.spec.control_dt)
        self.roles = list(dict.fromkeys(contract['pattern']))
        members = [[i for i, r in enumerate(rows) if r['family'] in FAMILIES[role]] for role in self.roles]
        for role, pool in zip(self.roles, members):
            if not pool:
                raise ValueError('Shuffled scene role has no originals: '+role)
        self.pool_sizes = torch.tensor([len(pool) for pool in members], device=device)
        self.pools = torch.zeros((len(members), max(map(len, members))), dtype=torch.long, device=device)
        for i, pool in enumerate(members):
            self.pools[i, :len(pool)] = torch.tensor(pool, device=device)
        self.pattern = torch.tensor([self.roles.index(role) for role in contract['pattern']], device=device)
        n = env.num_envs
        self.slots = torch.zeros(n, dtype=torch.long, device=device)
        self.yaw = yaw_quaternion(torch.zeros(n, device=device))
        self.shift = torch.zeros((n, 3), device=device)
        self.has_boundary = torch.zeros(n, dtype=torch.bool, device=device)
        self.episode_bridge_steps = torch.zeros(n, dtype=torch.long, device=device)
        ids = torch.arange(len(rows), device=device)
        self.first = env.library.frames(ids, torch.zeros_like(ids))
        self.last = env.library.frames(ids, env.library.lengths-1)
        self.start_tilt = quat_mul(quat_inv(yaw_quaternion(heading(self.first['root_orientation']))),
                                   self.first['root_orientation'])
        self.end_tilt = quat_mul(quat_inv(yaw_quaternion(heading(self.last['root_orientation']))),
                                 self.last['root_orientation'])
        self.can_hold = torch.tensor([r.get('standing_padding', {}).get('finish', {}).get('feasible') is True
                                      for r in rows], device=device)
        self.holds = {k: torch.zeros((n, *v.shape[1:]), dtype=v.dtype, device=device)
                      for k, v in self.first.items()}
        self.boundaries = {k: torch.zeros_like(v) for k, v in self.holds.items()}
        # Common metrics interface with the checked-bridge composer.
        self.rejections = Counter()
        self.cache_hits = self.cache_misses = 0

    def sample_starts(self, ids):
        if self.contract['random_start_slot']:
            self.slots[ids] = torch.randint(len(self.pattern), (len(ids),), device=self.env.device)
        role = self.pattern[self.slots[ids]]
        draw = (torch.rand(len(ids), device=self.env.device)*self.pool_sizes[role]).long()
        return self.pools[role, draw]

    def reset(self, ids):
        self.slots[ids] = 0
        angles = ((torch.rand(len(ids), device=self.env.device)*2-1)*math.pi
                  if self.contract['random_initial_heading'] else torch.zeros(len(ids), device=self.env.device))
        self.yaw[ids] = yaw_quaternion(angles)
        self.shift[ids] = 0
        self.has_boundary[ids] = False
        self.episode_bridge_steps[ids] = 0

    def frames(self, frames, keys=None):
        ref = transform_reference(self.env.library.frames(self.env.clips, frames, keys), self.yaw, self.shift)
        for key, v in ref.items():
            shape = (-1, *([1]*(v.ndim-1)))
            v = torch.where(((frames == 0) & self.has_boundary).reshape(shape), self.boundaries[key], v)
            ref[key] = torch.where((frames < 0).reshape(shape), self.holds[key], v)
        return ref

    def prepare(self, eligible):
        ids = eligible.nonzero(as_tuple=False).flatten()
        if not len(ids):
            return dict(ids=ids, source=ids, destination=ids, slots=ids)
        source = self.env.clips[ids]
        slots = (self.slots[ids]+1) % len(self.pattern)
        role = self.pattern[slots]
        draws = (torch.rand((len(ids), int(self.contract['candidate_count'])), device=self.env.device)
                 *self.pool_sizes[role, None]).long()
        candidates = self.pools[role[:, None], draws]
        valid = (candidates != source[:, None])
        valid &= (self.first['joint_position'][candidates]-self.last['joint_position'][source, None]).abs().amax(-1) <= self.contract['max_joint_jump_rad']
        valid &= (self.first['root_position'][candidates, 2]-self.last['root_position'][source, None, 2]).abs() <= self.contract['max_height_jump_m']
        cosine = (self.start_tilt[candidates]*self.end_tilt[source, None]).sum(-1).abs().clamp(max=1)
        valid &= cosine >= math.cos(self.contract['max_tilt_jump_rad']/2)
        available = valid.any(-1)
        destination = candidates.gather(1, valid.long().argmax(-1, keepdim=True)).squeeze(-1)
        return dict(ids=ids[available], source=source[available], destination=destination[available],
                    slots=slots[available])

    def pending_mask(self, pending):
        mask = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        mask[pending['ids']] = True
        return mask

    def commit(self, pending):
        ids = pending['ids']
        if not len(ids):
            return
        env, c = self.env, self.contract
        source, dest = pending['source'], pending['destination']
        before = transform_reference({k: v[source] for k, v in self.last.items()}, self.yaw[ids], self.shift[ids])
        change = (torch.rand(len(ids), device=env.device)*2-1)*c['max_heading_change_rad']
        q = yaw_quaternion(heading(before['root_orientation'])-heading(self.first['root_orientation'][dest])+change)
        shift = before['root_position']-quat_apply(q, self.first['root_position'][dest])
        shift[:, 2] = 0
        after = transform_reference({k: v[dest] for k, v in self.first.items()}, q, shift)
        dt = env.spec.control_dt
        after['joint_velocity'] = (after['joint_position']-before['joint_position'])/dt
        after['landmark_velocity'] = (after['landmarks']-before['landmarks'])/dt
        after['root_velocity'] = torch.cat(((after['root_position']-before['root_position'])/dt,
            angular_velocity(before['root_orientation'], after['root_orientation'], dt)), -1)
        for key in self.holds:
            self.boundaries[key][ids] = after[key]
            self.holds[key][ids] = before[key]
        for key in ('joint_velocity', 'root_velocity', 'landmark_velocity', 'age'):
            self.holds[key][ids] = 0
        lo, hi = [round(v/dt) for v in c['pause_seconds']]
        pause = torch.randint(lo, hi+1, (len(ids),), device=env.device)*self.can_hold[source]
        self.yaw[ids], self.shift[ids] = q, shift
        self.slots[ids] = pending['slots']
        self.has_boundary[ids] = True
        env.clips[ids] = dest
        env.frames[ids] = -pause
        env.command_frames[ids] = -pause
        env.scene_steps[ids] = 0
