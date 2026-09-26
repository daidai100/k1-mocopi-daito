"""Versioned static endpoint holds; preserve every original training recording.

The COM/support test is a static geometric condition, not dynamic qualification.
Original trajectories, admission receipts and held-out recordings are immutable.
"""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import copy
import hashlib
from itertools import product
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import ConvexHull, QhullError

from .contracts import MotionClip
from .recovery_geometry import geometry_forward, recovery_model
from .reference_admission import reference_rejections, take_family
from .reference_velocity_clock import playback_derivatives, velocity_errors

VERSION = 'static-endpoint-holds-v1'
FLOOR_TOLERANCE_M = .001


def support_margin(point, corners):
    """Signed distance to a convex support boundary, None for line/point support."""
    points = np.unique(np.asarray(corners, dtype=float).reshape(-1, 2), axis=0)
    if len(points) < 3:
        return None
    try:
        hull = ConvexHull(points)
    except QhullError:
        return None
    equations = hull.equations
    return float(np.min(-(equations[:, :2] @ np.asarray(point) + equations[:, 2])
                        / np.linalg.norm(equations[:, :2], axis=1)))


def endpoint_support(clip, index, robot):
    model, data = robot.model, mujoco.MjData(robot.model)
    values = clip.values
    data.qpos[:] = np.r_[values['root_position'][index], values['root_orientation'][index],
                        values['joint_position'][index]]
    geometry_forward(model, data)
    feet = [model.body(side+'_ankle_roll_link').id for side in ('left', 'right')]
    signs = np.asarray(list(product((-1., 1.), repeat=3)))
    points, minima, supporting = [], [], []
    for foot in feet:
        corners = []
        for geom in range(model.ngeom):
            if model.geom_bodyid[geom] != foot or not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
                continue
            if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_BOX:
                raise ValueError('Static support requires the K1 foot-box geometry contract')
            corners.extend(data.geom_xpos[geom] + (signs*model.geom_size[geom])
                           @ data.geom_xmat[geom].reshape(3, 3).T)
        corners = np.asarray(corners)
        if corners.size == 0:
            raise ValueError('Missing collidable foot geometry')
        minima.append(float(corners[:, 2].min()))
        near = corners[np.abs(corners[:, 2]) <= FLOOR_TOLERANCE_M+1e-9]
        supporting.append(bool(len(near)))
        points.extend(near[:, :2])
    com = data.subtree_com[model.body('trunk').id].copy()
    margin = support_margin(com[:2], points)
    reasons = []
    if margin is None:
        reasons.append('no_ground_support_polygon')
    elif margin < -1e-9:
        reasons.append('com_outside_support')
    if min(minima) < -FLOOR_TOLERANCE_M-1e-9:
        reasons.append('foot_penetration')
    for contact in data.contact[:data.ncon]:
        bodies = model.geom_bodyid[[contact.geom1, contact.geom2]]
        if 0 not in bodies and contact.dist < -.0001:
            reasons.append('self_penetration')
        if 0 in bodies and contact.dist <= FLOOR_TOLERANCE_M and not any(b in feet for b in bodies):
            reasons.append('nonfoot_ground_support')
    q = values['joint_position'][index]
    if np.any(q < robot.limits[:, 0]-1e-7) or np.any(q > robot.limits[:, 1]+1e-7):
        reasons.append('joint_limits')
    return dict(feasible=not reasons, reasons=sorted(set(reasons)), com_world_m=com.tolist(),
        com_margin_m=margin, supporting_feet=supporting,
        support_points_xy_m=np.asarray(points).reshape(-1, 2).tolist(),
        foot_minimum_z_m=minima, floor_tolerance_m=FLOOR_TOLERANCE_M,
        criterion='whole-robot mass COM projected inside convex hull of near-floor foot-box corners',
        physics_qualified=False)


def _moving(clip, first, count):
    # The first causal derivative is zero even when frame one is moving.
    joint, root = playback_derivatives(clip)
    segment = slice(1, min(len(clip.times), count+1)) if first else slice(max(1, len(clip.times)-count), None)
    speeds = dict(joint_rad_s=float(np.abs(joint[segment]).max()),
        root_linear_m_s=float(np.linalg.norm(root[segment, :3], axis=-1).max()),
        root_angular_rad_s=float(np.linalg.norm(root[segment, 3:], axis=-1).max()))
    return (speeds['joint_rad_s'] > .05 or speeds['root_linear_m_s'] > .01
            or speeds['root_angular_rad_s'] > .05), speeds


def pad_clip(clip, robot, hold_s=.3):
    if 'standing_padding' in clip.metadata:
        raise ValueError('Clip is already padded/audited')
    dt = robot.control_dt
    ticks = round(hold_s/dt)
    if not np.isfinite(hold_s) or ticks < 1 or abs(ticks*dt-hold_s) > 1e-10:
        raise ValueError('Hold must be a positive integer number of control ticks')
    if (clip.metadata.get('model_signature') != robot.signature or not clip.values['valid'].all()
            or not np.allclose(np.diff(clip.times), dt, rtol=0, atol=1e-10)):
        raise ValueError('Padding requires valid robot-matched control-clock references')
    if max(velocity_errors(clip).values()) > 1e-8:
        raise ValueError('Original playback velocity clock is inconsistent')
    start, finish = endpoint_support(clip, 0, robot), endpoint_support(clip, -1, robot)
    for first, endpoint in ((True, start), (False, finish)):
        endpoint['moving'], endpoint['endpoint_speed'] = _moving(clip, first, ticks)
    leading = ticks if start['feasible'] and start['moving'] else 0
    trailing = ticks if finish['feasible'] and finish['moving'] else 0
    audit = dict(version=VERSION, hold_s=hold_s, control_dt=dt,
        original_frames=len(clip.times), original_duration_s=float(clip.times[-1]-clip.times[0]),
        leading_frames=leading, trailing_frames=trailing, start=start, finish=finish,
        original_pose_arrays_unchanged=True, original_source_clock_unchanged=True,
        derivative_clock='causal backward difference of padded playback poses',
        scope='static endpoint geometry; original moving command path and admission remain unchanged',
        physics_qualified=False)
    metadata = copy.deepcopy(clip.metadata)
    metadata.update(standing_padding=audit, frames=len(clip.times)+leading+trailing,
        valid_ticks=len(clip.times)+leading+trailing,
        retargeted_seconds=audit['original_duration_s']+(leading+trailing)*dt)
    indices = np.r_[np.zeros(leading, int), np.arange(len(clip.times)),
                    np.full(trailing, len(clip.times)-1, int)]
    values = {k: v[indices].copy() for k, v in clip.values.items()}
    for segment, endpoint in ((slice(0, leading), start),
                              (slice(len(indices)-trailing, len(indices)), finish)):
        values['contacts'][segment] = endpoint['supporting_feet']
        values['contact_confidence'][segment] = 1.
    if 'knee_contacts' in metadata:
        metadata['knee_contacts'] = np.asarray(metadata['knee_contacts'])[indices].tolist()
    # Apply the same arithmetic to original playback and arrival clocks; a new
    # arange can otherwise put an arrival a few ULPs after its playback tick.
    times = np.r_[clip.times[0]+np.arange(leading)*dt, clip.times+leading*dt,
                  clip.times[-1]+(leading+np.arange(1, trailing+1))*dt]
    source = None if clip.source_times is None else clip.source_times[indices]
    received = None
    if clip.received_times is not None:
        received = np.r_[times[:leading], clip.received_times+leading*dt,
                         times[len(indices)-trailing:] if trailing else []]
    result = MotionClip(times, values, metadata, source, received)
    joint, root = playback_derivatives(result)
    result = MotionClip(times, {**result.values, 'joint_velocity': joint, 'root_velocity': root},
                        metadata, source, received)
    return result, audit


def _prepare_one(job):
    row, source, staging, destination = job
    path = (Path(source)/row['reference_path']).resolve(strict=True)
    clip = MotionClip.load(path)
    for key in ('id', 'capture_group', 'split', 'is_mirror', 'model_signature'):
        if clip.metadata.get(key) != row.get(key):
            raise ValueError('Manifest/payload identity differs: '+row['id']+' '+key)
    if len(clip.times) != row['frames']:
        raise ValueError('Manifest/payload frame count differs: '+row['id'])
    result, audit = pad_clip(clip, recovery_model('audit'))
    output_path = Path(staging)/'clips'/(row['id']+'.npz')
    if audit['leading_frames'] or audit['trailing_frames']:
        result.save(output_path)
        loaded = MotionClip.load(output_path)
        if max(velocity_errors(loaded).values()) > 1e-8:
            raise ValueError('Saved padding derivative mismatch')
        start = audit['leading_frames']
        for key in ('root_position', 'root_orientation', 'joint_position', 'landmarks', 'valid'):
            if not np.array_equal(loaded.values[key][start:start+len(clip.times)], clip.values[key]):
                raise ValueError('Padding changed original poses')
        published = str(Path(destination)/'clips'/output_path.name)
    else:
        published = str(path)
    updated = {**row, 'reference_path': published, 'frames': len(result.times),
        'valid_ticks': len(result.times), 'retargeted_seconds': float(result.times[-1]-result.times[0]),
        'standing_padding': audit, 'unpadded_reference_path': str(path)}
    return updated


def prepare_library(source, output, *, workers=8):
    source, output = Path(source).resolve(strict=True), Path(output).absolute()
    if output.exists() or output.with_name(output.name+'.partial').exists():
        raise FileExistsError('Refusing to overwrite a prepared library or partial run')
    if workers < 1:
        raise ValueError('Positive worker count required')
    payload = (source/'index.jsonl').read_bytes()
    rows = [json.loads(line) for line in payload.splitlines()]
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Unique nonempty training originals required')
    for row in rows:
        if (row.get('split') != 'train' or row.get('is_mirror') is not False
                or row.get('training_eligible') is not True or reference_rejections(row)
                or Path(row['id']).name != row['id'] or row['id'] in ('.', '..')):
            raise ValueError('Ineligible training original: '+row['id'])
    staging = output.with_name(output.name+'.partial')
    (staging/'clips').mkdir(parents=True)
    jobs = [(r, str(source), str(staging), str(output)) for r in rows]
    if workers == 1:
        padded = list(map(_prepare_one, jobs))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            padded = list(pool.map(_prepare_one, jobs, chunksize=16))
    # Source duration, not artificial padding, decides inclusion.
    selected = [r for r in padded if r['standing_padding']['original_duration_s'] > 10.+1e-9]
    if not selected:
        raise ValueError('No original training recordings longer than ten seconds')
    for name, subset in (('all-padded', padded), ('library', selected)):
        (staging/name).mkdir()
        (staging/name/'index.jsonl').write_text(''.join(json.dumps(r, sort_keys=True)+'\n' for r in subset))
    def census(subset):
        return dict(originals=len(subset), by_family=dict(sorted(Counter(r['family'] for r in subset).items())),
            by_source=dict(sorted(Counter(r['dataset'] for r in subset).items())),
            related_take_families=len({take_family(r['capture_group']) for r in subset}),
            original_hours=sum(r['standing_padding']['original_duration_s'] for r in subset)/3600,
            leading_holds=sum(r['standing_padding']['leading_frames'] > 0 for r in subset),
            trailing_holds=sum(r['standing_padding']['trailing_frames'] > 0 for r in subset),
            unpadded=sum(not (r['standing_padding']['leading_frames'] or r['standing_padding']['trailing_frames'])
                         for r in subset),
            endpoint_reasons=dict(Counter(reason for r in subset for side in ('start', 'finish')
                for reason in r['standing_padding'][side]['reasons'])))
    report = dict(version=VERSION, source_library=str(source),
        source_manifest_sha256=hashlib.sha256(payload).hexdigest(),
        training_manifest_sha256=hashlib.sha256((staging/'library/index.jsonl').read_bytes()).hexdigest(),
        all_originals=len(padded), training_originals=len(selected),
        selection='all admitted train originals with unpadded playback duration strictly greater than 10 seconds',
        all=census(padded), training=census(selected), invalid_ticks=0,
        heldout_modified=False, mirrors=0, physics_qualified=False)
    (staging/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    staging.rename(output)
    return report
