#!/usr/bin/env python3
"""Freeze a small original-timing training-only locomotion/retention library."""
# ruff: noqa: E402 - resolve the checkout before importing its package
import json
import os
from pathlib import Path
import re
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
os.environ['K1_MOTION_ROOT'] = str(ROOT)
from k1_motion.contracts import MotionClip
from k1_motion.math3d import rotation


def main():
    base = ROOT/'artifacts/easy-walk-campaign-20260923'
    base.mkdir(exist_ok=True)
    library = ROOT/'artifacts/next-policy-plan-20260922/clock-repaired-library-source-order-v2'
    rows = [json.loads(s) for s in (library/'index.jsonl').read_text().splitlines()]
    candidates = []
    for row in rows:
        if not re.match(r'^walk_ff_loop_\d+_R_(normal_pace|slow|very_slow)_\d+__', row.get('filename', '')):
            continue
        clip = MotionClip.load(library/row['reference_path'])
        value = clip.values
        p = value['root_position']
        path = np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1).sum()
        yaw = np.unwrap(rotation(value['root_orientation']).as_euler('xyz')[:, 2])
        speed = np.linalg.norm(value['root_velocity'][:, :2], axis=1)
        candidates.append(dict(id=row['id'], name=row['filename'], capture_group=row['capture_group'],
            seconds=float(clip.times[-1]-clip.times[0]), path_m=float(path),
            straightness=float(np.linalg.norm(p[-1, :2]-p[0, :2])/max(path, 1e-9)),
            yaw_range_rad=float(np.ptp(yaw)), speed_mean_m_s=float(speed.mean()),
            speed_p95_m_s=float(np.quantile(speed, .95)),
            joint_speed_p95_rad_s=float(np.quantile(np.abs(value['joint_velocity']), .95))))
    candidates.sort(key=lambda r: (r['speed_p95_m_s'], r['id']))
    (base/'walk-candidates.json').write_text(json.dumps(candidates, indent=2)+'\n')
    chosen, groups = [], set()
    for row in candidates:
        if (row['straightness'] < .95 or row['yaw_range_rad'] > .25
                or not 2 <= row['seconds'] <= 15 or row['path_m'] < .5
                or row['capture_group'] in groups):
            continue
        chosen.append(row)
        groups.add(row['capture_group'])
        if len(chosen) == 3:
            break
    if len(chosen) != 3:
        raise ValueError(f'Insufficient naturally easy straight walks: {chosen}')
    replay = ROOT/'artifacts/next-policy-plan-20260922/locomotion-audit/train-replay'
    retention = []
    for path in sorted(replay.glob('*.json')):
        row = json.loads(path.read_text())
        if row['clean_success'] and row['absolute_motion_v1']['clean']:
            retention.append(row['id'])
    walk_ids = [r['id'] for r in chosen]
    selected = [r for r in rows if r['id'] in walk_ids+retention]
    assert len(selected) == len(walk_ids)+len(retention)
    assert all(r['split'] == 'train' and not r.get('is_mirror', False) for r in selected)
    bundle = base/'bundle'
    if bundle.exists():
        raise ValueError('Refusing to overwrite prepared bundle')
    for d in ('library/clips', 'configs', 'manifests', 'scripts'):
        (bundle/d).mkdir(parents=True, exist_ok=True)
    checks = []
    for row in selected:
        source = library/row['reference_path']
        clip = MotionClip.load(source)
        assert clip.values['valid'].all() and np.allclose(np.diff(clip.times), .02)
        velocity = np.vstack((np.zeros((1, 3)), np.diff(clip.values['root_position'], axis=0)/.02))
        qvelocity = np.vstack((np.zeros((1, 22)), np.diff(clip.values['joint_position'], axis=0)/.02))
        checks.append(dict(id=row['id'], frames=len(clip.times), duration_s=float(clip.times[-1]-clip.times[0]),
            root_velocity_error=float(np.max(np.abs(velocity-clip.values['root_velocity'][:, :3]))),
            joint_velocity_error=float(np.max(np.abs(qvelocity-clip.values['joint_velocity'])))))
        assert checks[-1]['root_velocity_error'] < 1e-7 and checks[-1]['joint_velocity_error'] < 1e-7
        dest = f"clips/{row['id']}.npz"
        shutil.copyfile(source, bundle/'library'/dest)
        row['reference_path'] = dest
    (bundle/'library/index.jsonl').write_text(''.join(json.dumps(r, sort_keys=True)+'\n' for r in selected))
    manifest = dict(version='easy-walk-curriculum-v1', train_ids=[r['id'] for r in selected],
        locomotion_ids=walk_ids, retention_ids=retention,
        target_transition_weights={r['id']: .8/len(walk_ids) if r['id'] in walk_ids else .2/len(retention) for r in selected},
        reset_mix={'start': .5, 'failure_biased': .25, 'uniform': .25},
        sampling={'locomotion_transition_share': .8},
        timing='unaltered native 50Hz playback; no root translation removal or retiming',
        retention_selection='previous training-only initializer replays passing both historical and world+safety gates')
    (bundle/'manifests/easy-walk-v1.json').write_text(json.dumps(manifest, indent=2)+'\n')
    shutil.copyfile(ROOT/'configs/controller-pv-official80-guard03-v1.json', bundle/'configs/controller.json')
    (bundle/'configs/decomposed-v1.json').write_text(json.dumps({'weights': {}, 'scales': {}}, indent=2)+'\n')
    shutil.copyfile(ROOT/'artifacts/policy-improvement-20260922/pilots/guard_world/training/checkpoint-000125.pt', bundle/'initialize.pt')
    report = dict(selected_walks=chosen, retention_ids=retention, reference_checks=checks,
        selected_originals=len(selected), source_originals=len(rows), mirrors=0,
        criteria='ordinary forward loops; >=95% straight; yaw range <=0.25rad; 2-15s; >=0.5m travel; rank by low native p95 speed; distinct capture groups',
        physics_qualified=False, confirmation_panel_consumed=False)
    (base/'data-selection.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
