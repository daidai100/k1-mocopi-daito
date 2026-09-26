#!/usr/bin/env python3
"""Versioned development replay summary separating travel, pose, and safety.

The safe-progress count is a diagnostic, not a replacement for historical or
world/safety clean gates. Executed-prefix errors and full-duration scores remain
distinct so falling early cannot improve the reported trajectory result.
"""

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path


VERSION = 'k1-motion-axes-v1'
PROGRESS_RANGE = (.7, 1.3)


def mean(rows, getter):
    values = [getter(row) for row in rows]
    values = [float(value) for value in values if value is not None]
    if not values:
        return None
    if not all(math.isfinite(value) for value in values):
        raise ValueError('Nonfinite motion-axis metric')
    return sum(values) / len(values)


def safe_progress(row):
    safety = row['actuator_safety']
    ratio = row['root_progress_ratio']
    return bool(row['completed'] and row['self_collision_ticks'] == 0
        and safety['operating_speed_fraction'] == 0
        and safety['joint_limit_fraction'] == 0
        and (ratio is None or PROGRESS_RANGE[0] <= ratio <= PROGRESS_RANGE[1]))


def reduce(rows):
    complete = [row for row in rows if row['completed']]
    return dict(trials=len(rows), completed=len(complete),
        safe_completion=sum(row['completed'] and row['self_collision_ticks'] == 0
            and row['actuator_safety']['operating_speed_fraction'] == 0
            and row['actuator_safety']['joint_limit_fraction'] == 0 for row in rows),
        safe_progress=sum(safe_progress(row) for row in rows),
        historical_clean=sum(bool(row['clean_success']) for row in rows),
        world_safety_clean=sum(bool(row['absolute_motion_v1']['clean']) for row in rows),
        collision_trials=sum(row['self_collision_ticks'] > 0 for row in rows),
        fell=sum(bool(row['fell']) for row in rows),
        horizontal=dict(
            full_duration_xy_score_mean=mean(rows, lambda row: row['trajectory_v3']['full_reference_duration_xy_score']),
            root_xy_rmse_m_completed_mean=mean(complete, lambda row: row['trajectory_v3']['root_xy_rmse_m']),
            root_xy_p95_m_completed_mean=mean(complete, lambda row: row['trajectory_v3']['root_xy_p95_m']),
            root_velocity_rmse_m_s_completed_mean=mean(complete, lambda row: row['root_velocity_rmse_m_s']),
            root_progress_ratio_completed_mean=mean(complete, lambda row: row['root_progress_ratio']),
            root_orientation_error_rad_completed_mean=mean(complete, lambda row: row['mean_root_orientation_error_rad'])),
        joints=dict(joint_rmse_rad_completed_mean=mean(complete, lambda row: row['joint_rmse_rad']),
                    relative_body_rmse_m_completed_mean=mean(complete, lambda row: row['relative_body_rmse_m'])))


def summarize(panel_path, replay, output):
    panel_path, replay, output = Path(panel_path), Path(replay), Path(output)
    panel_bytes = panel_path.read_bytes()
    panel = json.loads(panel_bytes)
    summary = json.loads((replay/'summary.json').read_text())
    if (not panel or len({row['id'] for row in panel}) != len(panel)
            or summary['contract']['panel_sha256'] != hashlib.sha256(panel_bytes).hexdigest()
            or summary['execution_errors'] != 0 or summary['resets_during_trials'] != 0
            or summary['all']['trials'] != len(panel)):
        raise ValueError('Replay panel, execution, or reset contract differs')
    rows = []
    groups = defaultdict(list)
    for expected in panel:
        row = json.loads((replay/(expected['id']+'.json')).read_text())
        if (row['id'] != expected['id'] or row['family'] != expected['family']
                or row['capture_group'] != expected['capture_group']
                or row['resets_during_trial'] != 0
                or row['trajectory_v3']['version'] != 'k1-trajectory-fidelity-v3-screening'):
            raise ValueError('Trial identity or trajectory contract differs')
        rows.append(row)
        groups[expected.get('semantic_group', 'family:'+expected['family'])].append(row)
    all_rows = reduce(rows)
    if any(all_rows[key] != summary['all'][field] for key, field in (
            ('completed', 'completed'), ('historical_clean', 'clean'),
            ('collision_trials', 'collision_trials'), ('fell', 'fell'))):
        raise ValueError('Motion-axis recount differs from replay summary')
    result = dict(version=VERSION, contract=summary['contract'], panel=str(panel_path),
        replay=str(replay), screening_only=True, behaviorally_accepted=False,
        horizontal_scope='XY root path and velocity; executed-prefix errors are separate from full-duration score',
        joint_scope='22 K1 joints and root-relative body shape; completed trials only for error means',
        safe_progress_definition='complete; no self collision, operating-speed or joint-limit violation; '
            'projected horizontal progress 0.7 to 1.3 when reference displacement is at least 0.5 m',
        all=all_rows, by_semantic_group={name: reduce(group) for name, group in sorted(groups.items())})
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix+'.partial')
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    temporary.replace(output)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel', type=Path, required=True)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.panel, args.replay, args.output)
    print(json.dumps({'version': result['version'], 'all': result['all']}, indent=2))


if __name__ == '__main__':
    main()
