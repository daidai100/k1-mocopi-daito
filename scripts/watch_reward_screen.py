#!/usr/bin/env python3
"""Mirror bounded campaigns and replay an arbitrary frozen diagnostic panel."""
import argparse
from collections import defaultdict
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import sys
import time

TERMINAL = {'completed', 'failed', 'interrupted', 'cancelled', 'completed_with_failures'}
NUMBERED = re.compile(r'checkpoint-[0-9]+\.pt')


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def load(path):
    return json.loads(Path(path).read_text())


def checked_training_directory(campaign_root, name, supplied):
    root, source = PurePosixPath(str(campaign_root)), PurePosixPath(str(supplied))
    if (not re.fullmatch(r'[A-Za-z0-9_-]+', name) or not root.is_absolute()
            or '..' in root.parts or '..' in source.parts or source != root/name/'training'):
        raise ValueError('Unexpected training directory outside the declared campaign/run root')
    return source


def capture_terminal_checkpoint(source, target):
    source, target = Path(source), Path(target)
    if target.exists():
        if target.is_symlink():
            raise ValueError('Terminal checkpoint cannot be a symlink')
        return
    if not source.is_file():
        return
    if source.is_symlink():
        raise ValueError('Terminal source checkpoint cannot be a symlink')
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix('.incoming')
    shutil.copy2(source, temporary)
    temporary.replace(target)


def checkpoint_candidates(training, terminal):
    training = Path(training)
    candidates = sorted(p for p in training.glob('checkpoint-*.pt') if NUMBERED.fullmatch(p.name))
    terminal_path = training/'checkpoint-terminal.pt'
    if terminal and terminal_path.is_file():
        # Successful final updates already have an immutable numbered snapshot.
        report = load(training/'report.json') if (training/'report.json').exists() else {}
        iteration = report.get('last_metrics', {}).get('iteration')
        if iteration is None or not (training/f'checkpoint-{iteration:06d}.pt').is_file():
            candidates.append(terminal_path)
    if any(p.is_symlink() or p.resolve().parent != training.resolve() for p in candidates):
        raise ValueError('Checkpoint path escaped the declared training directory')
    return candidates


def refresh_campaigns(args, bundle):
    """Preserve numbered milestones and copy current weights only after termination."""
    host = getattr(args, 'server_host', 'server-wired')
    remote_root = str(args.server_root)
    # Validate the root independently before constructing a remote shell command.
    checked_training_directory(remote_root, 'probe', str(PurePosixPath(remote_root)/'probe/training'))
    reply = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=7', host,
        'cat -- '+shlex.quote(remote_root+'/status.json')], check=True, capture_output=True, text=True, timeout=20)
    server, desktop = json.loads(reply.stdout), load(args.desktop_root/'status.json')
    for label, state in [('server', server), ('desktop', desktop)]:
        if state.get('bundle') != bundle or state.get('host') != label:
            raise ValueError('Campaign bundle or host contract differs')
    combined = {'bundle': bundle, 'updated_unix': time.time(), 'runs': {},
                'host_phases': {'server': server['phase'], 'desktop': desktop['phase']},
                'behaviorally_accepted': False, 'hardware_verified': False, 'automatically_promoted': False}
    for name, run in server['runs'].items():
        source = checked_training_directory(remote_root, name, run['training_directory'])
        destination = args.root/'server-mirror'/name/'training'
        destination.mkdir(parents=True, exist_ok=True)
        if destination.resolve() != (args.root/'server-mirror'/name/'training').absolute():
            raise ValueError('Local mirror path contains a symlink')
        if run['phase'] not in {'queued', 'planned', 'cancelled'}:
            subprocess.run(['rsync', '-a', '--ignore-existing', '--protect-args', '--timeout=30',
                '--include=checkpoint-[0-9]*.pt', '--include=config.json', '--include=report.json', '--exclude=*',
                host+':'+str(source)+'/', str(destination)+'/'], check=True, timeout=120)
        if run['phase'] in TERMINAL and run['phase'] != 'cancelled':
            target = destination/'checkpoint-terminal.pt'
            if not target.exists():
                incoming = target.with_suffix('.incoming')
                result = subprocess.run(['rsync', '-a', '--protect-args', '--timeout=30',
                    host+':'+str(source)+'/checkpoint.pt', str(incoming)], timeout=120, capture_output=True, text=True)
                if result.returncode == 0:
                    incoming.replace(target)
                elif run['phase'] == 'completed':
                    raise RuntimeError('Completed remote run has no readable terminal checkpoint: '+result.stderr[-1000:])
        local = copy.deepcopy(run)
        local.update(host='server', remote_training_directory=str(source), training_directory=str(destination))
        combined['runs']['server/'+name] = local
    for name, run in desktop['runs'].items():
        source = Path(checked_training_directory(args.desktop_root, name, run['training_directory']))
        if source.resolve() != source.absolute():
            raise ValueError('Desktop training path contains a symlink')
        if run['phase'] in TERMINAL:
            capture_terminal_checkpoint(source/'checkpoint.pt', source/'checkpoint-terminal.pt')
            if run['phase'] == 'completed' and not (source/'checkpoint-terminal.pt').exists():
                raise ValueError('Completed desktop run has no terminal checkpoint')
        combined['runs']['desktop/'+name] = {**run, 'host': 'desktop', 'training_directory': str(source)}
    phases = [server['phase'], desktop['phase']]
    combined['phase'] = ('completed' if all(p == 'completed' for p in phases) else
                         'completed_with_failures' if all(p in TERMINAL for p in phases) else 'running')
    write_json(args.root/'server-status.json', server)
    write_json(args.root/'all-hosts/status.json', combined)
    return combined


def aggregate_replay(panel, trials, summary, panel_hash, revision):
    expected = {r['id']: r for r in panel}
    observed = {r['id']: r for r in trials}
    if len(expected) != len(panel) or len(observed) != len(trials) or set(expected) != set(observed):
        raise ValueError('Missing, duplicate or unexpected replay trial')
    contract = summary.get('contract', {})
    if contract.get('panel_sha256') != panel_hash or contract.get('source_revision') != revision:
        raise ValueError('Replay source/panel contract differs')
    if summary.get('execution_errors') != 0 or summary.get('resets_during_trials') != 0:
        raise ValueError('Replay has execution errors or trial resets')
    world_available = any('absolute_motion_v1' in row for row in trials)
    world_means = ('world_body_rmse_m', 'world_body_p95_m', 'full_reference_duration_world_score')
    safety_fields = ('operating_speed_fraction', 'operating_speed_max_ratio',
                     'nominal_speed_fraction', 'nominal_speed_max_ratio',
                     'joint_limit_fraction', 'joint_limit_max_error', 'torque_saturation',
                     'operating_speed_excess_squared')
    groups = defaultdict(list)
    for key, row in observed.items():
        for field in ['capture_group', 'family']:
            if row.get(field) != expected[key].get(field):
                raise ValueError('Replay trial identity differs from the panel')
        if row.get('resets_during_trial') != 0:
            raise ValueError('Replay trial contains resets')
        v3 = row.get('trajectory_v3', {})
        if v3.get('version') != 'k1-trajectory-fidelity-v3-screening' or v3.get('screening_only') is not True:
            raise ValueError('Missing versioned trajectory screening trial evidence')
        for field in ['full_reference_duration_xy_score', 'root_xy_rmse_m', 'survived_fraction']:
            if not isinstance(v3.get(field), (int, float)) or not math.isfinite(v3[field]):
                raise ValueError('Trajectory diagnostics must be finite')
        if not 0 <= v3['full_reference_duration_xy_score'] <= 1+1e-9 or not 0 <= v3['survived_fraction'] <= 1+1e-9:
            raise ValueError('Trajectory score or survival fraction outside declared bounds')
        if v3['clean_v3'] and not (row['clean_success'] and row['completed']):
            raise ValueError('Screening clean cannot promote a failed historical clean trial')
        if world_available:
            absolute = row.get('absolute_motion_v1', {})
            safety = row.get('actuator_safety', {})
            if (absolute.get('version') != 'world-position-safety-v1'
                    or absolute.get('world_rmse_limit_m') != .15 or absolute.get('world_p95_limit_m') != .30):
                raise ValueError('Missing or changed absolute world/safety contract')
            for container, fields in ((row, world_means), (safety, safety_fields)):
                if any(not isinstance(container.get(field), (int, float))
                       or not math.isfinite(container[field]) or container[field] < 0 for field in fields):
                    raise ValueError('World/safety diagnostics must be complete, finite and nonnegative')
            if (row['full_reference_duration_world_score'] > 1+1e-9
                    or any(safety[field] > 1 for field in ('operating_speed_fraction',
                            'nominal_speed_fraction', 'joint_limit_fraction', 'torque_saturation'))):
                raise ValueError('World/safety score or fraction outside declared bounds')
            if (not isinstance(safety.get('sample_count'), int) or safety['sample_count'] <= 0
                    or not isinstance(safety.get('sample_period_s'), (int, float))
                    or not math.isfinite(safety['sample_period_s']) or safety['sample_period_s'] <= 0
                    or not isinstance(safety.get('contract'), dict)):
                raise ValueError('World/safety diagnostics lack substep sample provenance')
            if absolute.get('full_duration_completed') != bool(row['completed']):
                raise ValueError('Absolute completion differs from trial completion')
            absolute_clean = bool(row['completed'] and row['self_collision_ticks'] == 0
                and row['world_body_rmse_m'] <= .15 and row['world_body_p95_m'] <= .30
                and safety['operating_speed_fraction'] == 0 and safety['joint_limit_fraction'] == 0)
            if absolute.get('clean') is not absolute_clean:
                raise ValueError('Absolute clean flag contradicts world/safety trial evidence')
        groups[expected[key].get('semantic_group', 'family:'+expected[key]['family'])].append(row)

    def reduce(rows):
        n = len(rows)
        result = {'trials': n, 'completed': sum(bool(r['completed']) for r in rows),
                'clean': sum(bool(r['clean_success']) for r in rows),
                'clean_v3': sum(bool(r['trajectory_v3']['clean_v3']) for r in rows),
                'collision_trials': sum(r['self_collision_ticks'] > 0 for r in rows),
                'fell': sum(bool(r['fell']) for r in rows),
                'full_reference_duration_xy_score': sum(r['trajectory_v3']['full_reference_duration_xy_score'] for r in rows)/n,
                'root_xy_rmse_m': sum(r['trajectory_v3']['root_xy_rmse_m'] for r in rows)/n,
                'survived_fraction': sum(r['trajectory_v3']['survived_fraction'] for r in rows)/n}
        if world_available:
            result.update({field: sum(r[field] for r in rows)/n for field in world_means})
            result.update(absolute_clean=sum(r['absolute_motion_v1']['clean'] for r in rows),
                operating_overspeed_trials=sum(r['actuator_safety']['operating_speed_fraction'] > 0 for r in rows),
                nominal_overspeed_trials=sum(r['actuator_safety']['nominal_speed_fraction'] > 0 for r in rows),
                joint_limit_violation_trials=sum(r['actuator_safety']['joint_limit_fraction'] > 0 for r in rows))
        return result

    all_trials = reduce(trials)
    if any(summary.get('all', {}).get(k) != all_trials[k] for k in ['trials', 'completed', 'clean', 'collision_trials', 'fell']):
        raise ValueError('Replay summary differs from recounted trial records')
    by_group = {name: reduce(rows) for name, rows in sorted(groups.items())}
    macro = {field: sum(group[field] for group in by_group.values())/len(by_group)
             for field in ['full_reference_duration_xy_score', 'root_xy_rmse_m', 'survived_fraction']}
    macro['clean_v3_rate'] = sum(group['clean_v3']/group['trials'] for group in by_group.values())/len(by_group)
    result = {'all': all_trials, 'by_semantic_group': by_group, 'trajectory_v3_macro': macro,
            'grouping': 'audited_semantic_group' if all('semantic_group' in r for r in panel) else 'includes_family_fallback',
            'root_xy_rmse_scope': 'executed prefix only; inspect survival alongside RMSE',
            'clean_ids': sorted(r['id'] for r in trials if r['clean_success']),
            'clean_v3_ids': sorted(r['id'] for r in trials if r['trajectory_v3']['clean_v3']),
            'contract': contract, 'execution_errors': 0, 'screening_only': True,
            'behaviorally_accepted': False, 'hardware_verified': False, 'automatically_promoted': False,
            'absolute_motion_available': world_available}
    if world_available:
        world_macro = {field: sum(group[field] for group in by_group.values())/len(by_group)
                       for field in world_means}
        for field in ('absolute_clean', 'operating_overspeed_trials', 'nominal_overspeed_trials',
                      'joint_limit_violation_trials'):
            world_macro[field+'_rate'] = sum(group[field]/group['trials'] for group in by_group.values())/len(by_group)
        result.update(world_safety_macro=world_macro,
            absolute_clean_ids=sorted(r['id'] for r in trials if r['absolute_motion_v1']['clean']),
            world_body_error_scope='Executed prefix only; full_reference_duration_world_score assigns zero to unexecuted tails')
    return result


def evaluate_checkpoint(checkpoint, key, args, panel, panel_hash, revision):
    source = args.bundle/'artifacts/source-snapshots'/revision
    if not (source/'k1_motion/trajectory_metrics.py').is_file():
        raise ValueError('Frozen bundle lacks versioned trajectory diagnostics')
    destination = args.root/'monitor'/key
    destination.mkdir(parents=True, exist_ok=True)
    environment = {**os.environ, 'K1_FROZEN_SOURCE': str(source/'k1_motion'),
        'K1_MOTION_ROOT': str(args.bundle), 'PYTHONPATH': str(source)+os.pathsep+os.environ.get('PYTHONPATH', ''),
        'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    summary_path = destination/'replay/summary.json'
    if not summary_path.exists():
        if (destination/'replay').exists():
            (destination/'replay').rename(destination/f'interrupted-replay-{time.time_ns()}')
        with (destination/'evaluation.log').open('a') as log:
            subprocess.run([sys.executable, '-m', 'k1_motion', 'export', str(checkpoint),
                            '--output', str(destination/'actor.pt')], check=True, cwd=args.bundle,
                           env=environment, stdout=log, stderr=subprocess.STDOUT)
            subprocess.run([sys.executable, str(args.bundle/'scripts/evaluate_rl_reference_pilot.py'),
                '--panel', str(args.panel), '--policy', str(destination/'actor.pt'),
                '--output', str(destination/'replay'), '--workers', '2'], check=True,
                cwd=args.bundle, env=environment, stdout=log, stderr=subprocess.STDOUT)
    trials = [load(destination/'replay'/(row['id']+'.json')) for row in panel]
    result = aggregate_replay(panel, trials, load(summary_path), panel_hash, revision)
    metadata = load(destination/'actor.json')
    result.update(checkpoint=str(checkpoint), actor=str(destination/'actor.pt'),
                  training_exposure={k: metadata.get(k) for k in ['checkpoint_iteration', 'checkpoint_transitions', 'checkpoint_optimizer_steps']})
    write_json(destination/'screen-summary.json', result)
    return result


def cycle(args, state, evaluator=evaluate_checkpoint):
    bundle = load(args.bundle/'bundle.json')
    revision = bundle['source_revision']
    panel_bytes = args.panel.read_bytes()
    panel, panel_hash = json.loads(panel_bytes), hashlib.sha256(panel_bytes).hexdigest()
    if not isinstance(panel, list) or not panel or len({r['id'] for r in panel}) != len(panel):
        raise ValueError('Expected a nonempty unique-ID panel list')
    if state and (state.get('source_revision') != revision or state.get('panel_sha256') != panel_hash):
        raise ValueError('Existing monitor source/panel contract differs')
    if not state:
        state.update(source_revision=revision, panel_sha256=panel_hash, panel=str(args.panel),
                     comparisons={}, behaviorally_accepted=False, hardware_verified=False, automatically_promoted=False)
    campaign = refresh_campaigns(args, bundle)
    state.update(updated_unix=time.time(), host_phases=campaign.get('host_phases', {}),
                 phase='scanning', error=None, trials=len(panel), pending=[])
    for name, run in campaign['runs'].items():
        terminal = run['phase'] in TERMINAL
        for checkpoint in checkpoint_candidates(Path(run['training_directory']), terminal):
            key = name+'/'+checkpoint.stem
            if key in state['comparisons']:
                continue
            state.update(phase='evaluating', current=key, updated_unix=time.time())
            write_json(args.root/'monitor/status.json', state)
            result = evaluator(checkpoint, key, args, panel, panel_hash, revision)
            state['comparisons'][key] = result
            state.update(latest=key, current=None, updated_unix=time.time())
            write_json(args.root/'monitor/status.json', state)
    hosts_terminal = campaign['phase'] in TERMINAL
    runs_terminal = all(run['phase'] in TERMINAL for run in campaign['runs'].values())
    terminal = hosts_terminal and runs_terminal
    state.update(phase=('completed' if campaign['phase'] == 'completed' else 'completed_with_failures')
                 if terminal else 'awaiting_terminal_run_evidence' if hosts_terminal else 'waiting_for_checkpoint',
                 updated_unix=time.time(), current=None)
    write_json(args.root/'monitor/status.json', state)
    return state, terminal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--server-root', required=True)
    parser.add_argument('--desktop-root', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--panel', type=Path, required=True)
    parser.add_argument('--server-host', default='server-wired')
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    args.root, args.bundle, args.panel, args.desktop_root = [p.resolve() for p in [args.root, args.bundle, args.panel, args.desktop_root]]
    (args.root/'monitor').mkdir(parents=True, exist_ok=True)
    lock = (args.root/'monitor/monitor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status = args.root/'monitor/status.json'
    state = load(status) if status.exists() else {}
    while True:
        try:
            state, done = cycle(args, state)
            if done or args.once:
                return
        except (subprocess.SubprocessError, OSError) as error:
            state.update(phase='retrying', updated_unix=time.time(), error=f'{type(error).__name__}: {error}')
            write_json(status, state)
            if args.once:
                raise
        except Exception as error:
            state.update(phase='failed', updated_unix=time.time(), error=f'{type(error).__name__}: {error}')
            write_json(status, state)
            raise
        time.sleep(20)


if __name__ == '__main__':
    main()
