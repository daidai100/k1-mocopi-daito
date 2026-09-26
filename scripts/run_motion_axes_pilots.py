#!/usr/bin/env python3
"""Run three seed-matched full-corpus root-cost pilots and no-reset replays."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

from freeze_source import freeze_source


WEIGHTS = (.25, 1., 2.)
UPDATES = 125
SOURCE_REVISION = '466e9372e0c9b48208581de347cf5055db5cb5663d266e7609bf201d5624917d'


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def build_runs(bundle, inputs, output, panel, python):
    bundle, inputs, output, panel, python = map(Path, (bundle, inputs, output, panel, python))
    runs = []
    for weight in WEIGHTS:
        name = 'root_'+str(weight).replace('.', '_')
        path = output/name
        command = ['taskset', '-c', '0-15', str(python), str(bundle/'scripts/train_warp.py'),
            '--backend', 'mujoco_cpp', '--library', str(inputs/'library'),
            '--output', str(path/'training'), '--stage', 'student', '--device', 'cuda:0',
            '--num-envs', '2048', '--iterations', str(UPDATES), '--horizon', '32',
            '--history', '10', '--hidden-sizes', '512', '256', '--sampling',
            'take_transition_balanced', '--reference-storage', 'packed', '--reference-cache',
            str(inputs/'reference-cache.pt'), '--minibatch', '4096', '--epochs', '4',
            '--learning-rate', '1e-5', '--min-learning-rate', '1e-6', '--kl-stop', '.02',
            '--bc-weight', '0', '--evaluation-interval', '0', '--checkpoint-interval', '25',
            '--milestone-interval', '125', '--threads', '1', '--seed', '44',
            '--reward-profile', 'world-pointwise-root-v1', '--world-reward-settings',
            str(bundle/'configs'/(name+'.json')), '--observation-profile', 'preview',
            '--preview-horizon-s', '.3', '--safety-profile', 'casual-safe-v1',
            '--action-settings', str(bundle/'configs/controller.json'),
            '--curriculum-manifest', str(inputs/'manifests/minimal-casual-curriculum-v1.json'),
            '--arm-workers', '8', '--initialize', str(bundle/'initialize.pt'),
            '--cpu-workers', '16', '--cpu-chunk-size', '4']
        runs.append(dict(name=name, root_xy_cost_weight=weight, seed=44, gpu_index=0,
            updates=UPDATES, transitions=UPDATES*65536, adam_steps_max=UPDATES*64,
            source_library=str(inputs/'library'),
            curriculum=str(inputs/'manifests/minimal-casual-curriculum-v1.json'),
            panel=str(panel), training_command=command))
    return runs


def require_complete(path, updates):
    report_path = Path(path)/'report.json'
    if not report_path.is_file():
        raise RuntimeError('Missing training report: '+str(report_path))
    report = json.loads(report_path.read_text())
    if (not report.get('finite_updates') or report.get('checkpoint_reload_max_error') != 0
            or report['last_metrics']['iteration'] != updates
            or not (Path(path)/f'checkpoint-{updates:06d}.pt').is_file()):
        raise RuntimeError('Training did not pass finite/reload/checkpoint gate: '+str(path))
    return report


def evaluate(bundle, panel, checkpoint, destination, python, source, *, name):
    destination = Path(destination)
    summary = destination/'replay/summary.json'
    axes = destination/'replay/motion-axes-summary.json'
    if summary.is_file() and axes.is_file():
        return json.loads(axes.read_text())
    if destination.exists() and any(destination.iterdir()):
        raise RuntimeError('Preserve partial evaluation for inspection: '+str(destination))
    destination.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, 'K1_FROZEN_SOURCE': str(source/'k1_motion'),
        'K1_MOTION_ROOT': str(bundle), 'PYTHONPATH': str(source),
        'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    with (destination/'evaluation.log').open('w') as log:
        subprocess.run(['taskset', '-c', '16-19', str(python), '-m', 'k1_motion',
            'export', str(checkpoint), '--output', str(destination/'actor.pt')],
            cwd=bundle, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run(['taskset', '-c', '16-19', str(python),
            str(bundle/'scripts/evaluate_rl_reference_pilot.py'), '--panel', str(panel),
            '--policy', str(destination/'actor.pt'), '--output', str(destination/'replay'),
            '--workers', '4'], cwd=bundle, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    result = json.loads(axes.read_text())
    write_json(destination/'evaluation-receipt.json', dict(name=name, checkpoint=str(checkpoint),
        checkpoint_iteration=json.loads((destination/'actor.json').read_text())['checkpoint_iteration'],
        panel_sha256=result['contract']['panel_sha256'],
        source_revision=result['contract']['source_revision'],
        execution_errors=0, resets_during_trials=0))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--panel', type=Path, required=True)
    parser.add_argument('--python', type=Path, required=True)
    parser.add_argument('--wait-for-status', type=Path)
    args = parser.parse_args()
    bundle, inputs, output, panel, python = (p.resolve() for p in
        (args.bundle, args.inputs, args.output, args.panel, args.python))
    required = [bundle/'src/k1_motion/world_objective.py', bundle/'initialize.pt',
        bundle/'scripts/train_warp.py', bundle/'scripts/evaluate_rl_reference_pilot.py',
        bundle/'scripts/summarize_motion_axes.py', inputs/'library/index.jsonl',
        inputs/'reference-cache.pt', inputs/'manifests/minimal-casual-curriculum-v1.json', panel]
    if any(not p.is_file() for p in required):
        raise FileNotFoundError([str(p) for p in required if not p.is_file()])
    source, revision = freeze_source(bundle, bundle/'src/k1_motion')
    if revision != SOURCE_REVISION:
        raise ValueError('Frozen pointwise-root pilot source revision differs')
    runs = build_runs(bundle, inputs, output, panel, python)
    plan = dict(version='k1-motion-axes-pilots-v1', source_revision=revision,
        source=str(source), panel=str(panel), initializer=str(bundle/'initialize.pt'),
        corpus='18,054 train originals; zero mirrors', target_updates=UPDATES,
        purpose='mean per-point exponential body tracking; linear-tail root XY cost sweep',
        runs=runs, confirmation_panel_consumed=False)
    if output.exists():
        if not (output/'plan.json').is_file() or json.loads((output/'plan.json').read_text()) != plan:
            raise FileExistsError('Existing pilot output has a different plan')
    else:
        output.mkdir(parents=True)
        write_json(output/'plan.json', plan)
    env = {k: v for k, v in os.environ.items() if k not in
        ('HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES')}
    env.update(K1_FROZEN_SOURCE=str(bundle/'src/k1_motion'), CUDA_VISIBLE_DEVICES='0',
        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        OMP_WAIT_POLICY='PASSIVE', GOMP_SPINCOUNT='0')
    try:
        baseline = output/'validation/retained_initializer'
        evaluate(bundle, panel, bundle/'initialize.pt', baseline, python, source,
            name='retained_initializer')
        write_json(output/'status.json', dict(phase='baseline_replayed', updated_at=datetime.now(timezone.utc).isoformat()))
        if args.wait_for_status:
            while True:
                status = json.loads(args.wait_for_status.read_text())
                if status.get('phase') == 'training_complete':
                    break
                if status.get('phase') == 'failed':
                    raise RuntimeError('Existing desktop learner failed; inspect before overlapping GPU')
                write_json(output/'status.json', dict(phase='waiting_for_desktop_gpu',
                    existing_phase=status.get('phase'), updated_at=datetime.now(timezone.utc).isoformat()))
                time.sleep(20)
        for run in runs:
            directory = output/run['name']
            for phase, updates in (('preflight', 25), ('training', UPDATES)):
                target = directory/phase
                command = run['training_command'].copy()
                command[command.index('--iterations')+1] = str(updates)
                command[command.index('--output')+1] = str(target)
                write_json(directory/(phase+'-command.json'), command)
                if (target/'report.json').exists():
                    require_complete(target, updates)
                    continue
                if target.exists():
                    raise RuntimeError('Preserve partial learner before restart: '+str(target))
                write_json(output/'status.json', dict(phase=phase, run=run['name'],
                    updated_at=datetime.now(timezone.utc).isoformat()))
                with (directory/(phase+'.log')).open('w') as log:
                    child = subprocess.Popen(command, cwd=bundle, env=env,
                        stdout=log, stderr=subprocess.STDOUT)
                    write_json(directory/'status.json', dict(phase=phase, pid=child.pid,
                        command_file=str(directory/(phase+'-command.json'))))
                    code = child.wait()
                if code:
                    raise RuntimeError(f'{run["name"]} {phase} exit code {code}')
                report = require_complete(target, updates)
                write_json(directory/'status.json', dict(phase=phase+'_complete',
                    updates=updates, transitions=report['last_metrics']['transitions'],
                    adam_steps=report['last_metrics']['optimizer_steps'], trainer_exit_code=0))
            write_json(output/'status.json', dict(phase='evaluating', run=run['name'],
                updated_at=datetime.now(timezone.utc).isoformat()))
            evaluate(bundle, panel, directory/'training/checkpoint-000125.pt',
                output/'validation'/run['name'], python, source, name=run['name'])
        write_json(output/'status.json', dict(phase='completed', runs=len(runs),
            updated_at=datetime.now(timezone.utc).isoformat(), confirmation_panel_consumed=False))
    except Exception as error:
        write_json(output/'status.json', dict(phase='failed', error=f'{type(error).__name__}: {error}',
            updated_at=datetime.now(timezone.utc).isoformat()))
        raise


if __name__ == '__main__':
    main()
