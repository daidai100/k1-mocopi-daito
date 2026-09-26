#!/usr/bin/env python3
"""Launch the four authorized S2--S5 learners with durable ten-hour budgets."""
import argparse
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
NAMES = ('s2_positions', 's3_timing', 's4_support', 's5_no_angles_warp')


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def recent_metrics(path):
    if not path.exists():
        return []
    with path.open('rb') as stream:
        size = stream.seek(0, 2)
        start = max(0, size-1024*1024)
        stream.seek(start)
        if start:
            stream.readline()
        data = stream.read()
    rows = []
    for line in data.splitlines(keepends=True):
        if line.endswith(b'\n'):
            rows.append(json.loads(line))
    return rows[-128:]


def learner_command(root, output, cache, name, num_envs, preflight, iterations=25):
    if name not in NAMES or num_envs != 2048:
        raise ValueError('Unknown treatment or unqualified environment count')
    number = NAMES.index(name)+2
    warp = number == 5
    command = [sys.executable, str(root/'scripts/train_warp.py'), '--backend', 'warp' if warp else 'mujoco_cpp',
        '--library', str(root/'library'), '--output', str(output), '--stage', 'student', '--device', 'cuda:0',
        '--num-envs', str(num_envs), '--iterations', str(iterations if preflight else 1000000),
        '--horizon', '32', '--history', '10', '--hidden-sizes', '512', '256',
        '--sampling', 'take_transition_balanced', '--reference-storage', 'packed', '--reference-cache', str(cache),
        '--minibatch', '4096', '--epochs', '4', '--learning-rate', '1e-5', '--min-learning-rate', '1e-6',
        '--kl-stop', '.02', '--bc-weight', '0', '--initialize', str(root/'initialize.pt'),
        '--evaluation-interval', '0', '--checkpoint-interval', '25', '--milestone-interval', '500',
        '--threads', '1', '--seed', '42', '--self-collision-weight', '1',
        '--reward-profile', f'spatial-s{number}-v1', '--residual-scale', '.25', '--command-velocity-limit', '6',
        '--action-settings', str(root/'configs/controller-pv-arm-feedback-v1.json'),
        '--curriculum-manifest', str(root/'manifests/planar-walking-curriculum.json'),
        '--observation-profile', 'planar', '--cpu-workers', '20', '--cpu-chunk-size', '4', '--arm-workers', '8']
    if warp:
        command += ['--nconmax', '64', '--njmax', '256', '--epa-horizon', '96']
    else:
        command += ['--ppo-update-lock', str(root.parent/('gpu1-ppo.lock' if number == 2 else 'gpu0-ppo.lock'))]
    if not preflight:
        command += ['--max-seconds', '36000']
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', choices=('server', 'desktop'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference-cache', type=Path, required=True)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--preflight-report', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a new output directory')
    bundle = json.loads((ROOT/'bundle.json').read_text())
    names = NAMES[:3] if args.host == 'server' else NAMES[3:]
    if not args.preflight:
        if args.preflight_report is None:
            raise ValueError('Production requires the matching completed preflight receipt')
        receipt = json.loads(args.preflight_report.read_text())
        if (receipt['phase'] != 'completed' or not receipt['preflight'] or receipt['bundle'] != bundle
                or receipt['host'] != args.host or set(receipt['runs']) != set(names)):
            raise ValueError('Preflight source, host or treatment contract differs')
    import torch
    import mujoco
    torch.set_num_threads(1)
    devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if mujoco.__version__ != '3.10.0':
        raise ValueError('MuJoCo runtime changed')
    if args.host == 'server':
        if len(devices) != 2 or 'R9700' not in devices[0] or '9060 XT' not in devices[1]:
            raise ValueError('Server GPU mapping changed')
        for core in range(32):
            siblings = Path(f'/sys/devices/system/cpu/cpu{core}/topology/thread_siblings_list').read_text().strip()
            if siblings != f'{core},{core+32}':
                raise ValueError('Server CPU topology changed')
    elif len(devices) != 1 or '5070 Ti' not in devices[0]:
        raise ValueError('Desktop GPU mapping changed')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True)
    state = dict(phase='starting', started_unix=time.time(), host=args.host, bundle=bundle,
        hours_per_run=10, num_envs=2048, preflight=args.preflight, behaviorally_accepted=False,
        hardware_verified=False, runs={})
    processes, logs = {}, []
    stopping = False

    def publish():
        state['updated_unix'] = time.time()
        write_json(args.output/'status.json', state)

    def stop(*_):
        nonlocal stopping
        stopping = True
        for process in processes.values():
            if process.poll() is None:
                process.terminate()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        for index, name in enumerate(names):
            directory = args.output/name/'training'
            directory.parent.mkdir()
            command = learner_command(ROOT, directory, args.reference_cache.resolve(), name, 2048, args.preflight)
            environment = {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                           'MKL_NUM_THREADS': '1', 'OMP_WAIT_POLICY': 'PASSIVE', 'GOMP_SPINCOUNT': '0'}
            if args.host == 'server':
                gpu = 1 if index == 0 else 0
                cores = list(range(index*10, index*10+10))
                mask = cores+[core+32 for core in cores]
                command = ['taskset', '-c', ','.join(map(str, mask)), *command]
                environment['HIP_VISIBLE_DEVICES'] = str(gpu)
                placement = dict(gpu=devices[gpu], hip_visible_devices=str(gpu), cpu_affinity=mask)
            else:
                gpu = 0
                environment['CUDA_VISIBLE_DEVICES'] = '0'
                placement = dict(gpu=devices[0], cuda_visible_devices='0')
            run = dict(phase='starting', training_directory=str(directory), training_directories=[str(directory)],
                       command=command, **placement)
            write_json(directory.parent/'command.json', run)
            log = (directory.parent/'training.log').open('w')
            logs.append(log)
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
            processes[name] = process
            state['runs'][name] = {**run, 'pid': process.pid}
            publish()
        while True:
            running = False
            for name, process in processes.items():
                run = state['runs'][name]
                directory = Path(run['training_directory'])
                rows = recent_metrics(directory/'metrics.jsonl')
                code = process.poll()
                run['exit_code'] = code
                if rows:
                    run['latest'] = rows[-1]
                    run['recent_transitions_per_second'] = statistics.median(
                        row['transitions_per_second'] for row in rows[-10:])
                if code is None:
                    running = True
                    run['phase'] = 'learner_updates' if rows else 'starting'
                elif code or not (directory/'report.json').exists():
                    run.update(phase='failed', error='No successful terminal report; inspect training.log')
                else:
                    report = json.loads((directory/'report.json').read_text())
                    valid = report['finite_updates'] and report['checkpoint_reload_max_error'] == 0
                    run.update(phase=('interrupted' if stopping else 'completed') if valid else 'failed',
                        finite_updates=report['finite_updates'], stop_reason=report['stop_reason'],
                        checkpoint_reload_max_error=report['checkpoint_reload_max_error'])
                    if args.preflight and len(rows) > 4:
                        measured = rows[4:]
                        run['benchmark'] = dict(measured_updates=len(measured),
                            transitions_per_second=2048*32*len(measured)/sum(r['iteration_seconds'] for r in measured),
                            mean_rollout_seconds=statistics.mean(r['rollout_seconds'] for r in measured),
                            mean_update_seconds=statistics.mean(r['update_seconds'] for r in measured),
                            peak_visible_gpu_bytes=max(r.get('visible_cuda_memory_used_bytes', 0) for r in rows))
            failed = any(run['phase'] == 'failed' for run in state['runs'].values())
            state['phase'] = 'running' if running else 'failed' if failed else 'interrupted' if stopping else 'completed'
            publish()
            if not running:
                return
            time.sleep(3)
    except Exception as error:
        state.update(phase='failed', error=f'{type(error).__name__}: {error}')
        publish()
        raise
    finally:
        for process in processes.values():
            if process.poll() is None:
                process.terminate()
        for process in processes.values():
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for log in logs:
            log.close()


if __name__ == '__main__':
    main()
