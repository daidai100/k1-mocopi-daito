#!/usr/bin/env python3
"""Three server ablations and one desktop combination, with checkpointed budgets."""
import argparse
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

from run_server_ablation import read_metrics, write_json

ROOT = Path(__file__).resolve().parents[1]
NAMES = ("a_planar_inputs", "b_collision_cost", "c_huber_cost", "d_combined_warp")


def learner_command(root, output, cache, name, num_envs, preflight, iterations=20):
    if name not in NAMES or num_envs not in (1024, 2048):
        raise ValueError("Unknown treatment or unqualified environment count")
    combined = name == NAMES[3]
    collision = name in (NAMES[1], NAMES[3])
    command = [sys.executable, str(root/'scripts/train_warp.py'), '--backend', 'warp' if combined else 'mujoco_cpp',
        '--library', str(root/'library'), '--output', str(output), '--stage', 'student', '--device', 'cuda:0',
        '--num-envs', str(num_envs), '--iterations', str(iterations if preflight else 1000000),
        '--horizon', '32', '--history', '10', '--hidden-sizes', '512', '256',
        '--sampling', 'take_transition_balanced', '--reference-storage', 'packed', '--reference-cache', str(cache),
        '--minibatch', '4096', '--epochs', '4', '--learning-rate', '1e-5', '--min-learning-rate', '1e-6',
        '--kl-stop', '.02', '--bc-weight', '0', '--initialize', str(root/'initialize.pt'),
        '--evaluation-interval', '0', '--checkpoint-interval', '25',
        '--milestone-interval', str(32768000//(num_envs*32)), '--threads', '1', '--seed', '42',
        '--self-collision-weight', '4' if collision else '1', '--first-collision-penalty', '.3' if collision else '0',
        '--root-velocity-weight', '4', '--root-velocity-sigma', '.5', '--residual-scale', '.25',
        '--command-velocity-limit', '6', '--action-settings', str(root/'configs/controller-pv-arm-feedback-v1.json'),
        '--curriculum-manifest', str(root/'manifests/planar-walking-curriculum.json'),
        '--observation-profile', 'planar' if name in (NAMES[0], NAMES[3]) else 'causal',
        '--cpu-workers', '20', '--cpu-chunk-size', '4', '--arm-workers', '8',
        '--nconmax', '64', '--njmax', '256']
    if name in (NAMES[2], NAMES[3]):
        command += ['--tracking-huber']
    if not preflight:
        command += ['--max-seconds', '36000']
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', choices=('server', 'desktop'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference-cache', type=Path, required=True)
    parser.add_argument('--num-envs', type=int, default=2048, choices=(1024, 2048))
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--preflight-report', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a new output directory')
    bundle = json.loads((ROOT/'bundle.json').read_text())
    names = NAMES[:3] if args.host == 'server' else NAMES[3:]
    if not args.preflight:
        if not args.preflight_report:
            raise ValueError('Production requires a completed matching preflight')
        receipt = json.loads(args.preflight_report.read_text())
        if (receipt['phase'] != 'completed' or not receipt['preflight'] or receipt['bundle'] != bundle
                or receipt['host'] != args.host or receipt['num_envs'] != args.num_envs
                or set(receipt['runs']) != set(names)):
            raise ValueError('Preflight does not qualify this placement, environment count, and source')
    import torch
    import mujoco
    torch.set_num_threads(1)
    devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if mujoco.__version__ != '3.10.0':
        raise ValueError('MuJoCo version changed')
    if args.host == 'server':
        if len(devices) != 2 or 'R9700' not in devices[0] or '9060 XT' not in devices[1]:
            raise ValueError('Server GPU mapping changed')
        for core in range(32):
            if Path(f'/sys/devices/system/cpu/cpu{core}/topology/thread_siblings_list').read_text().strip() != f'{core},{core+32}':
                raise ValueError('Server CPU mapping changed')
    elif len(devices) != 1 or '5070 Ti' not in devices[0]:
        raise ValueError('Desktop GPU mapping changed')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True)
    state = dict(phase='starting', started_unix=time.time(), host=args.host, bundle=bundle,
                 hours_per_run=10, preflight=args.preflight, num_envs=args.num_envs,
                 behaviorally_accepted=False, hardware_verified=False, runs={})
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
            command = learner_command(ROOT, directory, args.reference_cache.resolve(), name, args.num_envs, args.preflight)
            environment = {**os.environ, 'OMP_NUM_THREADS':'1', 'OPENBLAS_NUM_THREADS':'1',
                           'MKL_NUM_THREADS':'1', 'OMP_WAIT_POLICY':'PASSIVE', 'GOMP_SPINCOUNT':'0'}
            placement = {}
            if args.host == 'server':
                gpu = 1 if index == 0 else 0
                cores = list(range(index*10, index*10+10))
                mask = cores + [c+32 for c in cores]
                command = ['taskset', '-c', ','.join(map(str, mask)), *command]
                environment['HIP_VISIBLE_DEVICES'] = str(gpu)
                placement = dict(gpu=devices[gpu], hip_visible_devices=str(gpu), cpu_affinity=mask)
            else:
                placement = dict(gpu=devices[0], cuda_visible_devices='0')
                environment['CUDA_VISIBLE_DEVICES'] = '0'
            run = dict(phase='starting', training_directory=str(directory), training_directories=[str(directory)],
                       command=command, **placement)
            write_json(directory.parent/'command.json', run)
            log = (directory.parent/'training.log').open('w')
            logs.append(log)
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
            processes[name] = process
            state['runs'][name] = {**run, 'pid':process.pid}
            publish()
        while True:
            running = False
            for name, process in processes.items():
                run = state['runs'][name]
                if run['phase'] in ('completed', 'failed', 'interrupted'):
                    continue
                directory = Path(run['training_directory'])
                rows = read_metrics(directory/'metrics.jsonl')
                code = process.poll()
                run['exit_code'] = code
                if rows:
                    run['latest'] = rows[-1]
                    run['recent_transitions_per_second'] = statistics.median(r['transitions_per_second'] for r in rows[-10:])
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
                            transitions_per_second=args.num_envs*32*len(measured)/sum(r['iteration_seconds'] for r in measured),
                            mean_rollout_seconds=statistics.mean(r['rollout_seconds'] for r in measured),
                            mean_update_seconds=statistics.mean(r['update_seconds'] for r in measured),
                            peak_visible_gpu_bytes=max(r.get('visible_cuda_memory_used_bytes', 0) for r in rows),
                            projected_updates_10h=36000/statistics.mean(r['iteration_seconds'] for r in measured))
            failed = any(r['phase'] == 'failed' for r in state['runs'].values())
            state['phase'] = 'running' if running else 'failed' if failed else 'interrupted' if stopping else 'completed'
            publish()
            if not running:
                break
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
                process.wait(timeout=300)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for log in logs:
            log.close()


if __name__ == '__main__':
    main()
