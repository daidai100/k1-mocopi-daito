#!/usr/bin/env python3
"""Six authorized 2000-update runs: three same-device seed-matched reward pairs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from freeze_source import freeze_source
from run_fidelity_campaign import input_identities, validate_preflight
from run_reward_screen import run_campaign, write_json


def build_plans(root, output, host, iterations=2000):
    if host not in ('desktop', 'server') or not isinstance(iterations, int) or not 1 <= iterations <= 2000:
        raise ValueError('Invalid host or finite update budget')
    root, output = Path(root), Path(output)
    plans = []
    # Alternate pair order to avoid always starting the control first.
    lanes = [(0, 42, 0, '0-15,32-47'), (1, 43, 1, '16-31,48-63')] if host == 'server' else [(0, 44, 0, '0-15')]
    for slot, seed, gpu, affinity in lanes:
        profiles = ['world-body-v1', 'world-decomposed-v1']
        if seed == 43:
            profiles.reverse()
        for profile in profiles:
            name = ('control' if profile == 'world-body-v1' else 'decomposed')+f'_seed{seed}'
            directory = output/name/'training'
            command = [sys.executable, str(root/'scripts/train_warp.py'),
                '--backend', 'mujoco_cpp', '--library', str(root/'library'), '--output', str(directory),
                '--stage', 'student', '--device', 'cuda:0', '--num-envs', '2048',
                '--iterations', str(iterations), '--horizon', '32', '--history', '10',
                '--hidden-sizes', '512', '256', '--sampling', 'take_transition_balanced',
                '--reference-storage', 'packed', '--reference-cache', str(root/'reference-cache.pt'),
                '--minibatch', '4096', '--epochs', '4', '--learning-rate', '1e-5',
                '--min-learning-rate', '1e-6', '--kl-stop', '.02', '--bc-weight', '0',
                '--evaluation-interval', '0', '--checkpoint-interval', '25', '--milestone-interval', '125',
                '--threads', '1', '--seed', str(seed), '--reward-profile', profile,
                '--observation-profile', 'preview', '--preview-horizon-s', '.3',
                '--safety-profile', 'casual-safe-v1', '--action-settings', str(root/'configs/controller.json'),
                '--curriculum-manifest', str(root/'manifests/easy-walk-v1.json'),
                '--arm-workers', '8', '--initialize', str(root/'initialize.pt'),
                '--cpu-workers', '24' if host == 'server' else '16', '--cpu-chunk-size', '4']
            if profile == 'world-decomposed-v1':
                command += ['--world-reward-settings', str(root/'configs/decomposed-v1.json')]
            environment = dict(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                               OMP_WAIT_POLICY='PASSIVE', GOMP_SPINCOUNT='0')
            environment['HIP_VISIBLE_DEVICES' if host == 'server' else 'CUDA_VISIBLE_DEVICES'] = str(gpu)
            plans.append(dict(name=name, slot=slot, command=command, environment=environment,
                host=host, gpu_index=gpu, gpu_logical_index=0, cpu_affinity=affinity,
                cpu_workers=24 if host == 'server' else 16, cwd=str(root), backend='mujoco_cpp',
                training_directory=str(directory), iterations=iterations, seed=seed,
                transitions_budget=iterations*65536, adam_steps_max=iterations*64,
                reward_profile=profile, initialization_semantics='guard_world125; fresh optimizer and physics'))
    return plans


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--host', choices=('desktop', 'server'), required=True)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--preflight-report', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    root, output = args.bundle.resolve(), args.output.resolve()
    plans = build_plans(root, output, args.host, 25 if args.preflight else 2000)
    frozen, revision = freeze_source(root)
    for plan in plans:
        plan['environment']['K1_FROZEN_SOURCE'] = str(frozen/'k1_motion')
    identities = input_identities(plans)
    for name, identity in identities.items():
        identity['decomposed_settings_sha256'] = hashlib.sha256((root/'configs/decomposed-v1.json').read_bytes()).hexdigest()
    bundle = dict(version='easy-walk-six-run-v1', source_revision=revision,
        input_identities=identities, design='three reward pairs matched within seed/device/backend',
        initializer_sha256=hashlib.sha256((root/'initialize.pt').read_bytes()).hexdigest(),
        recovery_resets=False, confirmation_panel_consumed=False,
        production_budget_per_run=2000, total_runs=6)
    if args.dry_run:
        write_json(output/'status.json', dict(phase='planned', bundle=bundle, runs=plans))
        return
    if not args.preflight:
        receipt = json.loads(args.preflight_report.read_text()) if args.preflight_report else None
        validate_preflight(receipt, plans, identities, revision)
    for plan in plans:
        env = {k: v for k, v in os.environ.items() if k not in
               ('HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES')}
        env.update(plan['environment'])
        probe = json.loads(subprocess.check_output([sys.executable, '-c',
            'import torch,mujoco,json; print(json.dumps(dict(mujoco=mujoco.__version__,torch=torch.__version__,devices=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])))'], env=env, text=True))
        expected = ('R9700' if plan['gpu_index'] == 0 else '9060 XT') if args.host == 'server' else '5070 Ti'
        if probe['mujoco'] != '3.10.0' or len(probe['devices']) != 1 or expected not in probe['devices'][0]:
            raise ValueError(f'Unexpected runtime {probe}')
        plan['runtime_probe'] = probe
    # Validate the underlying trainer argv first, then apply identical affinity
    # to both stages. The supervisor itself never initializes a GPU context.
    for plan in plans:
        plan['command'] = ['taskset', '-c', plan['cpu_affinity'], *plan['command']]
    # Save the unwrapped command in the receipt for the existing stage matcher.
    state = run_campaign(plans, output, bundle, args.host, args.preflight)
    if args.preflight:
        for run in state['runs'].values():
            run['command'] = run['command'][3:]
        write_json(output/'status.json', state)
    if state['phase'] != 'completed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
