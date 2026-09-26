#!/usr/bin/env python3
"""Three concurrent matched action-chunk runs on the two AMD server GPUs."""
import argparse
import json
from pathlib import Path
import sys

from run_reward_screen import run_campaign, verify_runtime, write_json


def build_plans(bundle,output,cache,updates):
    bundle,output,cache = map(Path,(bundle,output,cache))
    if type(updates) is not int or updates < 1:
        raise ValueError('Update budget must be positive')
    plans = []
    for slot,length in enumerate((2,4,8)):
        name,gpu = f'chunk_{length}',1 if slot == 0 else 0
        cores = list(range(slot*10,slot*10+10))
        affinity = cores+[core+32 for core in cores]
        directory = output/name/'training'
        command = ['taskset','-c',','.join(map(str,affinity)),sys.executable,str(bundle/'scripts/train_warp.py'),
            '--backend','mujoco_cpp','--device','cuda:0','--stage','student',
            '--library',str(bundle/'library'),'--reference-cache',str(cache),
            '--output',str(directory),'--iterations',str(updates),'--num-envs','2048',
            '--horizon','32','--history','10','--hidden-sizes','2048','1024',
            '--action-chunk-size',str(length),'--sampling','take_transition_balanced','--reference-storage','packed',
            '--minibatch','4096','--epochs','4','--learning-rate','1e-5',
            '--min-learning-rate','1e-6','--max-learning-rate','3e-5','--kl-stop','.02','--bc-weight','0',
            '--evaluation-interval','0','--checkpoint-interval','25','--milestone-interval','1000',
            '--threads','1','--seed','45','--reward-profile','causal-balanced-v1',
            '--safety-profile','casual-safe-v1','--observation-profile','preview','--preview-horizon-s','.3',
            '--action-settings',str(bundle/'controller.json'),
            '--curriculum-manifest',str(bundle/'inputs/curriculum.json'),
            '--initialize',str(bundle/f'initializers/{name}.pt'),
            '--cpu-workers','20','--cpu-chunk-size','4','--arm-workers','8',
            '--ppo-update-lock',str(bundle.parent/f'gpu{gpu}-chunk-ppo.lock')]
        environment = dict(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
            OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0',HIP_VISIBLE_DEVICES=str(gpu),
            K1_MOTION_ROOT=str(bundle),K1_FROZEN_SOURCE=str(bundle/'src/k1_motion'))
        plans.append(dict(name=name,slot=slot,chunk_length=length,actor_parameters=5564438,
            total_parameters=11393069,hidden_sizes=[2048,1024],iterations=updates,seed=45,
            gpu_index=gpu,cpu_affinity=affinity,command=command,environment=environment,cwd=str(bundle),
            training_directory=str(directory),iteration_budget_semantics='fresh 8000-update training',
            comparison_contract=dict(chunk_length=length,actor_parameters=5564438,num_envs=2048,horizon=32,
                minibatch_control_ticks=4096,reward_profile='causal-balanced-v1',preview_horizon_s=.3,
                initializer='shared weights; zero time direction; fresh optimizer',backend='mujoco_cpp')))
    return plans


def validate_preflight(report,plan,updates,originals):
    if (report.get('backend') != 'mujoco_cpp' or report.get('device') != 'cuda:0'
            or report.get('num_envs') != 2048 or report.get('actor_parameters') != 5564438
            or report.get('action_chunk',{}).get('length') != plan['chunk_length']
            or report.get('finite_updates') is not True or report.get('checkpoint_reload_max_error') != 0
            or report.get('transitions') != updates*2048*32
            or report.get('last_metrics',{}).get('iteration') != updates):
        raise ValueError('Server preflight backend, capacity, budget or reload differs')
    exposure = report['last_metrics'].get('reference_exposure',{})
    if exposure.get('seen_originals') != originals or exposure.get('total_originals') != originals:
        raise ValueError('Server preflight requires measured full-original coverage')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference-cache',type=Path,required=True)
    parser.add_argument('--updates',type=int,default=8000)
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--preflight-report',type=Path)
    parser.add_argument('--dry-run',action='store_true')
    args = parser.parse_args()
    bundle,output,cache = args.bundle.resolve(),args.output.resolve(),args.reference_cache.resolve()
    receipt = json.loads((bundle/'bundle.json').read_text())
    updates = 25 if args.preflight else args.updates
    plans = build_plans(bundle,output,cache,updates)
    if args.dry_run:
        write_json(output/'plan.json',dict(bundle=receipt,runs=plans))
        return
    devices = verify_runtime('server')
    for plan in plans:
        plan['gpu'] = devices[plan['gpu_index']]
    if not args.preflight:
        if args.preflight_report is None:
            raise ValueError('Production requires a complete full-corpus concurrent preflight')
        prior = json.loads(args.preflight_report.read_text())
        if prior.get('phase') != 'completed' or prior.get('bundle') != receipt:
            raise ValueError('Preflight source or completion differs')
        for plan in plans:
            run = prior['runs'][plan['name']]
            if run.get('comparison_contract') != plan['comparison_contract'] or run.get('phase') != 'completed':
                raise ValueError('Preflight comparison contract differs')
            validate_preflight(json.loads(Path(run['report']).read_text()),plan,25,receipt['originals'])
    state = run_campaign(plans,output,receipt,'server',args.preflight)
    if state['phase'] == 'completed':
        for plan in plans:
            validate_preflight(json.loads((Path(plan['training_directory'])/'report.json').read_text()),
                               plan,updates,receipt['originals'])
    return 0 if state['phase'] == 'completed' else 1


if __name__ == '__main__':
    sys.exit(main())
