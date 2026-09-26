#!/usr/bin/env python3
"""Finite matched preview/world-objective experiments with common actuator safety."""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_reward_screen import run_campaign, verify_runtime, write_json  # noqa: E402

VERSION = 'k1-preview-world-campaign-v1'
TREATMENTS = {
    'simple_masked': ('simple-track-v2', 0.),
    'simple_preview': ('simple-track-v2', .3),
    'world_masked': ('world-body-v1', 0.),
    'world_preview': ('world-body-v1', .3),
}
ACTION_FILE = 'controller-pv-official80-v1.json'
CURRICULUM_FILE = 'minimal-casual-curriculum-v1.json'


def build_plans(bundle, output, cache, host, names, iterations, resume_from=None, seed=42):
    bundle, output, cache = Path(bundle), Path(output), Path(cache)
    if not isinstance(iterations, int) or iterations <= 0:
        raise ValueError('Iteration budget must be positive')
    if not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError('Seed must be an integer in [0, 2**32)')
    if host not in ('server', 'desktop') or not names or len(set(names)) != len(names):
        raise ValueError('Choose a host and unique treatment names')
    if any(name not in TREATMENTS for name in names):
        raise ValueError('Unknown preview treatment')
    if host == 'desktop' and names != ['world_preview']:
        raise ValueError('The desktop queue is reserved for world_preview')
    plans=[]
    for name in names:
        reward, preview = TREATMENTS[name]
        slot=list(TREATMENTS).index(name)%3 if host=='server' else 0
        gpu=1 if host=='server' and slot==0 else 0
        backend='mujoco_cpp' if host=='server' else 'warp'
        directory=output/name/'training'
        comparison=dict(version=VERSION,reward_profile=reward,observation_profile='preview',
            preview_horizon_s=preview,safety_profile='casual-safe-v1',action_settings_file=ACTION_FILE,
            upper_body_residual_scale=0,legacy_self_collision_weight=0,first_collision_penalty=0,
            corruption=False,curriculum=CURRICULUM_FILE,backend=backend,num_envs=2048,
            horizon=32,history=10,hidden_sizes=[512,256],sampling='take_transition_balanced',
            reference_storage='packed',minibatch=4096,epochs=4,learning_rate=1e-5,
            min_learning_rate=1e-6,kl_stop=.02,bc_weight=0,seed=seed,
            checkpoint_interval=25,milestone_interval=125)
        command=[sys.executable,str(bundle/'scripts/train_warp.py'),
            '--backend',backend,'--library',str(bundle/'library'),'--output',str(directory),
            '--stage','student','--device','cuda:0','--num-envs','2048','--iterations',str(iterations),
            '--horizon','32','--history','10','--hidden-sizes','512','256',
            '--sampling','take_transition_balanced','--reference-storage','packed',
            '--reference-cache',str(cache),'--minibatch','4096','--epochs','4',
            '--learning-rate','1e-5','--min-learning-rate','1e-6','--kl-stop','.02','--bc-weight','0',
            '--evaluation-interval','0','--checkpoint-interval','25','--milestone-interval','125',
            '--threads','1','--seed',str(seed),'--self-collision-weight','0','--first-collision-penalty','0',
            '--reward-profile',reward,'--observation-profile','preview','--preview-horizon-s',str(preview),
            '--safety-profile','casual-safe-v1','--action-settings',str(bundle/'configs'/ACTION_FILE),
            '--curriculum-manifest',str(bundle/'manifests'/CURRICULUM_FILE),
            '--cpu-workers','20','--cpu-chunk-size','4','--arm-workers','8']
        if resume_from is None:
            command += ['--initialize',str(bundle/'initialize.pt')]
        else:
            checkpoint=Path(resume_from)/name/'training/checkpoint.pt'
            if not checkpoint.is_file():
                raise ValueError(f'Missing resume checkpoint: {checkpoint}')
            command += ['--resume',str(checkpoint)]
        environment={'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
                     'OMP_WAIT_POLICY':'PASSIVE','GOMP_SPINCOUNT':'0'}
        affinity=[]
        if host=='server':
            cores=list(range(slot*10,slot*10+10))
            affinity=cores+[core+32 for core in cores]
            command += ['--ppo-update-lock',str(bundle.parent/f'gpu{gpu}-ppo.lock')]
            command=['taskset','-c',','.join(map(str,affinity)),*command]
            environment['HIP_VISIBLE_DEVICES']=str(gpu)
        else:
            command += ['--nconmax','64','--njmax','256','--epa-horizon','96']
            environment['CUDA_VISIBLE_DEVICES']='0'
        plans.append(dict(name=name,slot=slot,command=command,environment=environment,cpu_affinity=affinity,
            gpu_index=gpu,cwd=str(bundle),training_directory=str(directory),iterations=iterations,seed=seed,
            iteration_budget_semantics='additional',reward_profile=reward,observation_profile='preview',
            preview_horizon_s=preview,safety_profile='casual-safe-v1',comparison_contract=comparison,
            initializer_semantics='strict_checkpoint_resume' if resume_from else 'fresh_optimizer_with_transferred_old_initializer',
            resume_from=str(resume_from) if resume_from is not None else None))
    return plans


def validate_receipt(receipt,bundle,host,plans,*,preflight):
    label='Preflight' if preflight else 'Resume'
    if (receipt.get('bundle')!=bundle or receipt.get('host')!=host
            or receipt.get('seed')!=plans[0]['seed']
            or preflight and (receipt.get('phase')!='completed' or receipt.get('preflight') is not True)):
        raise ValueError(f'{label} source, host, seed or completion contract differs')
    for plan in plans:
        prior=receipt.get('runs',{}).get(plan['name'],{})
        if (prior.get('comparison_contract')!=plan['comparison_contract']
                or prior.get('seed')!=plan['seed']
                or preflight and prior.get('phase')!='completed'):
            raise ValueError(f'{label} selected treatment contract differs: {plan["name"]}')


def verify_inputs(root,cache,*,initialize):
    paths=[cache,root/'scripts/train_warp.py',root/'library/index.jsonl',
           root/'manifests'/CURRICULUM_FILE,root/'configs'/ACTION_FILE,root/'configs/k1.json']
    if initialize:
        paths.append(root/'initialize.pt')
    for path in paths:
        if not path.is_file():
            raise ValueError(f'Missing required input: {path}')
    action=json.loads((root/'configs'/ACTION_FILE).read_text())
    nominal=json.loads((root/'configs/k1.json').read_text())['velocity_limit']
    limits=action.get('command_velocity_limit')
    if (action.get('upper_body_residual_scale')!=0 or not isinstance(limits,list) or len(limits)!=22
            or len(nominal)!=22 or any(not isinstance(value,(int,float)) or not math.isfinite(value)
                                      or abs(value-.8*bound)>1e-10 for value,bound in zip(limits,nominal))
            or action.get('actuator_profile')!='booster-train-k1-actuator-v1'):
        raise ValueError('Controller official80 speed/safety/arm-mask contract differs')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,default=ROOT)
    parser.add_argument('--host',choices=('server','desktop'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference-cache',type=Path,required=True)
    parser.add_argument('--names',nargs='+',choices=list(TREATMENTS))
    parser.add_argument('--iterations',type=int,help='Additional iterations, also when strictly resuming')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--preflight-report',type=Path)
    parser.add_argument('--resume-from',type=Path)
    parser.add_argument('--dry-run',action='store_true',help='Write reviewable plans without importing GPU runtime')
    args=parser.parse_args()
    root,output,cache=args.bundle.resolve(),args.output.resolve(),args.reference_cache.resolve()
    if output.exists():
        raise ValueError('Use a new output directory')
    names=args.names or (list(TREATMENTS) if args.host=='server' else ['world_preview'])
    iterations=args.iterations if args.iterations is not None else 25 if args.preflight else 125 if args.host=='server' else 1000
    bundle=json.loads((root/'bundle.json').read_text())
    plans=build_plans(root,output,cache,args.host,names,iterations,
                      args.resume_from.resolve() if args.resume_from else None,args.seed)
    if args.preflight_report:
        validate_receipt(json.loads(args.preflight_report.read_text()),bundle,args.host,plans,preflight=True)
    if args.resume_from:
        validate_receipt(json.loads((args.resume_from/'status.json').read_text()),bundle,args.host,plans,preflight=False)
    if args.dry_run:
        state=dict(phase='planned',campaign_version=VERSION,host=args.host,bundle=bundle,preflight=args.preflight,
            seed=args.seed,automatically_promoted=False,runs={p['name']:{**p,'phase':'planned'} for p in plans})
        write_json(output/'status.json',state)
        for plan in plans:
            write_json(output/plan['name']/'command.json',plan)
        print(json.dumps(dict(phase='planned',status=str(output/'status.json'))))
        return 0
    verify_inputs(root,cache,initialize=args.resume_from is None)
    devices=verify_runtime(args.host)
    for plan in plans:
        plan['gpu']=devices[plan['gpu_index']]
    state=run_campaign(plans,output,bundle,args.host,args.preflight)
    return 1 if state['phase']=='failed' else 130 if state['phase']=='interrupted' else 0


if __name__=='__main__':
    sys.exit(main())
