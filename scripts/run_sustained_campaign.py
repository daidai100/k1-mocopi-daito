#!/usr/bin/env python3
"""Run a matched reset-duration pair with mandatory exported milestone review."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from freeze_source import freeze_source

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.partial')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def main():
    from k1_motion.sustained_training import assess_milestone
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--initializer',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--updates',type=int,default=125)
    p.add_argument('--milestone',type=int,default=125)
    p.add_argument('--preflight-updates',type=int,default=5)
    p.add_argument('--num-envs',type=int,default=2048)
    p.add_argument('--cpu-workers',type=int,default=12)
    p.add_argument('--evaluation-workers',type=int,default=6)
    p.add_argument('--seed',type=int,default=44)
    p.add_argument('--initializer-evaluation',type=Path,
        help='Reuse an exact-checkpoint/panel/source initializer evaluation receipt')
    p.add_argument('--plan-only',action='store_true')
    args=p.parse_args()
    if min(args.updates,args.milestone,args.preflight_updates,args.num_envs,args.cpu_workers,args.evaluation_workers)<1:
        p.error('All budgets must be positive')
    for key in ('inputs','cache','initializer'):
        setattr(args,key,getattr(args,key).resolve(strict=True))
    args.output=args.output.resolve()
    control=json.loads((args.inputs/'control.json').read_text())
    sustained=json.loads((args.inputs/'sustained.json').read_text())
    quality=json.loads((args.inputs/'reference-consistency.json').read_text())
    manifest_digest=hashlib.sha256((args.inputs/'library/index.jsonl').read_bytes()).hexdigest()
    if not quality['ready_for_bounded_training'] or quality['library_manifest_sha256']!=manifest_digest:
        raise ValueError('New training references lack a matching passing contact-consistency audit')
    # The comparison changes only reset placement/available duration. Fidelity
    # event diagnostics, data weights, and every controller/reward setting match.
    allowed={'version','reset_mix','minimum_remaining_s'}
    if {k:v for k,v in control.items() if k not in allowed}!={k:v for k,v in sustained.items() if k not in allowed}:
        raise ValueError('Matched arms differ beyond their declared reset treatment')
    snapshot,revision=freeze_source(ROOT)
    plan=dict(version='sustained-duration-pair-v1',source_revision=revision,inputs=str(args.inputs),
        initializer=str(args.initializer),initializer_sha256=hashlib.sha256(args.initializer.read_bytes()).hexdigest(),
        library_manifest_sha256=manifest_digest,
        arms=['control','sustained'],seed=args.seed,updates=args.updates,milestone=args.milestone,
        num_envs=args.num_envs,horizon=32,epochs=4,minibatch=4096,
        reward_profile='world-velocity-v1',reference_scale=None,regression_budget=0,
        required_review=['training','development'],confirmation_panel_used=False,
        matched_change='80/10/10 versus 50/25/25 resets; >=10s remaining in sustained interiors',
        comparison_scope='Both arms share absolute-position/velocity reward and the expanded quality-weighted base')
    if args.plan_only:
        print(json.dumps(plan,indent=2))
        return
    args.output.mkdir(parents=True,exist_ok=False)
    write(args.output/'plan.json',plan)
    import torch
    initial=torch.load(args.initializer,map_location='cpu',weights_only=True)
    if initial['reward_settings'].get('reference_scale') is not None:
        raise ValueError('Initializer must use original targets')
    write(args.output/'controller.json',initial['action_settings'])
    del initial
    environment={**os.environ,'K1_FROZEN_SOURCE':str(snapshot/'k1_motion'),
        'K1_MOTION_ROOT':str(ROOT),'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    def evaluate(checkpoint,destination):
        with destination.with_suffix('.log').open('w') as log:
            subprocess.run([sys.executable,str(ROOT/'scripts/evaluate_sustained_milestone.py'),
                '--checkpoint',str(checkpoint),'--inputs',str(args.inputs),'--output',str(destination),
                '--workers',str(args.evaluation_workers)],env=environment,cwd=ROOT,
                stdout=log,stderr=subprocess.STDOUT,check=True)
        return json.loads((destination/'evaluation.json').read_text())
    write(args.output/'status.json',dict(phase='initializer_evaluation',plan=plan))
    if args.initializer_evaluation:
        baseline=json.loads(args.initializer_evaluation.read_text())
        if (baseline['source_revision']!=revision or baseline['execution_errors']!=0
                or baseline['checkpoint_sha256']!=plan['initializer_sha256']):
            raise ValueError('Initializer evaluation checkpoint/source differs')
        for role,filename in [('training','training-panel.json'),('development','development-panel.json')]:
            if baseline['contracts'][role]['panel_sha256']!=hashlib.sha256((args.inputs/filename).read_bytes()).hexdigest():
                raise ValueError('Initializer evaluation panels differ')
        write(args.output/'initializer-reuse.json',dict(evaluation=str(args.initializer_evaluation.resolve()),
            source_revision=revision,checkpoint_sha256=plan['initializer_sha256']))
    else:
        baseline=evaluate(args.initializer,args.output/'initializer')
    results={}
    for arm in plan['arms']:
        root=args.output/arm
        root.mkdir()
        reviews=root/'reviews'
        reviews.mkdir()
        protected=baseline
        champion=str(args.initializer)
        for phase,updates in [('preflight',args.preflight_updates),('training',args.updates)]:
            training=root/phase
            command=[sys.executable,str(ROOT/'scripts/train_warp.py'),'--backend','mujoco_cpp',
                '--library',str(args.inputs/'library'),'--reference-cache',str(args.cache),
                '--output',str(training),'--stage','student','--device','cuda:0',
                '--num-envs',str(args.num_envs),'--iterations',str(updates),'--horizon','32',
                '--history','10','--hidden-sizes','512','256','--sampling','take_transition_balanced',
                '--reference-storage','packed','--minibatch','4096','--epochs','4',
                '--learning-rate','1e-5','--min-learning-rate','1e-6','--kl-stop','.02',
                '--bc-weight','0','--evaluation-interval','0','--checkpoint-interval',str(args.milestone),
                '--milestone-interval',str(args.milestone),'--threads','1','--seed',str(args.seed),
                '--reward-profile','world-velocity-v1','--observation-profile','preview',
                '--preview-horizon-s','.3','--safety-profile','casual-safe-v1',
                '--action-settings',str(args.output/'controller.json'),
                '--curriculum-manifest',str(args.inputs/(arm+'.json')),
                '--initialize',str(args.initializer),'--cpu-workers',str(args.cpu_workers),
                '--cpu-chunk-size','4','--arm-workers','4']
            if phase=='training':
                command+=['--milestone-review-directory',str(reviews),'--milestone-review-timeout-s','900']
            write(root/(phase+'-command.json'),command)
            with (root/(phase+'.log')).open('w') as log:
                process=subprocess.Popen(command,env=environment,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                write(args.output/'status.json',dict(phase=phase,arm=arm,trainer_pid=process.pid,plan=plan))
                try:
                    while process.poll() is None:
                        if phase=='training':
                            for checkpoint in sorted(training.glob('checkpoint-[0-9]*.pt')):
                                receipt=reviews/(checkpoint.stem+'.json')
                                if receipt.exists():
                                    continue
                                write(args.output/'status.json',dict(phase='milestone_evaluation',arm=arm,
                                    checkpoint=str(checkpoint),trainer_pid=process.pid,plan=plan))
                                candidate=evaluate(checkpoint,root/('evaluation-'+checkpoint.stem))
                                assessments={role:assess_milestone(candidate['trials'][role],protected['trials'][role],
                                    candidate['contracts'][role],protected['contracts'][role])
                                    for role in ('training','development')}
                                keep_going=all(a['continue_training'] for a in assessments.values())
                                if keep_going and assessments['development']['replace_champion']:
                                    protected=candidate
                                    champion=str(checkpoint)
                                decision=dict(checkpoint_sha256=candidate['checkpoint_sha256'],
                                    continue_training=keep_going,reason='retained' if keep_going else 'regression',
                                    assessments=assessments,champion=champion,behaviorally_accepted=False)
                                write(receipt,decision)
                                write(root/'champion.json',dict(checkpoint=champion,behaviorally_accepted=False))
                                print(json.dumps(dict(arm=arm,milestone=checkpoint.name,
                                    continue_training=keep_going,summary=candidate['summary'])),flush=True)
                        time.sleep(.25)
                except BaseException:
                    process.terminate()  # Trainer catches SIGTERM and saves at a completed update.
                    process.wait(timeout=60)
                    write(args.output/'status.json',dict(phase='failed',arm=arm,reason='review_or_supervision_error'))
                    raise
            if process.returncode:
                write(args.output/'status.json',dict(phase='failed',arm=arm,stage=phase,
                    trainer_exit_code=process.returncode,reason='trainer_process_failed'))
                raise RuntimeError(f'{arm} {phase} exited {process.returncode}; see its log')
            report=json.loads((training/'report.json').read_text())
            if not report['finite_updates'] or report['checkpoint_reload_max_error']!=0:
                write(args.output/'status.json',dict(phase='failed',arm=arm,stage=phase,
                    reason='finite_or_reload_validation_failed'))
                raise ValueError('Finite/reload preflight failed')
            if phase=='preflight' and report['iterations']!=updates:
                raise ValueError('Preflight did not complete its declared budget')
            if phase=='training':
                latest=training/f"checkpoint-{report['last_metrics']['iteration']:06d}.pt"
                if not (reviews/(latest.stem+'.json')).is_file():
                    raise ValueError('Terminal checkpoint has no behavioral review')
                results[arm]=dict(iteration=report['last_metrics']['iteration'],transitions=report['transitions'],
                    optimizer_steps=report['optimizer_steps'],stop_reason=report['stop_reason'],champion=champion)
    write(args.output/'status.json',dict(phase='complete',plan=plan,runs=results,
        behaviorally_accepted=False,hardware_verified=False))


if __name__=='__main__':
    main()
