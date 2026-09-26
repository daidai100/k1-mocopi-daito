#!/usr/bin/env python3
"""Replay bounded, immutable training/development panels using an exported actor."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys

from freeze_source import freeze_source

ROOT=Path(os.environ.get('K1_MOTION_ROOT',Path(__file__).resolve().parents[1]))
FROZEN,REVISION=freeze_source(ROOT,os.environ.get('K1_FROZEN_SOURCE'))
os.environ['K1_MOTION_ROOT']=str(ROOT)
sys.path.insert(0,str(FROZEN))
ROBOT=POLICY=None
AUTHORITY_TRACE=False


def initialize(actor, authority_trace=False):
    import torch
    from k1_motion.robot import K1Model
    from k1_motion.learning import Policy
    global ROBOT,POLICY,AUTHORITY_TRACE
    AUTHORITY_TRACE=authority_trace
    torch.set_num_threads(1)
    ROBOT=K1Model()
    POLICY=Policy(actor,ROBOT.signature)


def trial(job):
    from k1_motion.contracts import MotionClip
    from k1_motion.control_validation import replay_clip
    from k1_motion.reference_admission import take_family
    role,row,output=job
    try:
        path=Path(row['reference_path'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row['reference_sha256']:
            raise ValueError('Frozen panel payload changed')
        clip=MotionClip.load(path)
        if any(clip.metadata.get(k)!=row[k] for k in ('id','capture_group','family')):
            raise ValueError('Panel/payload identity differs')
        if role=='development' and take_family(row['capture_group']) in {
                take_family(g) for g in POLICY.metadata['train_parents']}:
            raise ValueError('Development reference overlaps training provenance')
        result={**row,**replay_clip(ROBOT,POLICY,clip,Path(output)/role/(row['id']+'.npz'),
                    authority_trace=AUTHORITY_TRACE and role=='training'),
                'execution_passed':True}
    except Exception as error:
        result={**row,'execution_passed':False,'error':f'{type(error).__name__}: {error}'}
    (Path(output)/role/(row['id']+'.json')).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return role,result


def main():
    from k1_motion.export import export_checkpoint
    from k1_motion.reference_admission import take_family
    from k1_motion.sustained_training import summarize_trials
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=6)
    p.add_argument('--authority-trace',action='store_true')
    args=p.parse_args()
    if args.workers<1:
        p.error('workers must be positive')
    args.output.mkdir(parents=True,exist_ok=False)
    export_checkpoint(args.checkpoint,args.output/'actor.pt')
    metadata=json.loads((args.output/'actor.json').read_text())
    if metadata.get('reference_scale') is not None:
        raise ValueError('Sustained baseline requires original unscaled targets')
    library=[json.loads(line) for line in (args.inputs/'library/index.jsonl').open()]
    train={r['id']:r for r in library}
    parents={take_family(r['capture_group']) for r in library}
    panels,contracts={},{}
    for role,filename in [('training','training-panel.json'),('development','development-panel.json')]:
        payload=(args.inputs/filename).read_bytes()
        rows=json.loads(payload)
        if not rows or len({r['id'] for r in rows})!=len(rows):
            raise ValueError('Panel must contain unique references')
        for r in rows:
            if Path(r['id']).name!=r['id'] or r['id'] in ('.','..','summary','contract'):
                raise ValueError('Unsafe panel identifier')
            if role=='training' and (r['id'] not in train or r['split']!='train'
                    or Path(r['reference_path']).resolve()!=Path(train[r['id']]['reference_path']).resolve()):
                raise ValueError('Training diagnostics must be drawn from the declared training pool')
            if role=='development' and take_family(r['capture_group']) in parents:
                raise ValueError('Development panel leaks into the declared expanded pool')
        panels[role]=rows
        contracts[role]=dict(panel_sha256=hashlib.sha256(payload).hexdigest(),source_revision=REVISION,
            reference_scale=None,action_settings=metadata['action_settings'],
            actuator_contract=metadata.get('actuator_contract'),observation=metadata['observation'],
            action_chunk=metadata.get('action_chunk'))
        (args.output/role).mkdir()
    results={role:[] for role in panels}
    jobs=[(role,row,str(args.output)) for role,rows in panels.items() for row in rows]
    with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize,
            initargs=(str(args.output/'actor.pt'),args.authority_trace)) as pool:
        for role,row in pool.map(trial,jobs,chunksize=1):
            results[role].append(row)
    errors=sum(not r['execution_passed'] for rows in results.values() for r in rows)
    result=dict(source_revision=REVISION,checkpoint=str(args.checkpoint.resolve()),
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        training_exposure={k:metadata.get(k) for k in ('checkpoint_iteration','checkpoint_transitions','checkpoint_optimizer_steps')},
        contracts=contracts,execution_errors=errors,trials=results,
        behaviorally_accepted=False,hardware_verified=False,confirmation_panel_used=False)
    if not errors:
        result['summary']={role:summarize_trials(rows) for role,rows in results.items()}
        result['by_family']={role:{family:summarize_trials([r for r in rows if r['family']==family])
            for family in sorted({r['family'] for r in rows})} for role,rows in results.items()}
    (args.output/'evaluation.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    if errors:
        raise RuntimeError(f'{errors} replay execution errors; see saved trial evidence')
    print(json.dumps(result['summary'],indent=2),flush=True)


if __name__=='__main__':
    main()
