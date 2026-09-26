#!/usr/bin/env python3
"""Replay fixed panels for each chunk's initial policy and durable milestones."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_padded_causal_campaign import review_candidate
from run_sustained_campaign import write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--initializer-only',action='store_true')
    args = parser.parse_args()
    bundle,campaign,output = args.bundle.resolve(),args.campaign.resolve(),args.output.resolve()
    sys.path.insert(0,str(bundle/'src'))
    receipt = json.loads((bundle/'bundle.json').read_text())
    environment = {**os.environ,'K1_MOTION_ROOT':str(bundle),'K1_FROZEN_SOURCE':str(bundle/'src/k1_motion'),
                   'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    state = dict(phase='initializer_evaluation',source_revision=receipt['source_revision'],
                 runs={},behaviorally_accepted=False,confirmation_panel_used=False)

    def publish(phase,**extra):
        state.update(phase=phase,updated_at=time.time(),**extra)
        write(output/'status.json',state)

    def evaluate(checkpoint,destination):
        path = destination/'evaluation.json'
        if not path.exists():
            destination.parent.mkdir(parents=True,exist_ok=True)
            with destination.with_suffix('.log').open('w') as log:
                subprocess.run([sys.executable,str(bundle/'scripts/evaluate_sustained_milestone.py'),
                    '--checkpoint',str(checkpoint),'--inputs',str(bundle/'inputs'),
                    '--output',str(destination),'--workers',str(args.workers)],env=environment,cwd=bundle,
                    stdout=log,stderr=subprocess.STDOUT,check=True)
        result = json.loads(path.read_text())
        if (result['source_revision'] != receipt['source_revision'] or result['execution_errors']
                or result['checkpoint_sha256'] != hashlib.sha256(checkpoint.read_bytes()).hexdigest()):
            raise ValueError('Evaluation is incomplete or differs from its immutable source/checkpoint')
        return result

    try:
        baseline,protected = {},{}
        for length in (2,4,8):
            name = f'chunk_{length}'
            publish('initializer_evaluation',active=name)
            initializer = bundle/f'initializers/{name}.pt'
            baseline[name] = evaluate(initializer,output/name/'initializer')
            protected[name] = baseline[name]
            state['runs'][name] = dict(baseline=baseline[name]['summary'],champion=str(initializer),evaluations={})
            publish('initializer_evaluation')
        if args.initializer_only:
            publish('initializers_complete',active=None)
            return
        while True:
            for length in (2,4,8):
                name = f'chunk_{length}'
                for checkpoint in sorted((campaign/name/'training').glob('checkpoint-[0-9]*.pt')):
                    if checkpoint.stem in state['runs'][name]['evaluations']:
                        continue
                    publish('milestone_evaluation',active=name,checkpoint=str(checkpoint))
                    candidate = evaluate(checkpoint,output/name/checkpoint.stem)
                    decision = review_candidate(candidate,baseline[name],protected[name])
                    if decision['replace_champion']:
                        protected[name] = candidate
                        state['runs'][name]['champion'] = str(checkpoint)
                    state['runs'][name]['evaluations'][checkpoint.stem] = dict(
                        summary=candidate['summary'],training_exposure=candidate['training_exposure'],
                        comparison=decision,checkpoint_sha256=candidate['checkpoint_sha256'])
                    write(output/name/'champion.json',dict(checkpoint=state['runs'][name]['champion'],
                        behaviorally_accepted=False,selection_panel='fixed training28 and development63'))
                    publish('waiting_for_checkpoint',active=None)
            status = campaign/'status.json'
            if status.exists():
                phase = json.loads(status.read_text())['phase']
                if phase in ('completed','failed','interrupted'):
                    publish('complete' if phase == 'completed' else 'learner_'+phase,active=None)
                    return
            publish('waiting_for_checkpoint',active=None)
            time.sleep(10)
    except BaseException as error:
        publish('failed',error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    main()
