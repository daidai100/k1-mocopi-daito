#!/usr/bin/env python3
"""Recount a bounded comparison, including real episode duration and retention."""
import argparse
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    from k1_motion.sustained_training import summarize_trials
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True)
    args=p.parse_args()
    root=args.campaign
    status=json.loads((root/'status.json').read_text())
    if status['phase']!='complete':
        raise ValueError('Both declared arms must finish before the matched comparison')
    reuse=root/'initializer-reuse.json'
    baseline_path=(Path(json.loads(reuse.read_text())['evaluation']) if reuse.exists()
                   else root/'initializer/evaluation.json')
    evaluations={'initializer':json.loads(baseline_path.read_text())}
    durations={}
    for arm,run in status['runs'].items():
        iteration=run['iteration']
        evaluations[arm]=json.loads((root/arm/f'evaluation-checkpoint-{iteration:06d}/evaluation.json').read_text())
        metrics=[json.loads(line) for line in (root/arm/'training/metrics.jsonl').open()]
        selected=metrics[-min(50,len(metrics)):]
        windows=[r['curriculum']['sustained_tracking'] for r in selected]
        ended=sum(r['ended_episodes'] for r in windows)
        durations[arm]=dict(updates=[selected[0]['iteration'],selected[-1]['iteration']],
            ended_episodes=ended,mean_uninterrupted_episode_s=sum(
                r['ended_episodes']*(r['ended_episode_mean_s'] or 0.) for r in windows)/max(1,ended),
            ended_episodes_10s=sum(r['ended_episodes_10s'] for r in windows),
            ended_episodes_20s=sum(r['ended_episodes_20s'] for r in windows),
            completed_from_recording_start=sum(r['completed_from_recording_start'] for r in windows),
            realized_duration_transition_shares={key:sum(r['duration_transition_shares'][key] for r in windows)/len(windows)
                for key in windows[0]['duration_transition_shares']},
            transitions_per_second=sum(r['transitions_per_second'] for r in selected)/len(selected))
    # These fixed diagnostic groups change no pass threshold or selection gate.
    groups={}
    for name,evaluation in evaluations.items():
        trials=evaluation['trials']['development']
        walks=[r for r in trials if r.get('semantic_group')=='ordinary_walk']
        stress=[r for r in walks if re.search(r'_(?:fast|exaggerated)_',r['capture_group'].lower())]
        stress_ids={r['id'] for r in stress}
        subsets={'routine_walk':[r for r in walks if r['id'] not in stress_ids],'stress_walk':stress,
            'ordinary_run':[r for r in trials if r.get('semantic_group')=='ordinary_run']}
        groups[name]={key:summarize_trials(rows) for key,rows in subsets.items() if rows}
    decisions={arm:json.loads((root/arm/'reviews'/f"checkpoint-{run['iteration']:06d}.json").read_text())
        for arm,run in status['runs'].items()}
    summary=dict(version='sustained-duration-pair-results-v1',
        summaries={k:r['summary'] for k,r in evaluations.items()},motion_groups=groups,episode_durations=durations,
        exposure=status['runs'],matched_transitions=len({r['transitions'] for r in status['runs'].values()})==1,
        matched_optimizer_steps=len({r['optimizer_steps'] for r in status['runs'].values()})==1,
        decisions=decisions,execution_errors=sum(r['execution_errors'] for r in evaluations.values()),
        behaviorally_accepted=False,hardware_verified=False,
        confirmation_panel_used=False,grouping='Existing ordinary walks separated by fast/exaggerated source names; no threshold changes')
    # Selection follows the durable review receipts, including a possible retained improvement.
    summary['retained_checkpoints']={arm:run['champion'] for arm,run in status['runs'].items()}
    output=root/'comparison.json'
    if output.exists():
        raise ValueError('Refusing to overwrite the comparison artifact')
    output.write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('decisions','motion_groups')},indent=2))


if __name__=='__main__':
    main()
