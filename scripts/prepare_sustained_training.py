#!/usr/bin/env python3
"""Freeze the expanded, clock-correct training pool, two reset arms and diagnostics."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]


def diagnostic_panel(rows, *, per_family=2):
    """Take-diverse panel selected without looking at any controller result."""
    from k1_motion.reference_admission import take_family
    from k1_motion.sustained_training import duration_bucket,quality_tier
    selected,used=set(),set()
    by_id={r['id']:r for r in rows}
    def choose(subset):
        ordered=sorted(subset,key=lambda r:(take_family(r['capture_group']) in used,
            quality_tier(r)!='strict_geometry',r['frames']<501,r['frames'],r['id']))
        if ordered:
            r=ordered[0]
            selected.add(r['id'])
            used.add(take_family(r['capture_group']))
    for family in sorted({r['family'] for r in rows}):
        for i in range(per_family):
            subset=[r for r in rows if r['family']==family and r['id'] not in selected
                and (i==0 or r['frames']>=1001)]
            choose(subset)
    for dataset in sorted({r['dataset'] for r in rows}):
        if not any(by_id[key]['dataset']==dataset for key in selected):
            choose([r for r in rows if r['dataset']==dataset])
    result=[]
    for key in sorted(selected):
        row=by_id[key]
        path=Path(row['reference_path']).resolve(strict=True)
        result.append({**{k:row[k] for k in ('id','family','capture_group','split','dataset')},
            'cohort':'training_diagnostic','semantic_group':'family:'+row['family'],
            'reference_path':str(path),'reference_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'duration_s':(row['frames']-1)*.02,'duration_bucket':duration_bucket(row),
            'quality_tier':quality_tier(row),'take_family':take_family(row['capture_group'])})
    return result


def main():
    from k1_motion.reference_velocity_clock import repair_training_library
    from k1_motion.sustained_training import build_curriculum,duration_bucket,quality_tier
    from k1_motion.reference_admission import take_family
    from prepare_broad_references import rows,atomic_json
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-library',type=Path,required=True)
    p.add_argument('--expanded-library',type=Path,required=True)
    p.add_argument('--parent-curriculum',type=Path,required=True)
    p.add_argument('--development-panel',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    args=p.parse_args()
    args.output=args.output.resolve()
    args.output.mkdir(parents=True,exist_ok=False)
    base=list(rows(args.base_library/'index.jsonl'))
    original_ids={r['id'] for r in base}
    expanded=list(rows(args.expanded_library/'index.jsonl'))
    parent=json.loads(args.parent_curriculum.read_text())
    if original_ids!=set(parent['train_ids']) or not original_ids<={r['id'] for r in expanded}:
        raise ValueError('Expanded source must preserve the entire frozen original training base')
    additions=[r for r in expanded if r['id'] not in original_ids]
    # Only the new originals require clock work. Already repaired base payloads
    # and their original immutable receipts remain untouched.
    if additions:
        repair_training_library(args.expanded_library,args.output/'added-clock-corrected',
            [r['id'] for r in additions],workers=args.workers)
        additions=list(rows(args.output/'added-clock-corrected/index.jsonl'))
        additions=[{**r,'reference_path':str((args.output/'added-clock-corrected'/r['reference_path']).resolve())}
                   for r in additions]
    base=[{**r,'reference_path':str((args.base_library/r['reference_path']).resolve(strict=True))} for r in base]
    merged=base+additions
    selected=set(parent['locomotion_ids'])
    for r in additions:
        longest=r['motion_features']['event_longest_s']
        travel=max(longest[k] for k in ('forward_travel','backward_travel','leftward_travel','rightward_travel'))
        if r['family'] in ('walk','run','turn') and (travel>=.3 or longest['turning']>=.2):
            selected.add(r['id'])
    dev=json.loads(args.development_panel.read_text())
    train_families={take_family(r['capture_group']) for r in merged}
    if train_families & {take_family(r['capture_group']) for r in dev}:
        raise ValueError('Expanded training source overlaps the frozen development panel')
    pool=args.output/'library'
    pool.mkdir()
    index=''.join(json.dumps(r,sort_keys=True,allow_nan=False)+'\n' for r in merged)
    (pool/'index.jsonl').write_text(index)
    for sustained in (False,True):
        contract=build_curriculum(merged,selected,sustained=sustained)
        contract['source']=dict(library_manifest_sha256=hashlib.sha256(index.encode()).hexdigest(),
            parent_curriculum_sha256=hashlib.sha256(args.parent_curriculum.read_bytes()).hexdigest())
        atomic_json(args.output/('sustained.json' if sustained else 'control.json'),contract)
    panel=diagnostic_panel(merged)
    atomic_json(args.output/'training-panel.json',panel)
    (args.output/'development-panel.json').write_bytes(args.development_panel.read_bytes())
    def census(subset):
        return dict(originals=len(subset),related_take_families=len({take_family(r['capture_group']) for r in subset}),
            reference_hours=sum((r['frames']-1)*.02 for r in subset)/3600,
            by_family=dict(sorted(Counter(r['family'] for r in subset).items())),
            by_source=dict(sorted(Counter(r['dataset'] for r in subset).items())),
            by_duration=dict(sorted(Counter(duration_bucket(r) for r in subset).items())),
            by_quality=dict(sorted(Counter(quality_tier(r) for r in subset).items())))
    atomic_json(args.output/'report.json',dict(version='sustained-training-base-v1',
        before=census(base),after=census(merged),added=census(additions),
        training_diagnostic_originals=len(panel),development_originals=len(dev),
        target_duration_transition_shares=contract['target_duration_transition_shares'],
        target_family_transition_shares=contract['target_family_transition_shares'],
        target_quality_transition_shares=contract['target_quality_transition_shares'],
        target_source_transition_shares=contract['target_source_transition_shares'],
        independent_physics_qualified_originals=0,mirrors=0,confirmation_panel_used=False,
        strict_reference_audit_hz=500,clock_correction='causal derivative channels only for newly added originals',
        purpose='Matched reset-duration comparison on a broader, quality-weighted base; no behavioral claim'))
    print((args.output/'report.json').read_text())


if __name__=='__main__':
    main()
