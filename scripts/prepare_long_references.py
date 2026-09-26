#!/usr/bin/env python3
"""Add strictly audited, complete long recordings from existing local corpora."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]


def select_sources(inventory, base, old_splits, *, per_family_dataset=4):
    from k1_motion.corpus import family_of, recording_split
    from k1_motion.reference_admission import take_family
    if per_family_dataset < 1:
        raise ValueError('Source budget must be positive')
    existing = {r['source_motion_id'] for r in base}
    heldout = {take_family(r['capture_group']) for r in base if r['split'] != 'train'}
    heldout.update(take_family(g) for g,s in old_splits.items() if s != 'train')
    excluded, buckets, seen = Counter(), defaultdict(list), set()
    for row in inventory:
        key = row['source_motion_id']
        if key in seen:
            excluded['duplicate_source'] += 1
            continue
        seen.add(key)
        family = family_of(row)
        reason = None
        if key in existing:
            reason = 'already_in_base'
        elif take_family(row['capture_group']) in heldout:
            reason = 'heldout_related_take'
        elif old_splits.get(row['capture_group'], recording_split(row['capture_group'])) != 'train':
            reason = 'heldout_split'
        elif row['duration_seconds'] < 10:
            reason = 'shorter_than_10s'
        elif family in ('external_support_or_recovery', 'unclassified'):
            reason = 'unsupported_semantics'
        elif row.get('numerical_validation') != 'passed' or row.get('is_mirror', False):
            reason = 'source_validation'
        if reason:
            excluded[reason] += 1
            continue
        family = {'reach':'gesture','stance':'idle_stance'}.get(family,family)
        buckets[row['dataset'],family].append({**row,'family':family,'split':'train'})
    selected = []
    for bucket in sorted(buckets):
        candidates = sorted(buckets[bucket], key=lambda r:(r['duration_seconds']<20,
            r['duration_seconds'], r['source_motion_id']))
        # Prefer independent capture groups before extra performers/repetitions.
        groups = set()
        distinct, repeats = [], []
        for row in candidates:
            (repeats if row['capture_group'] in groups else distinct).append(row)
            groups.add(row['capture_group'])
        selected.extend((distinct+repeats)[:per_family_dataset])
        excluded['bounded_conversion_budget'] += max(0,len(candidates)-per_family_dataset)
    return selected, dict(inventory_originals=len(seen), excluded=dict(sorted(excluded.items())),
        selected_originals=len(selected), selected_by_source_family=dict(sorted(Counter(
            r['dataset']+'/'+r['family'] for r in selected).items())), minimum_source_duration_s=10,
        preferred_source_duration_s=20, whole_recording=True, windows_created=0)


def main():
    from freeze_source import freeze_source
    from prepare_broad_references import rows, atomic_json, worker_init
    from prepare_complementary_references import convert
    from k1_motion.reference_admission import admit_reference, reference_rejections, take_family
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-library',type=Path,required=True)
    p.add_argument('--split-library',type=Path,action='append',default=[])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--per-family-dataset',type=int,default=4)
    p.add_argument('--plan-only',action='store_true')
    args=p.parse_args()
    if args.workers<1:
        p.error('workers must be positive')
    base=list(rows(args.base_library/'index.jsonl'))
    # Resolve existing payloads once; the new manifest may live elsewhere.
    base=[{**r,'reference_path':str((args.base_library/r['reference_path']).resolve(strict=True))} for r in base]
    split_rows=base.copy()
    for path in args.split_library:
        split_rows.extend(rows(path/'index.jsonl'))
    splits={}
    for r in split_rows:
        old=splits.setdefault(r['capture_group'],r['split'])
        if (old=='train') != (r['split']=='train'):
            raise ValueError('Conflicting training/held-out source split')
    inventory=[r for dataset in ('lafan1','kit_motion_language','bandai_namco')
        for r in rows(ROOT/'manifests'/f'{dataset}.motions.jsonl')]
    selected, plan=select_sources(inventory,base,splits,per_family_dataset=args.per_family_dataset)
    selected=[{**r,'recovery_profile':'control-tick-hold-v1'} for r in selected]
    plan['recovery_profile']='control-tick-hold-v1'
    if args.plan_only:
        print(json.dumps(plan,indent=2))
        return
    snapshot,revision=freeze_source(ROOT)
    args.output=args.output.resolve()
    for name in ('','attempts','clips','pool'):
        (args.output/name).mkdir(parents=True,exist_ok=True)
    contract=dict(version='long-source-expansion-v2',source_revision=revision,
        converter_sha256=hashlib.sha256((ROOT/'scripts/prepare_complementary_references.py').read_bytes()).hexdigest(),
        base_manifest_sha256=hashlib.sha256((args.base_library/'index.jsonl').read_bytes()).hexdigest(),
        base_library=str(args.base_library.resolve()),selected=selected,plan=plan,
        reference_scale=None,whole_recording=True,gate='unchanged strict 500 Hz recovery audit',
        controller_success_used_for_admission=False,physics_qualified=False)
    contract_path=args.output/'campaign.json'
    if contract_path.exists() and json.loads(contract_path.read_text())!=contract:
        raise ValueError('Long-reference campaign contract changed; use a new versioned output')
    atomic_json(contract_path,contract)
    ledger=args.output/'index.jsonl'
    done=list(rows(ledger)) if ledger.exists() else []
    completed={r['source_motion_id'] for r in done}
    expected={r['source_motion_id'] for r in selected}
    if len(completed)!=len(done) or not completed<=expected:
        raise ValueError('Invalid resumable source ledger')
    start=time.monotonic()
    with ledger.open('a') as sink, ProcessPoolExecutor(max_workers=args.workers,
            initializer=worker_init,initargs=(snapshot,)) as pool:
        futures=[pool.submit(convert,(r,str(args.output),revision)) for r in selected if r['source_motion_id'] not in completed]
        for future in as_completed(futures):
            row=future.result()  # Programming/resource errors are never motion rejects.
            done.append(row)
            sink.write(json.dumps(row,allow_nan=False)+'\n')
            sink.flush()
            status=dict(running=True,processed=len(done),expected=len(selected),
                strict_accepted=sum(r['kinematics_accepted'] for r in done),elapsed_s=time.monotonic()-start)
            atomic_json(args.output/'status.json',status)
            print(json.dumps(status),flush=True)
    heldout={take_family(r['capture_group']) for r in split_rows if r['split']!='train'}
    added=[]
    for row in sorted(done,key=lambda r:r['id']):
        if row['kinematics_accepted']:
            reasons=reference_rejections(row,heldout)
            if reasons:
                raise ValueError(f'Admission disagrees with conversion: {reasons}')
            added.append(admit_reference(row))
    merged=base+added
    if len({r['id'] for r in merged})!=len(merged) or len({r['source_motion_id'] for r in merged})!=len(merged):
        raise ValueError('Duplicate source or motion in expanded pool')
    text=''.join(json.dumps(r,sort_keys=True,allow_nan=False)+'\n' for r in merged)
    index=args.output/'pool/index.jsonl'
    if index.exists() and index.read_text()!=text:
        raise ValueError('Frozen expanded pool differs')
    index.write_text(text)
    summary=dict(complete=True,**plan,processed=len(done),base_originals_preserved=len(base),
        new_admitted_originals=len(added),new_reference_hours=sum((r['frames']-1)*.02 for r in added)/3600,
        added_by_source_family=dict(Counter(r['dataset']+'/'+r['family'] for r in added)),
        rejection_reasons=dict(Counter(reason for r in done if not r['kinematics_accepted']
            for reason in r.get('span_reference_audit',{}).get('rejection_reasons',['source_error']))),
        pool_manifest_sha256=hashlib.sha256(text.encode()).hexdigest(),physics_qualified=False,
        physics_qualified_originals=0,mirrors=0,confirmation_panel_used=False)
    atomic_json(args.output/'summary.json',summary)
    atomic_json(args.output/'status.json',dict(running=False,complete=True,processed=len(done)))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
