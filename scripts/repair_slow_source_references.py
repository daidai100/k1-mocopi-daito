#!/usr/bin/env python3
"""Bounded versioned source-hold repair; preserve old rejects and all audit gates."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]


def main():
    from freeze_source import freeze_source
    from prepare_broad_references import atomic_json,worker_init,rows
    from prepare_complementary_references import convert
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=2)
    args=p.parse_args()
    if args.workers<1:
        p.error('workers must be positive')
    old={r['source_motion_id']:r for r in rows(args.parent/'index.jsonl')}
    records=json.loads((args.parent/'campaign.json').read_text())['selected']
    selected=[{**r,'recovery_profile':'control-tick-hold-v1'} for r in records
        if r['dataset']=='lafan1' and old[r['source_motion_id']].get('rejected_ticks')==0
        and set(old[r['source_motion_id']]['span_reference_audit']['rejection_reasons'])
            =={'control_clock_velocity_limit','ground_penetration_on_command_path'}]
    snapshot,revision=freeze_source(ROOT)
    args.output=args.output.resolve()
    args.output.mkdir(parents=True,exist_ok=False)
    for directory in ('attempts','clips'):
        (args.output/directory).mkdir()
    atomic_json(args.output/'campaign.json',dict(version='slow-source-repair-v1',source_revision=revision,
        parent=str(args.parent.resolve()),selected=selected,whole_recording=True,
        audit_gates_changed=False,selection='Prior zero-invalid-tick sources with only velocity and ground failures',
        converter_sha256=hashlib.sha256((ROOT/'scripts/prepare_complementary_references.py').read_bytes()).hexdigest()))
    results=[]
    with (args.output/'index.jsonl').open('w') as sink, ProcessPoolExecutor(max_workers=args.workers,
            initializer=worker_init,initargs=(snapshot,)) as pool:
        for future in as_completed([pool.submit(convert,(r,str(args.output),revision)) for r in selected]):
            row=future.result()
            results.append(row)
            sink.write(json.dumps(row,allow_nan=False)+'\n')
            sink.flush()
            print(json.dumps(dict(processed=len(results),expected=len(selected),
                accepted=sum(r['kinematics_accepted'] for r in results))),flush=True)
    summary=dict(version='slow-source-repair-v1',complete=True,processed=len(results),
        original_accepted=sum(old[r['source_motion_id']]['kinematics_accepted'] for r in results),
        repaired_accepted=sum(r['kinematics_accepted'] for r in results),physics_qualified=False,
        comparison=[dict(source_motion_id=r['source_motion_id'],frames=r['frames'],
            old_max_joint_speed_rad_s=old[r['source_motion_id']]['recovery_audit']['joint_speed_max_rad_s'],
            new_max_joint_speed_rad_s=r['recovery_audit']['joint_speed_max_rad_s'],
            old_reasons=old[r['source_motion_id']]['span_reference_audit']['rejection_reasons'],
            new_reasons=r['span_reference_audit']['rejection_reasons'],accepted=r['kinematics_accepted'])
            for r in results],admitted_to_running_training=False)
    atomic_json(args.output/'summary.json',summary)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
