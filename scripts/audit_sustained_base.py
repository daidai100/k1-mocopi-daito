#!/usr/bin/env python3
"""Check new originals and the bounded training panel for reference consistency."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
ROBOT=None


def initialize():
    from k1_motion.robot import K1Model
    global ROBOT
    ROBOT=K1Model()


def audit(row):
    from k1_motion.contracts import MotionClip
    from k1_motion.reference_quality import audit_reference_consistency
    path=Path(row['reference_path'])
    clip=MotionClip.load(path)
    if any(row.get(k)!=clip.metadata.get(k) for k in ('id','family','capture_group')):
        raise ValueError('Reference provenance changed')
    return dict(id=row['id'],family=row['family'],dataset=row['dataset'],
        reference_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        **audit_reference_consistency(clip,ROBOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--workers',type=int,default=6)
    args=p.parse_args()
    if args.workers<1:
        p.error('workers must be positive')
    out=args.inputs/'reference-consistency.json'
    if out.exists():
        raise ValueError('Refusing to overwrite reference audit')
    by_id={r['id']:r for line in (args.inputs/'library/index.jsonl').open() if (r:=json.loads(line))}
    added={json.loads(line)['id'] for line in (args.inputs/'added-clock-corrected/index.jsonl').open()}
    panel={r['id'] for r in json.loads((args.inputs/'training-panel.json').read_text())}
    with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize) as pool:
        records=list(pool.map(audit,[by_id[key] for key in sorted(added|panel)],chunksize=1))
    failed_added=[r['id'] for r in records if r['id'] in added and not r['contact_consistent']]
    summary=dict(version='sustained-base-consistency-v1',
        library_manifest_sha256=hashlib.sha256((args.inputs/'library/index.jsonl').read_bytes()).hexdigest(),
        audited_references=len(records),new_originals=len(added),new_originals_consistent=len(added)-len(failed_added),
        failed_added_ids=failed_added,ready_for_bounded_training=not failed_added,records=records,
        full_base_independently_reaudited=False,physics_qualified=False,
        scope='New originals gate plus training-panel diagnostics; prior 500 Hz geometry receipts retained')
    out.write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='records'},indent=2))
    if failed_added:
        raise SystemExit('New references failed contact consistency; rebuild the versioned base before training')


if __name__=='__main__':
    main()
