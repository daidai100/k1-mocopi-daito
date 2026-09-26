#!/usr/bin/env python3
"""Paired complete-motion parity and timing against a preserved recovery source."""
# ruff: noqa: E402
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
import numpy as np
from k1_motion.contracts import MotionClip
import recover_bones_seed as candidate


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compare(job):
    row, args, index = job
    original = Path(args.original_source)
    old_retarget = load_module('k1_motion._benchmark_retarget', original / 'src/k1_motion/retarget_recovery.py')
    old_audit = load_module('k1_motion._benchmark_audit', original / 'src/k1_motion/recovery_validation.py')
    old_audit.RecoveryRetargeter = old_retarget.RecoveryRetargeter
    old_runner = load_module('_benchmark_runner', original / 'scripts/recover_bones_seed.py')
    old_runner.retarget_recovery = old_audit.retarget_recovery
    old_runner.audit_recovery = old_audit.audit_recovery
    baseline, dataset, output = map(Path, (args.baseline, args.dataset_root, args.output))
    if row.get('reference_path'):
        (baseline / row['reference_path']).read_bytes()
        (dataset / row['source_path']).read_bytes()
    pair = {}
    for mode in (['original', 'optimized'] if index % 2 == 0 else ['optimized', 'original']):
        runner = old_runner if mode == 'original' else candidate
        started = time.perf_counter()
        result = runner.recover((row, dataset, baseline, output / mode))
        pair[mode] = (result, time.perf_counter() - started)
    a, b = pair['original'][0], pair['optimized'][0]
    # The output directories and elapsed-time fields are intentionally different;
    # compare every other ledger value, not just the acceptance boolean.
    ignored = {'elapsed_seconds', 'retarget_p95_ms'}
    assert {k:v for k,v in a.items() if k not in ignored} == {k:v for k,v in b.items() if k not in ignored}, row['id']
    for key in ['reference_path', 'attempt_reference_path']:
        if key not in a:
            continue
        ac, bc = [MotionClip.load(output / mode / result[key]) for mode, result in [('original', a), ('optimized', b)]]
        assert set(ac.values) == set(bc.values)
        for field in ac.values:
            assert np.array_equal(ac.values[field], bc.values[field]), (row['id'], key, field)
        for field in ['times', 'source_times', 'received_times']:
            assert np.array_equal(getattr(ac, field), getattr(bc, field)), (row['id'], key, field)
    return {'id':row['id'], 'family':row['family'], 'is_mirror':row['is_mirror'],
            'status':a['recovery_status'], 'frames':row.get('frames'),
            'full_seconds':pair['original'][1], 'optimized_seconds':pair['optimized'][1],
            'payload_and_audit_exact_match':True}


def main():
    parser = argparse.ArgumentParser()
    for name in ['original-source', 'baseline', 'dataset-root', 'panel', 'output']:
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    old_contract = json.loads((Path(args.original_source) / 'campaign.json').read_text())
    for name, digest in old_contract['source_hashes'].items():
        assert hashlib.sha256((Path(args.original_source) / name).read_bytes()).hexdigest() == digest, name
    panel = json.loads(Path(args.panel).read_text())
    ids = {r['id'] if isinstance(r, dict) else r for r in panel}
    rows = [r for line in (Path(args.baseline) / 'index.jsonl').open() if (r := json.loads(line))['id'] in ids]
    assert len(rows) == len(ids)
    report = {'baseline_source_hashes':old_contract['source_hashes'],
              'candidate_source_hashes':candidate.source_hashes(), 'complete':False, 'rows':[],
              'workers':args.workers, 'python':sys.version, 'hostname':os.uname().nodename,
              'benchmark_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(compare, [(r,args,i) for i,r in enumerate(rows)]):
            report['rows'].append(result)
            candidate.atomic_json(output / 'parity.json', report)
            print(json.dumps(result), flush=True)
    assert report['candidate_source_hashes'] == candidate.source_hashes(), 'Sources changed during benchmark'
    report['complete'] = True
    report['speedup'] = sum(r['full_seconds'] for r in report['rows']) / sum(r['optimized_seconds'] for r in report['rows'])
    candidate.atomic_json(output / 'parity.json', report)
    print(json.dumps({'complete':True, 'motions':len(rows), 'speedup':report['speedup']}), flush=True)


if __name__ == '__main__':
    main()
