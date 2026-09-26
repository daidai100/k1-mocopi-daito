#!/usr/bin/env python3
"""Print local campaign evidence; safe to run remotely through stdin."""
import json
import os
from pathlib import Path
import sys

root = Path(sys.argv[1])
results = {}
for row in json.loads((root/'plan.json').read_text())['runs']:
    directory = root/row['name']
    status = json.loads((directory/'status.json').read_text())
    result = dict(status)
    pid = status.get('trainer_pid')
    if pid and Path(f'/proc/{pid}/environ').exists():
        try:
            environment = Path(f'/proc/{pid}/environ').read_bytes().split(b'\0')
            result['gpu_environment'] = [v.decode() for v in environment if v.startswith((b'HIP_VISIBLE_DEVICES=', b'CUDA_VISIBLE_DEVICES='))]
            result['cpu_affinity_live'] = sorted(os.sched_getaffinity(pid))
        except (FileNotFoundError, ProcessLookupError):
            result['trainer_exited_during_read'] = True
    for phase in ('preflight', 'training'):
        metrics = directory/phase/'metrics.jsonl'
        if metrics.exists():
            lines = metrics.read_text().splitlines()
            for line in reversed(lines):
                try:
                    m = json.loads(line)
                except json.JSONDecodeError:
                    continue
                result[phase] = {k:m[k] for k in ('iteration','transitions','optimizer_steps','iteration_seconds','transitions_per_second')}
                result[phase]['checkpoint_exists'] = (directory/phase/'checkpoint.pt').is_file()
                break
        report = directory/phase/'report.json'
        if report.exists():
            r = json.loads(report.read_text())
            result.setdefault(phase,{}) .update(finite_updates=r['finite_updates'],checkpoint_reload_max_error=r['checkpoint_reload_max_error'])
    results[row['name']] = result
print(json.dumps(results, indent=2))
