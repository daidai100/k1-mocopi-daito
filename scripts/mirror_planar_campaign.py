#!/usr/bin/env python3
"""Mirror immutable milestones over wired SSH for one-machine evaluation."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import time

from run_server_ablation import write_json

SERVER_ROOT = '/mnt/ssd1/k1-motion/experiments/planar-campaign-20260921/server'
SERVER_NAMES = {'a_planar_inputs', 'b_collision_cost', 'c_huber_cost'}
TERMINAL = {'completed', 'failed', 'interrupted', 'completed_with_failures'}


def refresh(root):
    desktop = json.loads((root/'desktop/status.json').read_text())
    reply = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=7', 'server-wired',
                            'cat', SERVER_ROOT+'/status.json'], text=True, capture_output=True, timeout=20, check=True)
    server = json.loads(reply.stdout)
    if server['bundle'] != desktop['bundle'] or set(server['runs']) != SERVER_NAMES:
        raise ValueError('Remote campaign source or treatment list differs')
    write_json(root/'server-status.json', server)
    state = copy.deepcopy(desktop)
    state.update(host='both', server_access='server-wired', server_updated_unix=server['updated_unix'],
                 desktop_updated_unix=desktop['updated_unix'], updated_unix=time.time(),
                 evaluation_host='desktop', scope='Same frozen scalar replay process for all four treatments')
    for name, remote in server['runs'].items():
        destination = root/'server-mirror'/name/'training'
        destination.mkdir(parents=True, exist_ok=True)
        source = remote['training_directory']
        if source != SERVER_ROOT+'/'+name+'/training':
            raise ValueError('Unexpected remote training directory')
        subprocess.run(['rsync', '-a', '--protect-args', '--timeout=30',
            '--include=checkpoint-*.pt', '--include=config.json', '--include=report.json', '--exclude=*',
            'server-wired:'+source+'/', str(destination)+'/'], check=True, timeout=120)
        if remote['phase'] in TERMINAL and not (destination/'report.json').exists():
            # Preserve the last atomic checkpoint of a failed learner as a final diagnostic candidate.
            target = destination/'checkpoint-recovered-terminal.pt'
            if not target.exists():
                incoming = target.with_suffix('.incoming')
                result = subprocess.run(['rsync', '-a', '--protect-args', '--timeout=30',
                    'server-wired:'+source+'/checkpoint.pt', str(incoming)], timeout=120)
                if result.returncode == 0:
                    incoming.replace(target)
        local = copy.deepcopy(remote)
        local.update(remote_training_directory=source, training_directory=str(destination),
                     training_directories=[str(destination)])
        state['runs'][name] = local
    phases = [desktop['phase'], server['phase']]
    state['phase'] = ('completed' if all(p == 'completed' for p in phases) else
                      'completed_with_failures' if all(p in TERMINAL for p in phases) else 'running')
    write_json(root/'all-hosts/status.json', state)
    return state['phase'] in TERMINAL


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    (root/'all-hosts').mkdir(exist_ok=True)
    while True:
        try:
            done = refresh(root)
            write_json(root/'mirror-status.json', dict(phase='completed' if done else 'monitoring',
                updated_unix=time.time(), server_access='server-wired', error=None))
            if done or args.once:
                return
        except (subprocess.SubprocessError, OSError, ValueError) as error:
            write_json(root/'mirror-status.json', dict(phase='retrying', updated_unix=time.time(),
                server_access='server-wired', error=f'{type(error).__name__}: {error}'))
            if args.once:
                raise
        time.sleep(30)


if __name__ == '__main__':
    main()
