#!/usr/bin/env python3
"""Train all >10 s originals with static holds and a hard 300 ms input horizon."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from freeze_source import freeze_source
from prepare_sustained_training import diagnostic_panel
from run_sustained_campaign import write

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def verify_membership(rows, all_rows, curriculum):
    expected = {r['id'] for r in all_rows if r['standing_padding']['original_duration_s'] > 10.+1e-9}
    ids = {r['id'] for r in rows}
    if (not ids or len(ids) != len(rows) or ids != expected or set(curriculum['train_ids']) != ids
            or any(r.get('split') != 'train' or r.get('is_mirror') is not False
                   or r.get('training_eligible') is not True for r in rows)):
        raise ValueError('Training must contain all admitted original clips longer than ten seconds')
    weights = curriculum['target_transition_weights']
    if set(weights) != ids or any(not math.isfinite(w) or w <= 0 for w in weights.values()):
        raise ValueError('Every training original needs positive sampling mass')


def review_candidate(candidate, baseline, protected):
    """Bounded exploration may continue; champion promotion retains the strict gate."""
    from k1_motion.sustained_training import assess_milestone
    if candidate['execution_errors']:
        raise ValueError('Milestone replay has execution errors')
    retention, comparisons, failed = {}, {}, []
    for role in ('training', 'development'):
        retention[role] = assess_milestone(candidate['trials'][role], protected['trials'][role],
            candidate['contracts'][role], protected['contracts'][role])
        comparison = assess_milestone(candidate['trials'][role], baseline['trials'][role],
            candidate['contracts'][role], baseline['contracts'][role])
        before, after = comparison['before'], comparison['after']
        n = before['trials']
        # These limits are fixed before training and do not relax promotion.
        # A small, noisy early regression is not evidence that learning is done.
        checks = dict(world_score=after['full_duration_world_score'] >= before['full_duration_world_score']-.05,
            raw=after['raw'] >= before['raw']-max(2, math.ceil(.1*n)),
            jointly_clean=after['jointly_clean'] >= before['jointly_clean']-max(2, math.ceil(.1*n)))
        for key in ('collisions', 'falls', 'joint_limit_trials', 'operating_speed_trials'):
            checks[key] = after[key] <= before[key]+max(2, math.ceil(.1*n))
        failed.extend(role+'/'+key for key, passed in checks.items() if not passed)
        comparisons[role] = dict(checks=checks, before=before, after=after)
    promote = (all(r['continue_training'] for r in retention.values())
               and retention['development']['replace_champion'])
    return dict(version='bounded-padded-causal-review-v1', continue_training=not failed,
        replace_champion=promote, failed_checks=failed, retention=retention, comparisons=comparisons,
        reason='within_declared_exploration_budget' if not failed else 'behavioral_degradation',
        behaviorally_accepted=False, hardware_verified=False)


def main():
    from k1_motion.sustained_training import build_curriculum
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--parent-curriculum', type=Path, required=True)
    parser.add_argument('--development-panel', type=Path, required=True)
    parser.add_argument('--initializer', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--hours', type=float, default=8.)
    parser.add_argument('--updates', type=int, default=12000)
    parser.add_argument('--milestone', type=int, default=500)
    parser.add_argument('--preflight-updates', type=int, default=5)
    parser.add_argument('--num-envs', type=int, default=2048)
    parser.add_argument('--cpu-workers', type=int, default=12)
    parser.add_argument('--evaluation-workers', type=int, default=6)
    parser.add_argument('--seed', type=int, default=45)
    args = parser.parse_args()
    if (not math.isfinite(args.hours) or args.hours <= 0
            or min(args.updates, args.milestone, args.preflight_updates, args.num_envs,
                   args.cpu_workers, args.evaluation_workers) < 1):
        parser.error('All budgets must be positive and finite')
    for key in ('inputs', 'parent_curriculum', 'development_panel', 'initializer'):
        setattr(args, key, getattr(args, key).resolve(strict=True))
    args.output = args.output.resolve()
    rows = [json.loads(line) for line in (args.inputs/'library/index.jsonl').read_text().splitlines()]
    all_rows = [json.loads(line) for line in (args.inputs/'all-padded/index.jsonl').read_text().splitlines()]
    parent = json.loads(args.parent_curriculum.read_text())
    selected = set(parent['locomotion_ids']) & {r['id'] for r in rows}
    curriculum = build_curriculum(rows, selected, sustained=True)
    curriculum['initial_coverage_sweep'] = True
    verify_membership(rows, all_rows, curriculum)
    padding_report = json.loads((args.inputs/'report.json').read_text())
    manifest_digest = hashlib.sha256((args.inputs/'library/index.jsonl').read_bytes()).hexdigest()
    if (padding_report['training_manifest_sha256'] != manifest_digest
            or padding_report['training_originals'] != len(rows) or padding_report['invalid_ticks'] != 0):
        raise ValueError('Training manifest differs from padding audit')
    cache = (args.inputs/'reference-cache.pt').resolve(strict=True)
    snapshot, revision = freeze_source(ROOT)
    args.output.mkdir(parents=True, exist_ok=False)
    inputs = args.output/'inputs'
    inputs.mkdir()
    (inputs/'library').symlink_to(args.inputs/'library', target_is_directory=True)
    curriculum['source'] = dict(library_manifest_sha256=manifest_digest,
        parent_curriculum_sha256=hashlib.sha256(args.parent_curriculum.read_bytes()).hexdigest())
    write(inputs/'curriculum.json', curriculum)
    write(inputs/'training-panel.json', diagnostic_panel(rows))
    (inputs/'development-panel.json').write_bytes(args.development_panel.read_bytes())
    import torch
    initial = torch.load(args.initializer, map_location='cpu', weights_only=True)
    if (initial['observation'].get('preview_horizon_s') != .3
            or initial['observation'].get('playback_delay_s') != .3
            or initial['reward_settings'].get('reference_scale') is not None):
        raise ValueError('Initializer must use original targets and the 300 ms buffered actor contract')
    write(args.output/'controller.json', initial['action_settings'])
    del initial
    plan = dict(version='all-long-padded-causal-v1', source_revision=revision,
        initializer=str(args.initializer), initializer_sha256=hashlib.sha256(args.initializer.read_bytes()).hexdigest(),
        library=str(args.inputs/'library'), library_manifest_sha256=manifest_digest,
        originals=len(rows), by_family=dict(Counter(r['family'] for r in rows)),
        hold_seconds=.3, maximum_preview_seconds=.3, playback_delay_seconds=.3,
        padding_report=padding_report, reward_profile='causal-balanced-v1',
        num_envs=args.num_envs, horizon=32, epochs=4, minibatch=4096,
        learning_rate=1e-5, maximum_learning_rate=3e-5, kl_stop=.02,
        max_seconds=args.hours*3600, maximum_updates=args.updates, milestone=args.milestone,
        review='strict zero-regression champion promotion; bounded exploratory continuation',
        continuation_limits=dict(world_score_drop=.05, count_worsening='max(2, ceil(10% of panel))'),
        full_pool_required=True, confirmation_panel_used=False, behaviorally_accepted=False)
    write(args.output/'plan.json', plan)
    environment = {**os.environ, 'K1_FROZEN_SOURCE':str(snapshot/'k1_motion'),
        'K1_MOTION_ROOT':str(ROOT), 'OMP_NUM_THREADS':'1', 'OPENBLAS_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1'}

    def status(phase, **extra):
        write(args.output/'status.json', dict(phase=phase, updated_at=time.time(), **extra,
            originals=len(rows), source_revision=revision, behaviorally_accepted=False))

    def evaluate(checkpoint, destination):
        with destination.with_suffix('.log').open('w') as log:
            subprocess.run([sys.executable, str(ROOT/'scripts/evaluate_sustained_milestone.py'),
                '--checkpoint',str(checkpoint),'--inputs',str(inputs),'--output',str(destination),
                '--workers',str(args.evaluation_workers)],env=environment,cwd=ROOT,
                stdout=log,stderr=subprocess.STDOUT,check=True)
        return json.loads((destination/'evaluation.json').read_text())

    base_command = [sys.executable,str(ROOT/'scripts/train_warp.py'),'--backend','mujoco_cpp',
        '--library',str(args.inputs/'library'),'--reference-cache',str(cache),
        '--stage','student','--device','cuda:0','--num-envs',str(args.num_envs),
        '--horizon','32','--history','10','--hidden-sizes','512','256',
        '--sampling','take_transition_balanced','--reference-storage','packed',
        '--minibatch','4096','--epochs','4','--learning-rate','1e-5',
        '--min-learning-rate','1e-6','--max-learning-rate','3e-5','--kl-stop','.02',
        '--bc-weight','0','--evaluation-interval','0','--threads','1','--seed',str(args.seed),
        '--reward-profile','causal-balanced-v1','--safety-profile','casual-safe-v1',
        '--observation-profile','preview','--preview-horizon-s','.3',
        '--action-settings',str(args.output/'controller.json'),
        '--curriculum-manifest',str(inputs/'curriculum.json'),'--initialize',str(args.initializer),
        '--cpu-workers',str(args.cpu_workers),'--cpu-chunk-size','4','--arm-workers','4']
    status('preflight')
    command = [*base_command,'--output',str(args.output/'preflight'),
        '--iterations',str(args.preflight_updates),'--checkpoint-interval',str(args.preflight_updates),
        '--milestone-interval',str(args.preflight_updates)]
    write(args.output/'preflight-command.json',command)
    with (args.output/'preflight.log').open('w') as log:
        subprocess.run(command,env=environment,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    report = json.loads((args.output/'preflight/report.json').read_text())
    if not report['finite_updates'] or report['checkpoint_reload_max_error'] != 0:
        raise ValueError('Preflight learning/export checks failed')
    status('initializer_evaluation')
    baseline = evaluate(args.initializer,args.output/'initializer')
    protected, champion = baseline, str(args.initializer)
    reviews = args.output/'reviews'
    reviews.mkdir()
    command = [*base_command,'--output',str(args.output/'training'),'--iterations',str(args.updates),
        '--max-seconds',str(args.hours*3600),'--checkpoint-interval','100',
        '--milestone-interval',str(args.milestone),'--milestone-review-directory',str(reviews),
        '--milestone-review-timeout-s','1800']
    # Every review must have a retained checkpoint, including custom short tests.
    command[command.index('--checkpoint-interval')+1] = str(math.gcd(100,args.milestone))
    write(args.output/'training-command.json',command)
    write(args.output/'champion.json',dict(checkpoint=champion,behaviorally_accepted=False))
    with (args.output/'training.log').open('w') as log:
        process = subprocess.Popen(command,env=environment,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        status('training',trainer_pid=process.pid)
        try:
            while process.poll() is None:
                for checkpoint in sorted((args.output/'training').glob('checkpoint-[0-9]*.pt')):
                    receipt = reviews/(checkpoint.stem+'.json')
                    if receipt.exists():
                        continue
                    status('milestone_evaluation',trainer_pid=process.pid,checkpoint=str(checkpoint))
                    candidate = evaluate(checkpoint,args.output/('evaluation-'+checkpoint.stem))
                    decision = review_candidate(candidate,baseline,protected)
                    if decision['replace_champion']:
                        protected, champion = candidate, str(checkpoint)
                    decision.update(checkpoint_sha256=candidate['checkpoint_sha256'],champion=champion)
                    write(receipt,decision)
                    write(args.output/'champion.json',dict(checkpoint=champion,behaviorally_accepted=False))
                    status('training' if decision['continue_training'] else 'stopping_for_regression',
                        trainer_pid=process.pid,review=str(receipt),summary=candidate['summary'])
                time.sleep(.5)
        except BaseException:
            process.terminate()
            process.wait(timeout=60)
            status('failed',reason='supervision_or_evaluation_error')
            raise
    if process.returncode:
        status('failed',trainer_exit_code=process.returncode)
        raise RuntimeError('Trainer failed; see training.log')
    report = json.loads((args.output/'training/report.json').read_text())
    if not report['finite_updates'] or report['checkpoint_reload_max_error'] != 0:
        status('failed',reason='terminal_finite_or_reload_failure')
        raise ValueError('Terminal training validation failed')
    status('complete',iteration=report['last_metrics']['iteration'],transitions=report['transitions'],
        optimizer_steps=report['optimizer_steps'],stop_reason=report['stop_reason'],champion=champion,
        reference_exposure=report['last_metrics']['reference_exposure'])


if __name__ == '__main__':
    main()
