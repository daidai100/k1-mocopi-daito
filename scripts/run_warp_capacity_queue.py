#!/usr/bin/env python3
"""Sequential, matched 8,000-update K1 capacity runs with CUDA MuJoCo Warp."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from freeze_source import freeze_source
from run_padded_causal_campaign import review_candidate, verify_membership
from run_sustained_campaign import write

ROOT = Path(os.environ.get('K1_MOTION_ROOT', Path(__file__).resolve().parents[1]))


def build_commands(output, scripts, library, cache, initializer, *, updates, num_envs, milestone, seed,
                   preflight_updates=3):
    if min(updates, num_envs, milestone, preflight_updates) < 1 or updates % milestone:
        raise ValueError('Positive budgets and an integral number of milestones are required')
    runs = []
    for name, widths in [('small', [512,256]), ('medium', [2048,1024]), ('large', [4096,2048])]:
        first, second = widths
        actor = (1680+1)*first+(first+1)*second+(second+1)*22
        critic = (1820+1)*first+(first+1)*second+(second+1)
        base = [sys.executable, str(scripts/'train_warp.py'), '--backend','warp','--device','cuda:0',
            '--library',str(library),'--reference-cache',str(cache),'--stage','student',
            '--num-envs',str(num_envs),'--horizon','32','--history','10',
            '--hidden-sizes',*map(str,widths),'--sampling','take_transition_balanced',
            '--reference-storage','packed','--minibatch','4096','--epochs','4',
            '--learning-rate','1e-5','--min-learning-rate','1e-6','--max-learning-rate','3e-5',
            '--kl-stop','.02','--bc-weight','0','--evaluation-interval','0','--threads','1',
            '--seed',str(seed),'--reward-profile','causal-balanced-v1','--safety-profile','casual-safe-v1',
            '--observation-profile','preview','--preview-horizon-s','.3',
            '--action-settings',str(output/'controller.json'),
            '--curriculum-manifest',str(output/'inputs/curriculum.json'),
            '--initialize',str(initializer),'--arm-workers','4',
            '--nconmax','64','--njmax','256','--epa-horizon','96']
        directory = output/name
        command = [*base,'--output',str(directory/'training'),'--iterations',str(updates),
            '--checkpoint-interval','100','--milestone-interval',str(milestone),
            '--milestone-review-directory',str(directory/'reviews'),'--milestone-review-timeout-s','3600']
        preflight = [*base,'--output',str(directory/'preflight'),'--iterations',str(preflight_updates),
                     '--checkpoint-interval',str(preflight_updates),'--milestone-interval',str(preflight_updates)]
        runs.append(dict(name=name,hidden_sizes=widths,actor_parameters=actor,critic_parameters=critic,
                         total_parameters=actor+critic+22,command=command,preflight_command=preflight))
    return runs


def validate_report(report, run, updates, num_envs, *, originals, require_coverage):
    physics = report.get('physics_contract', {})
    metrics = report.get('last_metrics', {})
    backend = run.get('backend', 'warp')
    physics_valid = (physics.get('implementation', '').startswith('mujoco-warp')
                     and physics.get('support', {}).get('device') == 'cuda'
                     and physics.get('epa_horizon') == 96)
    if backend == 'mujoco_cpp':
        physics_valid = (physics.get('backend') == 'native-mujoco-cpp-openmp'
            and physics.get('substeps') == 10 and physics.get('physics_dt') == .002
            and physics.get('control_dt') == .02 and physics.get('actuator', {}).get('safety_dt') == .002)
    if (report.get('backend') != backend or report.get('device') != 'cuda:0'
            or not physics_valid
            or report.get('num_envs') != num_envs
            or report.get('config', {}).get('hidden_sizes') != run['hidden_sizes']
            or report.get('observation', {}).get('preview_horizon_s') != .3
            or report.get('finite_updates') is not True or report.get('checkpoint_reload_max_error') != 0
            or metrics.get('iteration') != updates or report.get('transitions') != updates*num_envs*32):
        raise ValueError('Run lacks exact-budget, finite/reloadable declared physics training evidence')
    exposure = metrics.get('reference_exposure', {})
    expected = run.get('expected_training_contract', {})
    if any((report.get('config', {}).get(key) if key in ('gamma', 'gae_lambda')
            else report.get(key)) != value for key, value in expected.items()):
        raise ValueError('Run training contract differs from the declared reward/discount/scenes')
    if require_coverage and (exposure.get('seen_originals') != originals
                            or exposure.get('total_originals') != originals):
        raise ValueError('Training did not sample every declared original')


def latest_metrics(path):
    if not path.exists():
        return {}
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size-65536))
        lines = stream.read().splitlines()
    for line in reversed(lines):
        try:
            row = json.loads(line)
            if 'iteration' in row:
                return {k:row[k] for k in ('iteration','transitions','optimizer_steps','transitions_per_second',
                    'rollout_seconds','update_seconds','reference_exposure','peak_torch_vram_bytes') if k in row}
        except (ValueError, UnicodeError):
            pass
    return {}


def run_post_experiment(job, environment, root):
    with Path(job['log']).open('a') as log:
        result = subprocess.run(job['command'],env=environment,cwd=root,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"{job['name']} experiment failed with exit code {result.returncode}")
    receipt = json.loads(Path(job['receipt']).read_text())
    if receipt.get('phase') != 'complete':
        raise ValueError('Post experiment did not publish a completion receipt')
    return receipt


def run_queue(plan, environment):
    output = Path(plan['output'])
    state = dict(phase='queued',source_revision=plan['source_revision'],backend=plan.get('backend','warp'),device='cuda:0',
        updates_each=plan['updates'],originals=plan['originals'],behaviorally_accepted=False,
        runs={r['name']:dict(phase='queued',hidden_sizes=r['hidden_sizes'],actor_parameters=r['actor_parameters'])
              for r in plan['runs']})
    active = None

    def status(phase, **extra):
        state.update(phase=phase,updated_at=time.time(),active=active,**extra)
        write(output/'status.json',state)

    def evaluate(checkpoint, destination):
        receipt = destination/'evaluation.json'
        if receipt.exists():
            value = json.loads(receipt.read_text())
            if (value['source_revision'] != plan['source_revision']
                    or value['checkpoint_sha256'] != hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                    or value['execution_errors']):
                raise ValueError('Saved evaluation differs from the frozen checkpoint/source')
            return value
        with destination.with_suffix('.log').open('w') as log:
            subprocess.run([sys.executable,str(Path(plan['scripts'])/'evaluate_sustained_milestone.py'),
                '--checkpoint',str(checkpoint),'--inputs',str(output/'inputs'),'--output',str(destination),
                '--workers',str(plan['evaluation_workers']),
                *(['--authority-trace'] if plan.get('authority_trace') else [])],env=environment,cwd=ROOT,
                stdout=log,stderr=subprocess.STDOUT,check=True)
        return json.loads(receipt.read_text())

    try:
        # Qualify the largest allocation first. Every production run starts
        # from the common initializer with a fresh optimizer and physical state.
        for run in reversed(plan['runs']):
            active = run['name']
            directory = output/active
            directory.mkdir(parents=True,exist_ok=True)
            state['runs'][active]['phase'] = 'preflight'
            status('preflight')
            receipt = directory/'preflight/report.json'
            if not receipt.exists():
                with (directory/'preflight.log').open('w') as log:
                    code = subprocess.run(run['preflight_command'],env=environment,cwd=ROOT,
                                          stdout=log,stderr=subprocess.STDOUT).returncode
                if code:
                    raise RuntimeError(f'{active} preflight failed with exit code {code}')
            validate_report(json.loads(receipt.read_text()),run,plan['preflight_updates'],plan['num_envs'],
                            originals=plan['originals'],require_coverage=False)
            state['runs'][active]['phase'] = 'queued'
            state['runs'][active]['preflight_passed'] = True
            status('preflight')
        active = None
        status('initializer_evaluation')
        baseline = evaluate(Path(plan['initializer']), output/'initializer')
        common_baseline = baseline
        for run in plan['runs']:
            active = run['name']
            directory = output/active
            training = directory/'training'
            reviews = directory/'reviews'
            reviews.mkdir(exist_ok=True)
            baseline_checkpoint = run.get('baseline_checkpoint', plan['initializer'])
            baseline = (evaluate(Path(baseline_checkpoint), directory/'initializer')
                        if baseline_checkpoint != plan['initializer'] else common_baseline)
            protected, champion = baseline, baseline_checkpoint
            state['runs'][active]['initial_controller_evaluation'] = baseline['summary']
            state['runs'][active]['common_initializer_evaluation'] = common_baseline['summary']

            def review_pending():
                nonlocal protected, champion
                for checkpoint in sorted(training.glob('checkpoint-[0-9]*.pt')):
                    receipt = reviews/(checkpoint.stem+'.json')
                    if receipt.exists():
                        continue
                    status('milestone_evaluation',checkpoint=str(checkpoint))
                    candidate = evaluate(checkpoint,directory/('evaluation-'+checkpoint.stem))
                    comparison = review_candidate(candidate,baseline,protected)
                    if comparison['replace_champion']:
                        protected, champion = candidate,str(checkpoint)
                    # User requested equal 8,000-update budgets. Record behavioral
                    # regressions, retain the champion, but do not confound exposure
                    # with model-dependent early stopping. Execution errors are fatal.
                    write(receipt,dict(continue_training=True,reason='fixed_capacity_comparison_budget',
                        comparison=comparison,checkpoint_sha256=candidate['checkpoint_sha256'],
                        champion=champion,behaviorally_accepted=False))
                    write(directory/'champion.json',dict(checkpoint=champion,behaviorally_accepted=False))
                    state['runs'][active].update(latest_evaluation=candidate['summary'],champion=champion)
                    status('training')

            command = list(run['command'])
            # An explicit rerun of the saved plan resumes optimizer/exposure from
            # the latest durable checkpoint; it never adds 8,000 more updates.
            checkpoint = training/'checkpoint.pt'
            if checkpoint.exists():
                import torch
                saved = torch.load(checkpoint,map_location='cpu',weights_only=True)
                completed = saved['iteration']
                del saved
                if completed > plan['updates']:
                    raise ValueError('Checkpoint exceeds the declared comparison budget')
                if completed < plan['updates']:
                    index = command.index('--initialize')
                    command[index:index+2] = ['--resume',str(checkpoint)]
                    command[command.index('--iterations')+1] = str(plan['updates']-completed)
                else:
                    command = None
                # Reconstruct selection state from the immutable saved decisions.
                prior = directory/'champion.json'
                if prior.exists():
                    champion = json.loads(prior.read_text())['checkpoint']
                    if champion != baseline_checkpoint:
                        protected = evaluate(Path(champion),directory/('evaluation-'+Path(champion).stem))
            if command is not None:
                write(directory/'actual-command.json',command)
                with (directory/'training.log').open('a') as log:
                    process = subprocess.Popen(command,env=environment,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                    state['runs'][active].update(phase='training',trainer_pid=process.pid)
                    status('training')
                    try:
                        while process.poll() is None:
                            state['runs'][active].update(latest_metrics(training/'metrics.jsonl'))
                            review_pending()
                            status('training')
                            time.sleep(1)
                    except BaseException:
                        process.terminate()
                        process.wait(timeout=60)
                        raise
                if process.returncode:
                    raise RuntimeError(f'{active} trainer failed with exit code {process.returncode}')
            review_pending()
            report = json.loads((training/'report.json').read_text())
            validate_report(report,run,plan['updates'],plan['num_envs'],originals=plan['originals'],
                            require_coverage=plan.get('require_full_coverage', True))
            state['runs'][active].update(phase='complete',iteration=report['last_metrics']['iteration'],
                transitions=report['transitions'],optimizer_steps=report['optimizer_steps'],champion=champion)
            status('between_runs')
        active = None
        for job in plan.get('post_experiments', []):
            state.setdefault('post_experiments', {})[job['name']] = {'phase': 'running'}
            status('post_experiment', post_active=job['name'])
            try:
                receipt = run_post_experiment(job,environment,ROOT)
                state['post_experiments'][job['name']] = receipt
            except BaseException as error:
                state['post_experiments'][job['name']] = dict(phase='failed', error=str(error))
                raise
        status('complete')
    except BaseException as error:
        if active:
            state['runs'][active]['phase'] = 'failed'
        status('failed',error=f'{type(error).__name__}: {error}')
        raise


def prepare(args):
    import torch
    args.output.mkdir(parents=True,exist_ok=False)
    output = args.output.resolve()
    padding = args.padded_inputs.resolve(strict=True)
    prior = args.prior_campaign.resolve(strict=True)
    rows = [json.loads(line) for line in (padding/'library/index.jsonl').read_text().splitlines()]
    all_rows = [json.loads(line) for line in (padding/'all-padded/index.jsonl').read_text().splitlines()]
    curriculum = json.loads((prior/'inputs/curriculum.json').read_text())
    verify_membership(rows,all_rows,curriculum)
    if curriculum.get('initial_coverage_sweep') is not True:
        raise ValueError('Initial full-corpus coverage sweep is required')
    audit = json.loads((padding/'report.json').read_text())
    manifest_digest = hashlib.sha256((padding/'library/index.jsonl').read_bytes()).hexdigest()
    if audit['training_manifest_sha256'] != manifest_digest or audit['invalid_ticks'] != 0:
        raise ValueError('Padding audit does not match the admitted manifest')
    inputs = output/'inputs'
    inputs.mkdir()
    (inputs/'library').symlink_to(padding/'library',target_is_directory=True)
    for name in ('curriculum.json','training-panel.json','development-panel.json'):
        shutil.copy2(prior/'inputs'/name,inputs/name)
    snapshot, revision = freeze_source(ROOT)
    # Saved script helpers add their own bundle's src directory to sys.path.
    # Pin their in-process milestone review imports to the same package too.
    (output/'src').symlink_to(snapshot,target_is_directory=True)
    scripts = output/'scripts'
    scripts.mkdir()
    # Freeze entrypoints as well as the imported controller package. Subsequent
    # workspace edits cannot change a queued size's implementation.
    for source in (ROOT/'scripts').glob('*.py'):
        shutil.copy2(source,scripts/source.name)
    initializer = output/'initialize.pt'
    shutil.copy2(args.initializer.resolve(strict=True),initializer)
    initial = torch.load(initializer,map_location='cpu',weights_only=True)
    if (initial['actor_size'] != 1680 or initial['critic_size'] != 1820
            or initial['observation'].get('preview_horizon_s') != .3
            or initial['observation'].get('playback_delay_s') != .3
            or initial['reward_settings'].get('reference_scale') is not None):
        raise ValueError('Initializer differs from the declared 300 ms buffered actor contract')
    write(output/'controller.json',initial['action_settings'])
    plan = dict(version='all-long-warp-capacity-v1',output=str(output),scripts=str(scripts),
        source_revision=revision,frozen_source=str(snapshot/'k1_motion'),
        initializer=str(initializer),initializer_sha256=hashlib.sha256(initializer.read_bytes()).hexdigest(),
        library_manifest_sha256=manifest_digest,originals=len(rows),by_family=dict(Counter(r['family'] for r in rows)),
        padding_report=str(padding/'report.json'),updates=args.updates,num_envs=args.num_envs,horizon=32,
        preflight_updates=args.preflight_updates,milestone=args.milestone,evaluation_workers=args.evaluation_workers,
        maximum_preview_seconds=.3,playback_delay_seconds=.3,seed=45,
        reward_profile='causal-balanced-v1',safety_profile='casual-safe-v1',
        training_physics='CUDA MuJoCo Warp',optimizer='CUDA PyTorch Adam',
        diagnostic_evaluator='identical native MuJoCo exported-actor panels; not training physics',
        ancillary_cpu_work='arm collision target projection, data orchestration, diagnostic replay',
        stopping='exact equal update budgets; fail on execution/nonfinite/overflow errors; retain behavioral champions',
        confirmation_panel_used=False,behaviorally_accepted=False,hardware_verified=False)
    plan['runs'] = build_commands(output,scripts,padding/'library',(padding/'reference-cache.pt').resolve(strict=True),
        initializer,updates=args.updates,num_envs=args.num_envs,milestone=args.milestone,seed=45,
        preflight_updates=args.preflight_updates)
    write(output/'plan.json',plan)
    write(output/'status.json',dict(phase='prepared',runs={r['name']:{'phase':'queued'} for r in plan['runs']}))
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--padded-inputs',type=Path)
    parser.add_argument('--prior-campaign',type=Path)
    parser.add_argument('--initializer',type=Path)
    parser.add_argument('--updates',type=int,default=8000)
    parser.add_argument('--num-envs',type=int,default=2048)
    parser.add_argument('--milestone',type=int,default=1000)
    parser.add_argument('--preflight-updates',type=int,default=3)
    parser.add_argument('--evaluation-workers',type=int,default=6)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--run-plan',type=Path)
    args = parser.parse_args()
    if args.run_plan:
        plan = json.loads(args.run_plan.read_text())
    else:
        if not all((args.output,args.padded_inputs,args.prior_campaign,args.initializer)):
            parser.error('Preparation requires output, padded-inputs, prior-campaign and initializer')
        plan = prepare(args)
    if not args.prepare_only:
        environment = {**os.environ,'K1_MOTION_ROOT':str(ROOT),'K1_FROZEN_SOURCE':plan['frozen_source'],
                       'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
        run_queue(plan,environment)


if __name__ == '__main__':
    main()
