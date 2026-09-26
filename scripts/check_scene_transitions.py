#!/usr/bin/env python3
"""Compare bounded native rollouts, then smoke-test learning on training scenes."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from freeze_source import freeze_source

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--panel', type=Path, required=True, help='Training-only diagnostic membership')
    parser.add_argument('--initializer', type=Path, required=True)
    parser.add_argument('--settings', type=Path, default=ROOT/'configs/scene-transitions-v1.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--num-envs', type=int, default=32)
    parser.add_argument('--seconds', type=float, default=30.)
    parser.add_argument('--train-updates', type=int, default=25)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=24)
    args = parser.parse_args()
    if min(args.num_envs, args.seconds, args.train_updates, args.workers) <= 0:
        parser.error('All budgets must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    snapshot, revision = freeze_source(ROOT)
    os.environ['K1_MOTION_ROOT'] = str(ROOT)
    os.environ['K1_SOURCE_REVISION'] = revision
    sys.path.insert(0, str(snapshot))
    import torch
    from k1_motion.learning import ActorCritic, MotionLibrary, TrainConfig, train
    from k1_motion.model_transfer import checkpoint_hidden_sizes
    from k1_motion.robot import K1Model
    from k1_motion.tracking_env import TrackerEnv
    from k1_motion.training_validation import checkpoint_environment_settings
    torch.set_num_threads(1)
    panel = json.loads(args.panel.read_text())
    selected = {r['id'] for r in panel}
    if len(selected) != len(panel) or any(r['split'] != 'train' for r in panel):
        raise ValueError('Diagnostic panel must contain unique training recordings')
    rows = []
    with (args.library/'index.jsonl').open() as source:
        for line in source:
            row = json.loads(line)
            if row['id'] in selected:
                if row['split'] != 'train':
                    raise ValueError('Held-out source in training diagnostic')
                row['reference_path'] = str((args.library/row['reference_path']).resolve(strict=True))
                rows.append(row)
    if len(rows) != len(selected):
        raise ValueError('Training diagnostic membership is missing from the library')
    library_path = args.output/'library'
    library_path.mkdir()
    manifest = ''.join(json.dumps(row, sort_keys=True)+'\n' for row in sorted(rows, key=lambda r:r['id']))
    (library_path/'index.jsonl').write_text(manifest)
    saved = torch.load(args.initializer, map_location='cpu', weights_only=True)
    settings = json.loads(args.settings.read_text())
    options = checkpoint_environment_settings(saved)
    if options.get('reference_scale') is not None or saved['stage'] != 'student':
        raise ValueError('Use an original-target student initializer')
    options.update(reward_profile='world-velocity-v1', safety_profile='casual-safe-v1')
    model = ActorCritic(saved['actor_size'], saved['critic_size'], checkpoint_hidden_sizes(saved))
    model.load_state_dict(saved['model'])
    model.eval()
    library = MotionLibrary(library_path, K1Model(), 'cpu', storage='packed')
    plan = dict(source_revision=revision, initializer=str(args.initializer.resolve()),
        initializer_sha256=hashlib.sha256(args.initializer.read_bytes()).hexdigest(),
        panel_sha256=hashlib.sha256(args.panel.read_bytes()).hexdigest(),
        library_manifest_sha256=hashlib.sha256(manifest.encode()).hexdigest(),
        originals=len(rows), source_ids=sorted(selected), settings=settings,
        num_envs=args.num_envs, seconds_per_world=args.seconds, seed=args.seed,
        reset='recording start', role='training diagnostic; no held-out selection or promotion',
        training='fresh optimizer from common initializer; short finite/reload smoke only')
    write(args.output/'plan.json', plan)

    def new_env(enabled):
        # Keep reference buffers resident; only sampler state is run-specific.
        independent = copy.copy(library)
        for key, value in library.__dict__.items():
            if torch.is_tensor(value):
                setattr(independent, key, value.clone())
        env = TrackerEnv(library_path, args.num_envs, 'cpu', 'mujoco_cpp', library=independent,
                        physics_options={'workers': args.workers}, **options,
                        scene_transitions=settings if enabled else None)
        reset = env.reset
        def from_start(ids=None, clips=None, frames=None):
            n = env.num_envs if ids is None else len(ids)
            return reset(ids, clips, torch.zeros(n, dtype=torch.long) if frames is None else frames)
        env.reset = from_start
        return env

    results = {}
    for enabled, name in ((False, 'control'), (True, 'transitions')):
        torch.manual_seed(args.seed)
        env = new_env(enabled)
        try:
            causal, _ = env.reset(clips=torch.arange(args.num_envs) % len(rows))
            totals = {}
            collision_world_ticks = 0.
            durations = []
            wall = time.monotonic()
            ticks = round(args.seconds/env.spec.control_dt)
            with torch.inference_mode():
                for step in range(ticks):
                    causal, _, _, done, info = env.step(model.actor(causal).tanh())
                    for key in ('ended_episodes', 'ended_episode_steps', 'completed', 'falls', 'tracking_failures',
                                'scene_transition_starts', 'scene_transition_completions', 'scene_transition_failures',
                                'scene_transition_unavailable', 'scene_episode_limits', 'scene_bridge_steps'):
                        totals[key] = totals.get(key, 0)+int(info.get(key, 0))
                    totals['ended_episode_bridge_steps'] = totals.get('ended_episode_bridge_steps', 0)+int(
                        info.get('ended_episode_bridge_steps', 0))
                    collision_world_ticks += float(info['self_collision_fraction'])*env.num_envs
                    if info['ended_episodes']:
                        durations.append(dict(step=step, ended=int(done.sum()),
                                              duration_sum_s=float(info['ended_episode_steps'])*env.spec.control_dt))
                    if (step+1) % 32 == 0:
                        env.library.update_sampling()
                    if (step+1) % 250 == 0:
                        write(args.output/'status.json', dict(phase='rollout', arm=name, step=step+1, total=ticks))
            results[name] = dict(**totals,
                mean_ended_episode_s=env.spec.control_dt*totals['ended_episode_steps']/max(1, totals['ended_episodes']),
                mean_ended_source_s=env.spec.control_dt*(totals['ended_episode_steps']
                    -totals['ended_episode_bridge_steps'])/max(1, totals['ended_episodes']),
                self_collision_world_ticks=round(collision_world_ticks),
                active_episode_seconds=(env.episode_steps*env.spec.control_dt).tolist(),
                elapsed_seconds=time.monotonic()-wall, physical_steps=ticks*args.num_envs,
                bridge_rejections=dict(env.scene_transitions.rejections) if enabled else {},
                scope='Physical survival and source completions; not clean task success')
            write(args.output/(name+'-episode-ends.json'), durations)
            write(args.output/'comparison.json', results)
            print(json.dumps(dict(arm=name, **results[name])), flush=True)
        finally:
            env.close()
    env = new_env(True)
    try:
        write(args.output/'status.json', dict(phase='ppo_smoke'))
        result = train(env, args.output/'training', TrainConfig(stage='student', iterations=args.train_updates,
            horizon=64, epochs=1, minibatch=args.num_envs*64, hidden_sizes=checkpoint_hidden_sizes(saved),
            evaluation_interval=0, checkpoint_interval=1, milestone_interval=1, bc_weight=0,
            learning_rate=1e-5, seed=args.seed), initialize_checkpoint=args.initializer)
        results['training_smoke'] = {k: result[k] for k in ('transitions', 'optimizer_steps', 'finite_updates',
            'checkpoint_reload_max_error', 'scene_transitions')}
        metrics = [json.loads(line) for line in (args.output/'training/metrics.jsonl').read_text().splitlines()]
        results['training_smoke']['handoffs_started'] = sum(r['scene_transitions']['started'] for r in metrics)
        results['training_smoke']['handoffs_completed'] = sum(r['scene_transitions']['completed'] for r in metrics)
        write(args.output/'comparison.json', results)
        write(args.output/'status.json', dict(phase='complete', behaviorally_accepted=False, hardware_verified=False,
            learning_encountered_handoffs=results['training_smoke']['handoffs_completed'] > 0))
    finally:
        env.close()


if __name__ == '__main__':
    main()
