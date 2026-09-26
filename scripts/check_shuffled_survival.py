#!/usr/bin/env python3
"""Real-library handoff/physics diagnostic and bounded PPO, with saved receipts."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

from freeze_source import freeze_source

ROOT = Path(__file__).resolve().parents[1]


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('library', 'cache', 'initializer', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--settings', type=Path, default=ROOT/'configs/shuffled-scenes-120s.json')
    p.add_argument('--num-envs', type=int, default=32)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seconds', type=float, default=120.)
    p.add_argument('--train-updates', type=int, default=8)
    p.add_argument('--seed', type=int, default=246)
    p.add_argument('--reward-profile', choices=('survival-position-v1', 'survival-position-v2'),
                   default='survival-position-v2')
    args = p.parse_args()
    if min(args.num_envs, args.workers, args.train_updates) <= 0 or not math.isfinite(args.seconds) or args.seconds <= 0:
        p.error('Positive finite diagnostic budgets required')
    args.output.mkdir(parents=True, exist_ok=False)
    snapshot, revision = freeze_source(ROOT)
    os.environ['K1_MOTION_ROOT'] = str(ROOT)
    os.environ['K1_SOURCE_REVISION'] = revision
    sys.path.insert(0, str(snapshot))
    import torch
    from k1_motion.learning import ActorCritic, TrainConfig, train
    from k1_motion.model_transfer import checkpoint_hidden_sizes
    from k1_motion.tracking_env import TrackerEnv
    from k1_motion.training_validation import checkpoint_environment_settings
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    saved = torch.load(args.initializer, map_location='cpu', weights_only=True)
    options = checkpoint_environment_settings(saved)
    if saved['stage'] != 'student' or options.get('reference_scale') is not None:
        raise ValueError('Use an original-target student initializer')
    options.pop('world_reward_settings', None)
    options.update(reward_profile=args.reward_profile, safety_profile='casual-safe-v1',
                   self_collision_weight=0., tracking_huber=False, first_collision_penalty=0.)
    settings = json.loads(args.settings.read_text())
    env = TrackerEnv(args.library, args.num_envs, 'cpu', 'mujoco_cpp', **options,
        reference_cache=args.cache, physics_options={'workers': args.workers}, scene_transitions=settings)
    model = ActorCritic(saved['actor_size'], saved['critic_size'], checkpoint_hidden_sizes(saved))
    model.load_state_dict(saved['model'])
    model.eval()
    gamma = math.exp(-env.spec.control_dt/60.)
    plan = dict(version='shuffled-survival-diagnostic-v1', source_revision=revision,
        source_path=str(snapshot), initializer=str(args.initializer.resolve()),
        initializer_sha256=hashlib.sha256(args.initializer.read_bytes()).hexdigest(),
        library=str(args.library.resolve()), library_fingerprint=env.library.fingerprint,
        originals=len(env.library.rows), families=dict(Counter(r['family'] for r in env.library.rows)),
        playback_hours_including_padding=sum((int(n)-1)*env.spec.control_dt for n in env.library.lengths)/3600,
        original_hours=sum(r.get('standing_padding', {}).get('original_duration_s',
            (int(n)-1)*env.spec.control_dt) for r, n in zip(env.library.rows, env.library.lengths))/3600,
        cache=str(args.cache.resolve()), cache_payload_rehashed=False,
        scene_transitions=env.scene_transition_contract, rewards=env.reward_settings,
        gamma=gamma, gae_lambda=.99, discount_time_constant_s=60.,
        curriculum=None, num_envs=args.num_envs, seconds_per_world=args.seconds, seed=args.seed,
        scope='Training originals only; mechanics and finite learning, no held-out acceptance or promotion')
    write(args.output/'plan.json', plan)
    try:
        # Bounded candidate availability across every source/next role. Reuse
        # resident tensors, without stepping physics or changing admissions.
        availability = {}
        manager = env.scene_transitions
        for role in manager.roles:
            slot = env.scene_transition_contract['pattern'].index(role)
            eligible_count = available_count = 0
            begin = time.perf_counter()
            for offset in range(0, len(env.library.rows), env.num_envs):
                count = min(env.num_envs, len(env.library.rows)-offset)
                env.clips[:] = torch.arange(env.num_envs) % count+offset
                manager.slots[:] = (slot-1) % len(manager.pattern)
                pending = manager.prepare(torch.arange(env.num_envs) < count)
                eligible_count += count
                available_count += len(pending['ids'])
            availability[role] = dict(sources=eligible_count, compatible_draws=available_count,
                unavailable=eligible_count-available_count, elapsed_s=time.perf_counter()-begin)
        write(args.output/'handoff-availability.json', availability)
        torch.manual_seed(args.seed)
        causal, _ = env.reset()
        totals, durations, collision_ticks = Counter(), [], 0
        begin = time.perf_counter()
        ticks = round(args.seconds/env.spec.control_dt)
        with torch.inference_mode():
            for tick in range(ticks):
                lengths = env.episode_steps.clone()+1
                causal, _, reward, done, info = env.step(model.actor(causal).tanh())
                for key in ('ended_episodes', 'ended_episode_steps', 'falls', 'tracking_failures',
                            'scene_transition_starts', 'scene_transition_unavailable', 'scene_episode_limits',
                            'scene_bridge_steps'):
                    totals[key] += int(info[key])
                totals['legacy_tracking_violation_world_ticks'] += round(float(
                    info['reward_components'].get('tracking/legacy_violation', 0.))*env.num_envs)
                totals['negative_nonfall_reward_world_ticks'] += int(((reward < 0) & ~env.last_step['fallen']).sum())
                durations.extend((lengths[done]*env.spec.control_dt).tolist())
                collision_ticks += round(float(info['self_collision_fraction'])*env.num_envs)
                if (tick+1) % 250 == 0:
                    write(args.output/'status.json', dict(phase='real_motion_rollout', tick=tick+1,
                        total_ticks=ticks, handoffs=totals['scene_transition_starts'], falls=totals['falls']))
        result = dict(**totals, collision_world_ticks=collision_ticks, execution_errors=0,
            mean_ended_episode_s=sum(durations)/max(1, len(durations)),
            longest_ended_episode_s=max(durations, default=0.),
            active_episode_seconds=(env.episode_steps*env.spec.control_dt).tolist(),
            elapsed_s=time.perf_counter()-begin, physical_transitions=ticks*env.num_envs,
            transitions_per_s=ticks*env.num_envs/(time.perf_counter()-begin),
            behaviorally_accepted=False, raw_source_completions_not_scene_success=True)
        write(args.output/'rollout.json', result)
        write(args.output/'status.json', dict(phase='ppo_smoke'))
        # The loaded tensors stay resident, while replay exposure must not be
        # charged to the learner's first rollout or sampler duration estimates.
        for key in ('transition_count', 'episode_count', 'episode_steps'):
            getattr(env.library, key).zero_()
        report = train(env, args.output/'training', TrainConfig(stage='student', iterations=args.train_updates,
            horizon=128, epochs=1, minibatch=args.num_envs*128, hidden_sizes=checkpoint_hidden_sizes(saved),
            gamma=gamma, gae_lambda=.99, bc_weight=0, evaluation_interval=0,
            learning_rate=1e-5, checkpoint_interval=args.train_updates, milestone_interval=args.train_updates,
            seed=args.seed), initialize_checkpoint=args.initializer)
        if not report['finite_updates'] or report['checkpoint_reload_max_error'] != 0:
            raise ValueError('PPO finite/reload check failed')
        if report['last_metrics']['reference_exposure']['total_transitions'] > report['transitions']:
            raise ValueError('Diagnostic replay contaminated training exposure')
        metrics = [json.loads(line) for line in (args.output/'training/metrics.jsonl').read_text().splitlines()]
        write(args.output/'status.json', dict(phase='complete', transitions=report['transitions'],
            optimizer_steps=report['optimizer_steps'], finite_updates=report['finite_updates'],
            checkpoint_reload_max_error=report['checkpoint_reload_max_error'],
            training_handoffs=sum(m['scene_transitions']['started'] for m in metrics),
            reference_exposure=report['last_metrics']['reference_exposure'],
            reward_profile=args.reward_profile, curriculum=None,
            behaviorally_accepted=False, hardware_verified=False))
        print(json.dumps(dict(rollout=result, training=report['last_metrics']), indent=2))
    finally:
        env.close()


if __name__ == '__main__':
    main()
