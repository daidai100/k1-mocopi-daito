"""Deterministic, uninterrupted native-physics replay during training.

This diagnostic deliberately reports native completion and tracking errors. The
independent exported-student acceptance panel also checks slip and collisions.
The same simulator stays loaded; returning to training creates declared resets.
"""

import json
from pathlib import Path

import torch

from .learning import MotionLibrary


def checkpoint_environment_settings(saved):
    """Reconstruct observation, action and objective options for native replay."""
    from .observations import observation_profile
    reward = saved.get('reward_settings', {})
    observation = saved['observation']
    options = dict(history=observation['history'],
        command_feedback='command_feedback' in observation,
        observation_profile=observation_profile(observation),
        preview_horizon_s=observation.get('preview_horizon_s', .3),
        action_settings=saved.get('action_settings'),
        reward_profile=reward.get('profile', 'legacy'),
        safety_profile=(reward['safety']['version'] if reward.get('safety') else None),
        self_collision_weight=reward.get('self_collision_weight', 0.),
        tracking_huber=bool(reward.get('tracking_huber')),
        first_collision_penalty=reward.get('first_collision_penalty', 0.),
        reference_storage='packed', candidate_training=saved.get('candidate_training', False))
    if reward.get('reference_scale'):
        options['reference_scale'] = reward['reference_scale']['scale']
    spatial = reward.get('spatial_tracking', {})
    if spatial.get('version') in ('world-decomposed-v1', 'world-pointwise-root-v1', 'world-capped-root-v1', 'survival-position-v2'):
        options['world_reward_settings'] = {k: spatial[k] for k in ('weights', 'scales')}
    for key in ('root_velocity_weight', 'root_velocity_sigma'):
        if key in reward:
            options[key] = reward[key]
    return options


@torch.no_grad()
def replay_panel(env, model, stage, directory, split="train"):
    original, corruption = env.library, env.corruption
    transitions = getattr(env, 'scene_transitions', None)
    if split == "train" and str(directory) == env.library_directory:
        library = original
    else:
        if not hasattr(env, "evaluation_libraries"):
            env.evaluation_libraries = {}
        key = (str(directory), split)
        if key not in env.evaluation_libraries:
            # Validation grows with the corpus too. Padding the entire held-out
            # set to its longest clip can exceed VRAM even when training fits.
            candidate = MotionLibrary(
                directory, env.spec, env.device, split, storage=original.storage,
                candidate_training=original.candidate_training)
            if env.reward_settings.get("reference_scale"):
                from .reference_scale import scale_library
                contract = scale_library(candidate, env.spec,
                                         env.reward_settings["reference_scale"]["scale"])
                if contract != env.reward_settings["reference_scale"]:
                    raise ValueError("Evaluation reference scale contract differs")
            env.evaluation_libraries[key] = candidate
        library = env.evaluation_libraries[key]
    trials = []
    env.library, env.corruption = library, False
    env.scene_transitions = None  # Frozen single-recording panels keep their original meaning.
    try:
        for offset in range(0, len(library.rows), env.num_envs):
            from .action_chunks import ChunkPlayback
            playback = ChunkPlayback(model,env.num_envs,env.device)
            count = min(env.num_envs, len(library.rows) - offset)
            indices = torch.arange(env.num_envs, device=env.device) % count + offset
            causal, privileged = env.reset(clips=indices, frames=torch.zeros_like(indices))
            alive = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
            completed = torch.zeros_like(alive)
            fallen = torch.zeros_like(alive)
            ticks = torch.zeros(env.num_envs, device=env.device)
            qerr, body_error, angle = [torch.zeros_like(ticks) for _ in range(3)]
            for _ in range(int(library.lengths[indices].max().item())):
                obs = privileged if stage == "teacher" else causal
                causal, privileged, _, done, _ = env.step(
                    playback.action(model,obs), auto_reset=False, actuation_mask=alive
                )
                data = env.last_step
                ticks += alive
                qerr += data["qerr"] * alive
                body_error += data["body_error"] * alive
                angle += data["angle"] * alive
                completed |= alive & data["clip_end"] & ~data["failed"]
                fallen |= alive & data["fallen"]
                alive &= ~done
                if not alive.any():
                    break
            if hasattr(env.physics, "validate"):
                env.physics.validate()
            for i in range(count):
                row = library.rows[offset + i]
                trials.append(
                    {
                        "id": row.get("id", row["capture_group"]),
                        "family": row["family"],
                        "capture_group": row["capture_group"],
                        "completed": bool(completed[i]),
                        "fell": bool(fallen[i]),
                        "simulated_s": float(ticks[i] * env.spec.control_dt),
                        "duration_s": float((library.lengths[offset + i] - 1) * env.spec.control_dt),
                        "joint_rmse_rad": float((qerr[i] / ticks[i]).sqrt()),
                        "body_rmse_m": float((body_error[i] / ticks[i]).sqrt()),
                        "orientation_error_rad": float(angle[i] / ticks[i]),
                        "resets_during_trial": 0,
                    }
                )
    finally:
        env.library, env.corruption = original, corruption
        env.scene_transitions = transitions
    return {
        "reference_scale": env.reward_settings.get("reference_scale"),
        "split": split,
        "stage": stage,
        "backend": env.backend_name,
        "action_settings": env.action_settings,
        "completed": sum(r["completed"] for r in trials),
        "total": len(trials),
        "trials": trials,
        "behaviorally_accepted": False,
        "scope": "Native physics development diagnostic; no early termination resets within trials.",
    }


def evaluate_training(env, model, stage, directory, output, iteration):
    evaluation = env
    if env.backend_name == "warp":
        if not hasattr(env, "evaluation_environment"):
            from .tracking_env import TrackerEnv

            # Long replays leave failed bodies on the floor without resetting
            # them. Their contacts need more room than short training episodes.
            # Evaluate each recording with a small persistent pool, rather than
            # duplicating the panel across thousands of training worlds.
            contract = env.physics.contract
            size = min(512, max(32, ((len(env.library.rows) + 31) // 32) * 32))
            env.evaluation_environment = TrackerEnv(
                directory,
                num_envs=size,
                device=env.device,
                backend="warp",
                **{k: v for k, v in checkpoint_environment_settings({
                    'observation': env.observation,
                    'reward_settings': env.reward_settings,
                    'action_settings': env.action_settings,
                    'candidate_training': env.library.candidate_training,
                }).items() if k != 'reference_scale'},
                arm_workers=env.arm_workers,
                library=env.library,
                physics_options={
                    "nconmax": max(256, contract["nconmax"]),
                    "njmax": max(2048, contract["njmax"]),
                    "conditional_graphs": contract["conditional_graphs"],
                    **({"epa_horizon": contract["epa_horizon"]} if "epa_horizon" in contract else {}),
                },
            )
        evaluation = env.evaluation_environment
        if env.reward_settings.get("reference_scale"):
            evaluation.reward_settings["reference_scale"] = env.reward_settings["reference_scale"]
    results = {}
    for split in ("train", "validation"):
        rows = [json.loads(s) for s in (Path(directory) / "index.jsonl").read_text().splitlines()]
        if not any(r["kinematics_accepted"] and r["split"] == split for r in rows):
            continue
        results[split] = replay_panel(evaluation, model, stage, directory, split)
        results[split]["physics_contract"] = getattr(evaluation.physics, "contract", None)
        results[split]["evaluation_worlds"] = evaluation.num_envs
    path = Path(output) / "native-evaluations"
    path.mkdir(exist_ok=True)
    (path / f"iteration-{iteration:06d}.json").write_text(json.dumps(results, indent=2) + "\n")
    return results
