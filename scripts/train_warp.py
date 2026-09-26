#!/usr/bin/env python3
"""Train the shared K1 task with GPU MuJoCo, without a graphical simulator runtime."""

import argparse
import json
import os
from pathlib import Path
import sys
from freeze_source import freeze_source

ROOT = Path(os.environ.get('K1_MOTION_ROOT', Path(__file__).resolve().parents[1]))
frozen_source, source_revision = freeze_source(ROOT, os.environ.get("K1_FROZEN_SOURCE"))
os.environ["K1_MOTION_ROOT"] = str(ROOT)
os.environ["K1_SOURCE_REVISION"] = source_revision
sys.path.insert(0, str(frozen_source))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--backend", choices=("warp", "mujoco_cpp"), default="warp")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-workers", type=int, default=32)
    parser.add_argument("--cpu-chunk-size", type=int, default=4)
    parser.add_argument("--cpu-unfused-arm-feedback", action="store_true",
                        help="Diagnostic: keep the separate arm projection/device-transfer path")
    parser.add_argument("--stage", choices=("teacher", "student"), default="teacher")
    parser.add_argument("--teacher")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--max-seconds", type=float, help="Finish the current update and save on wall-time budget exhaustion")
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument('--gamma', type=float, default=.99, help='Discount per control tick')
    parser.add_argument('--gae-lambda', type=float, default=.95)
    parser.add_argument("--history", type=int, default=10)
    parser.add_argument("--observation-profile", choices=("causal", "planar", "preview"), default="causal")
    parser.add_argument("--preview-horizon-s", type=float, choices=(0., .3), default=.3)
    parser.add_argument('--command-feedback', action='store_true')
    parser.add_argument("--safety-profile", choices=('casual-safe-v1',))
    parser.add_argument("--tracking-huber", action="store_true")
    parser.add_argument("--first-collision-penalty", type=float, default=0.)
    parser.add_argument("--arm-workers", type=int, default=8)
    parser.add_argument("--hidden-sizes", type=int, nargs=2, default=(512, 256))
    parser.add_argument('--action-chunk-size', type=int, default=1)
    parser.add_argument("--matmul-precision", choices=("highest", "high"), default="high")
    parser.add_argument(
        "--sampling", choices=("episode_balanced", "transition_balanced", "take_transition_balanced"),
        default="transition_balanced"
    )
    parser.add_argument("--corruption", action="store_true")
    parser.add_argument("--resume")
    parser.add_argument("--allow-env-resize", action="store_true",
                        help="Explicitly resume optimizer/model with a new environment count; retain exposure totals")
    parser.add_argument("--initialize")
    parser.add_argument("--minibatch", type=int, default=16384)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-learning-rate", type=float, default=1e-3)
    parser.add_argument("--kl-stop", type=float)
    parser.add_argument("--curriculum-manifest")
    parser.add_argument("--ppo-update-lock", help="Shared local file for optional per-GPU PPO scheduling")
    parser.add_argument("--bc-weight", type=float, default=1.0)
    parser.add_argument("--self-collision-weight", type=float, default=0.0)
    parser.add_argument("--root-velocity-weight", type=float)
    parser.add_argument("--root-velocity-sigma", type=float)
    from k1_motion.tracking_rewards import REWARD_PROFILES
    parser.add_argument("--reward-profile", choices=REWARD_PROFILES, default="legacy")
    parser.add_argument("--world-reward-settings", help="JSON weights/scales for configurable world reward profiles")
    parser.add_argument("--reference-scale", type=float)
    parser.add_argument("--reference-cache", help="Verified immutable CPU reference tensors; independent sampler per run")
    parser.add_argument('--scene-transitions', help='JSON settings for uninterrupted scene handoffs (opt-in)')
    parser.add_argument("--residual-scale", type=float)
    parser.add_argument("--command-velocity-limit", type=float)
    parser.add_argument("--action-settings", help="JSON controller settings; stored in checkpoints and exports")
    parser.add_argument("--candidate-training", action="store_true",
                        help="Allow explicitly audited simulation candidates without claiming physics qualification")
    parser.add_argument("--reference-storage", choices=("padded", "packed"), default="packed")
    parser.add_argument("--evaluation-interval", type=int, default=250)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--milestone-interval", type=int, default=500)
    parser.add_argument('--milestone-review-directory', help='External fixed-panel review receipts; wait before continuing')
    parser.add_argument('--milestone-review-timeout-s', type=float, default=900.)
    parser.add_argument("--nconmax", type=int, default=64)
    parser.add_argument("--njmax", type=int, default=256)
    parser.add_argument("--epa-horizon", type=int)
    parser.add_argument("--no-conditional-graphs", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    import torch
    from k1_motion.learning import TrainConfig, train
    from k1_motion.tracking_env import TrackerEnv

    torch.set_num_threads(args.threads)
    settings = json.loads(Path(args.action_settings).read_text()) if args.action_settings else {}
    settings.update({name: value for name, value in (
        ("residual_scale", args.residual_scale),
        ("command_velocity_limit", args.command_velocity_limit),
    ) if value is not None})
    env = TrackerEnv(
        args.library,
        args.num_envs,
        args.device,
        args.backend,
        history=args.history,
        observation_profile=args.observation_profile,
        preview_horizon_s=args.preview_horizon_s,
        command_feedback=args.command_feedback,
        safety_profile=args.safety_profile,
        tracking_huber=args.tracking_huber,
        first_collision_penalty=args.first_collision_penalty,
        arm_workers=args.arm_workers,
        corruption=args.corruption,
        action_settings=settings,
        candidate_training=args.candidate_training,
        reference_storage=args.reference_storage,
        root_velocity_weight=args.root_velocity_weight,
        root_velocity_sigma=args.root_velocity_sigma,
        reward_profile=args.reward_profile,
        world_reward_settings=(json.loads(Path(args.world_reward_settings).read_text())
                               if args.world_reward_settings else None),
        reference_cache=args.reference_cache,
        scene_transitions=(json.loads(Path(args.scene_transitions).read_text()) if args.scene_transitions else None),
        reference_scale=args.reference_scale,
        **({"self_collision_weight": args.self_collision_weight} if args.self_collision_weight else {}),
        physics_options=({
            "nconmax": args.nconmax,
            "njmax": args.njmax,
            "conditional_graphs": not args.no_conditional_graphs,
            **({"epa_horizon": args.epa_horizon} if args.epa_horizon is not None else {}),
        } if args.backend == "warp" else {"workers": args.cpu_workers, "chunk_size": args.cpu_chunk_size,
                                          "fuse_arm_feedback": not args.cpu_unfused_arm_feedback}),
    )
    try:
        train(
            env,
            args.output,
            TrainConfig(
                stage=args.stage,
                iterations=args.iterations,
                horizon=args.horizon,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                minibatch=args.minibatch,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                min_learning_rate=args.min_learning_rate,
                max_learning_rate=args.max_learning_rate,
                kl_stop=args.kl_stop,
                curriculum_manifest=args.curriculum_manifest,
                ppo_update_lock=args.ppo_update_lock,
                bc_weight=args.bc_weight,
                evaluation_interval=args.evaluation_interval,
                checkpoint_interval=args.checkpoint_interval,
                seed=args.seed,
                hidden_sizes=tuple(args.hidden_sizes),
                action_chunk_size=args.action_chunk_size,
                matmul_precision=args.matmul_precision,
                sampling=args.sampling,
                allow_env_resize=args.allow_env_resize,
                milestone_interval=args.milestone_interval,
                milestone_review_directory=args.milestone_review_directory,
                milestone_review_timeout_s=args.milestone_review_timeout_s,
                max_seconds=args.max_seconds,
            ),
            args.teacher,
            args.resume,
            args.initialize,
        )
    finally:
        env.close()
        receipt = Path(args.output) / "source.json"
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(
            json.dumps({"revision": source_revision, "path": str(frozen_source)}, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
