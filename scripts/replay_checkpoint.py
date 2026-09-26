#!/usr/bin/env python3
"""Independent CPU MuJoCo diagnostic with saved task settings and a declared current evaluator."""

import argparse
import json
import os
from pathlib import Path
import sys

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--library", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--num-envs", type=int, default=32)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    torch.set_num_threads(1)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    from freeze_source import freeze_source
    source, revision = freeze_source(root)
    sys.path.insert(0, str(source))
    os.environ["K1_MOTION_ROOT"] = str(root)
    from k1_motion.tracking_env import TrackerEnv
    from k1_motion.action_chunks import make_model, checkpoint_chunk_size
    from k1_motion.model_transfer import checkpoint_hidden_sizes
    from k1_motion.training_validation import replay_panel, checkpoint_environment_settings

    model = make_model(checkpoint['actor_size'],checkpoint['critic_size'],checkpoint_hidden_sizes(checkpoint),
                       checkpoint_chunk_size(checkpoint))
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    rows = [json.loads(s) for s in (Path(args.library) / "index.jsonl").read_text().splitlines()]
    selected = [r for r in rows if r["split"] == args.split]
    from k1_motion.reference_admission import take_family
    if args.split != "train" and {take_family(p) for p in checkpoint["train_parents"]} & {take_family(r["capture_group"]) for r in selected}:
        raise ValueError("Evaluation recording appeared in checkpoint training provenance")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    env = TrackerEnv(args.library, args.num_envs, "cpu",
                     **checkpoint_environment_settings(checkpoint))
    try:
        if (checkpoint["model_signature"] != env.spec.signature or checkpoint["task_version"] != env.task_version
                or checkpoint['observation'] != env.observation):
            raise ValueError("Checkpoint physics/task/observation contract mismatch")
        if checkpoint.get("reward_settings", {"self_collision_weight": 0.0}) != env.reward_settings:
            raise ValueError("Checkpoint reward/reference scale contract differs from evaluator")
        report = replay_panel(env, model, checkpoint["stage"], args.library, args.split)
    finally:
        env.close()
    report.update(
        {
            "checkpoint": args.checkpoint,
            "checkpoint_iteration": checkpoint["iteration"],
            "checkpoint_source_revision": checkpoint.get("source_revision"),
            "evaluation_source_revision": revision,
            "source_change_declared": checkpoint.get("source_revision") != revision,
            "library": args.library,
        }
    )
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "trials"}, indent=2))


if __name__ == "__main__":
    main()
