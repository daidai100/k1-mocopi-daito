#!/usr/bin/env python3
"""Isolated native replay to qualify a declared Warp allocation change."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

from freeze_source import freeze_source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--library", required=True)
    parser.add_argument("--source-directory", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epa-horizon", type=int)
    parser.add_argument("--num-envs", type=int, default=288)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source, revision = freeze_source(root, args.source_directory)
    sys.path.insert(0, str(source))
    os.environ["K1_MOTION_ROOT"] = str(root)
    import torch
    from k1_motion.action_chunks import make_model, checkpoint_chunk_size
    from k1_motion.tracking_env import TrackerEnv
    from k1_motion.training_validation import replay_panel, checkpoint_environment_settings

    torch.set_num_threads(2)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    torch.set_float32_matmul_precision(checkpoint["config"].get("matmul_precision", "high"))
    model = make_model(
        checkpoint["actor_size"], checkpoint["critic_size"], tuple(checkpoint["hidden_sizes"]),
        checkpoint_chunk_size(checkpoint)
    ).to("cuda:0")
    model.load_state_dict(checkpoint["model"])
    model.eval()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "checkpoint": args.checkpoint,
        "checkpoint_iteration": checkpoint["iteration"],
        "checkpoint_source_revision": checkpoint["source_revision"],
        "replay_source_revision": revision,
        "source_change_declared": True,
        "library": args.library,
        "started_unix": time.time(),
        "behaviorally_accepted": False,
        "scope": "Isolated GPU checkpoint diagnostic and allocation qualification; no acceptance result",
    }
    env = None
    try:
        env = TrackerEnv(
            args.library,
            args.num_envs,
            "cuda:0",
            "warp",
            **checkpoint_environment_settings(checkpoint),
            physics_options={
                "nconmax": 256,
                "njmax": 2048,
                "conditional_graphs": False,
                **({"epa_horizon": args.epa_horizon} if args.epa_horizon is not None else {}),
            },
        )
        if env.spec.signature != checkpoint["model_signature"] or env.task_version != checkpoint["task_version"]:
            raise ValueError("Checkpoint robot/task contract mismatch")
        rows = [json.loads(line) for line in (Path(args.library) / "index.jsonl").read_text().splitlines()]
        if set(checkpoint["train_parents"]) & {r["capture_group"] for r in rows if r["split"] == "validation"}:
            raise ValueError("Validation captures overlap training provenance")
        report["physics_contract"] = env.physics.contract
        report["results"] = {}
        for split in ("train", "validation"):
            report["results"][split] = replay_panel(env, model, checkpoint["stage"], args.library, split)
            output.with_suffix(".partial.json").write_text(json.dumps(report, indent=2) + "\n")
        env.close()
        env = None
        report["execution_passed"] = True
    except Exception as error:
        report.update({"execution_passed": False, "error": f"{type(error).__name__}: {error}"})
        raise
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as error:
                report["close_error"] = f"{type(error).__name__}: {error}"
        report["finished_unix"] = time.time()
        output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2))
    print(json.dumps({split: {k: r[k] for k in ("completed", "total")} for split, r in report["results"].items()}))


if __name__ == "__main__":
    main()
