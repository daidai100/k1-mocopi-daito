#!/usr/bin/env python3
"""Launch local/TSUBAME GPU physics and the causal teacher/student task."""

import argparse
import json
import os
from pathlib import Path
import sys
import traceback
from freeze_source import freeze_source

ROOT = Path(__file__).resolve().parents[1]

frozen_source, source_revision = freeze_source(ROOT, os.environ.get("K1_FROZEN_SOURCE"))
os.environ["K1_MOTION_ROOT"] = str(ROOT)
os.environ["K1_SOURCE_REVISION"] = source_revision
for directory in ("src", "third_party/booster_assets/src", "third_party/booster_train/source/booster_train"):
    sys.path.insert(0, str(ROOT / directory))
sys.path.insert(0, str(frozen_source))

parser = argparse.ArgumentParser()
parser.add_argument("--library", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--stage", choices=("teacher", "student"), default="teacher")
parser.add_argument("--teacher")
parser.add_argument("--iterations", type=int, default=50)
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--horizon", type=int, default=32)
parser.add_argument("--history", type=int, default=4)
parser.add_argument("--hidden-sizes", type=int, nargs=2, default=(256, 128))
parser.add_argument("--matmul-precision", choices=("highest", "high"), default="high")
parser.add_argument(
    "--sampling", choices=("episode_balanced", "transition_balanced"), default="episode_balanced"
)
parser.add_argument("--corruption", action="store_true")
parser.add_argument("--resume")
parser.add_argument("--initialize")
parser.add_argument("--minibatch", type=int, default=2048)
parser.add_argument("--epochs", type=int, default=4)
parser.add_argument("--learning-rate", type=float, default=3e-4)
parser.add_argument("--evaluation-interval", type=int, default=250)
parser.add_argument("--checkpoint-interval", type=int, default=100)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

from isaaclab.app import AppLauncher  # noqa: E402 -- Kit must start before physics imports.

app = AppLauncher(headless=True, enable_cameras=False).app
exit_code = 0
try:
    import torch

    torch.set_num_threads(4)
    from k1_motion.tracking_env import TrackerEnv
    from k1_motion.learning import TrainConfig, train

    env = TrackerEnv(
        args.library, args.num_envs, "cuda:0", "isaac", history=args.history, corruption=args.corruption
    )
    train(
        env,
        args.output,
        TrainConfig(
            stage=args.stage,
            iterations=args.iterations,
            horizon=args.horizon,
            minibatch=args.minibatch,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            evaluation_interval=args.evaluation_interval,
            checkpoint_interval=args.checkpoint_interval,
            seed=args.seed,
            hidden_sizes=tuple(args.hidden_sizes),
            matmul_precision=args.matmul_precision,
            sampling=args.sampling,
        ),
        args.teacher,
        args.resume,
        args.initialize,
    )
    env.close()
except BaseException:
    exit_code = 1
    traceback.print_exc()
finally:
    source_receipt = Path(args.output) / "source.json"
    source_receipt.parent.mkdir(parents=True, exist_ok=True)
    source_receipt.write_text(
        json.dumps({"revision": source_revision, "path": str(frozen_source)}, indent=2) + "\n"
    )
    from k1_motion.isaac_shutdown import shutdown

    shutdown(app, exit_code, Path(args.output) / "shutdown.json")
raise SystemExit(exit_code)
