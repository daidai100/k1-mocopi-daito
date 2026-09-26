#!/usr/bin/env python3
"""Supervise a qualified full-pool Warp continuation with durable live status."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from training_status import status

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-iteration", type=int, default=8000)
    args = parser.parse_args()
    report = json.loads((args.preflight / "report.json").read_text())
    source = json.loads((args.preflight / "source.json").read_text())
    if not report["finite_updates"] or report["checkpoint_reload_max_error"] != 0:
        raise ValueError("Preflight failed")
    start = report["last_metrics"]["iteration"]
    if args.target_iteration <= start:
        raise ValueError("Target must follow preflight")
    args.output.mkdir(parents=True, exist_ok=False)
    config = report["config"]
    settings = args.output / "action-settings.json"
    settings.write_text(json.dumps(report["action_settings"], indent=2)+"\n")
    environment = {**os.environ, "K1_FROZEN_SOURCE": str(Path(source["path"]) / "k1_motion"),
                   "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "2"}
    training = args.output / "training"
    command = [sys.executable, str(ROOT / "scripts/train_warp.py"),
               "--library", report["library"], "--output", str(training), "--stage", "student",
               "--resume", str(args.preflight / "checkpoint.pt"),
               "--iterations", str(args.target_iteration-start), "--num-envs", str(report["num_envs"]),
               "--horizon", str(config["horizon"]), "--history", str(report["observation"]["history"]),
               "--hidden-sizes", *map(str, config["hidden_sizes"]),
               "--minibatch", str(config["minibatch"]), "--epochs", str(config["epochs"]),
               "--learning-rate", str(config["learning_rate"]), "--bc-weight", "0",
               "--sampling", config["sampling"], "--reference-storage", "packed",
               "--action-settings", str(settings), "--self-collision-weight", "1",
               "--root-velocity-weight", "2", "--root-velocity-sigma", ".5",
               "--evaluation-interval", "0", "--checkpoint-interval", "25",
               "--nconmax", "128", "--njmax", "1024", "--epa-horizon", "96",
               "--no-conditional-graphs", "--threads", "2", "--seed", str(config["seed"])]
    (args.output / "command.json").write_text(json.dumps(command, indent=2)+"\n")
    (args.output / "source.json").write_text(json.dumps(source, indent=2)+"\n")
    with (args.output / "training.log").open("w") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
        while True:
            code = process.poll()
            current = status(training)
            current.update(supervisor_pid=os.getpid(), trainer_pid=process.pid, trainer_exit_code=code,
                           preflight=str(args.preflight.resolve()), checkpoint_interval=25,
                           hardware_verified=False, updated_unix=time.time())
            if code is not None and code != 0:
                current.update(state="failed", phase="failed")
            temporary = args.output / "status.partial"
            temporary.write_text(json.dumps(current, indent=2, allow_nan=False)+"\n")
            temporary.replace(args.output / "status.json")
            if code is not None:
                raise SystemExit(code)
            time.sleep(20)


if __name__ == "__main__":
    main()
