#!/usr/bin/env python3
"""Continue a verified student benchmark inside an existing TSUBAME allocation."""

import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time

from freeze_source import freeze_source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--persistent-root", required=True)
    parser.add_argument("--resume", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--target-iteration", type=int, default=1000)
    parser.add_argument("--run-name", default="student-v1")
    parser.add_argument("--source-directory")
    parser.add_argument("--recover-saved-checkpoint", action="store_true")
    parser.add_argument("--allow-source-change", action="store_true")
    parser.add_argument("--fresh-optimizer", action="store_true")
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--minibatch", type=int)
    parser.add_argument("--nconmax", type=int)
    parser.add_argument("--njmax", type=int)
    parser.add_argument("--epa-horizon", type=int)
    parser.add_argument("--checkpoint-interval", type=int, default=25)
    parser.add_argument("--evaluation-interval", type=int, default=100)
    args = parser.parse_args()
    if args.evaluation_interval < 0 or args.checkpoint_interval < 1:
        raise ValueError("Evaluation interval must be nonnegative and checkpoint interval positive")
    job = os.environ["JOB_ID"]
    scratch = Path(os.environ["T4TMPDIR"])
    root = Path(__file__).resolve().parents[1]
    if not root.is_relative_to(scratch) or not scratch.is_dir():
        raise RuntimeError("Run this from node-local scratch inside the allocated job")
    persistent = Path(args.persistent_root).resolve()
    checkpoint = Path(args.resume).resolve()
    prior = json.loads((checkpoint.parent / "config.json").read_text())
    if prior["backend"] != "warp" or prior["config"]["stage"] != "student":
        raise ValueError("Resume candidate must pass the Warp student learner/export preflight")
    if args.recover_saved_checkpoint:
        import math
        import torch

        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        start = saved["iteration"]
        metrics = [json.loads(s) for s in (checkpoint.parent / "metrics.jsonl").read_text().splitlines()]
        retained = [m for m in metrics if m["iteration"] == start]
        if (
            not retained
            or not all(math.isfinite(retained[-1][key]) for key in ("loss", "gradient_norm", "bc_loss"))
            or not all(torch.isfinite(value).all() for value in saved["model"].values())
            or saved["source_revision"] != prior["source_revision"]
            or saved["stage"] != "student"
        ):
            raise ValueError("Saved checkpoint or its pre-evaluation learner evidence is invalid")
    else:
        report = json.loads((checkpoint.parent / "report.json").read_text())
        if not report["finite_updates"] or report["checkpoint_reload_max_error"] > 1e-6:
            raise ValueError("Candidate failed learner/export checks")
        start = report["last_metrics"]["iteration"]
    parent_iteration = start
    if args.fresh_optimizer:
        start = 0
    num_envs = args.num_envs or prior["num_envs"]
    if num_envs != prior["num_envs"] and not args.fresh_optimizer:
        raise ValueError("A different rollout size requires explicit fresh optimizer initialization")
    if args.target_iteration <= start:
        raise ValueError("Target iteration must exceed the retained checkpoint")
    source, revision = freeze_source(root, args.source_directory)
    if revision != prior["source_revision"] and not args.allow_source_change:
        raise ValueError("Review source changes before continuing the measured benchmark")
    shutil.copytree(source, persistent / "artifacts/source-snapshots" / revision, dirs_exist_ok=True)
    output = persistent / "artifacts" / f"tsubame-{job}" / args.run_name
    if output.exists():
        raise FileExistsError(output)
    log_path = persistent / "logs" / f"tsubame-{job}-{args.run_name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    operation = persistent / "operations" / f"allocation-{job}"
    operation.mkdir(parents=True, exist_ok=True)
    command = [
        str(root / "venv/bin/python"),
        "scripts/train_warp.py",
        "--library",
        prior.get("library", "artifacts/references-v7-contact-root"),
        "--output",
        str(output),
        "--stage",
        "student",
        "--teacher",
        args.teacher,
        "--initialize" if args.fresh_optimizer else "--resume",
        str(checkpoint),
        "--iterations",
        str(args.target_iteration - start),
        "--num-envs",
        str(num_envs),
        "--history",
        str(prior["observation"]["history"]),
        "--hidden-sizes",
        *map(str, prior["config"]["hidden_sizes"]),
        "--horizon",
        str(prior["config"]["horizon"]),
        "--minibatch",
        str(args.minibatch or prior["config"]["minibatch"]),
        "--epochs",
        str(prior["config"]["epochs"]),
        "--learning-rate",
        str(prior["config"]["learning_rate"]),
        "--bc-weight",
        str(prior["config"]["bc_weight"]),
        "--sampling",
        prior["config"]["sampling"],
        "--checkpoint-interval",
        str(args.checkpoint_interval),
        "--evaluation-interval",
        str(args.evaluation_interval),
        "--threads",
        "2",
        "--corruption",
        "--no-conditional-graphs",
        "--nconmax",
        str(args.nconmax or prior["physics_contract"]["nconmax"]),
        "--njmax",
        str(args.njmax or prior["physics_contract"]["njmax"]),
    ]
    horizon_capacity = args.epa_horizon or prior["physics_contract"].get("epa_horizon")
    if horizon_capacity is not None:
        command.extend(["--epa-horizon", str(horizon_capacity)])
    for name, value in prior.get("action_settings", {}).items():
        command.extend(["--" + name.replace("_", "-"), str(value)])
    collision_weight = prior.get("reward_settings", {}).get("self_collision_weight", 0.0)
    if collision_weight:
        command.extend(["--self-collision-weight", str(collision_weight)])
    environment = {
        **os.environ,
        "K1_FROZEN_SOURCE": str(source / "k1_motion"),
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "2",
        "PYTHONUNBUFFERED": "1",
        "XDG_CACHE_HOME": str(scratch / "cache"),
    }
    receipt = {
        "job_id": job,
        "node": socket.gethostname(),
        "source_revision": revision,
        "prior_source_revision": prior["source_revision"],
        "source_change_declared": revision != prior["source_revision"],
        "recovery_from_saved_checkpoint": args.recover_saved_checkpoint,
        "command": command,
        "output": str(output),
        "log": str(log_path),
        "start_iteration": start,
        "initialization_parent_iteration": parent_iteration if args.fresh_optimizer else None,
        "optimizer_state": "fresh; changed rollout size" if args.fresh_optimizer else "restored",
        "target_iteration": args.target_iteration,
        "started_unix": time.time(),
        "checkpoint_storage": "persistent home; atomic replacement",
        "evaluation": "native replay in learner process"
        if args.evaluation_interval
        else "separate checkpoint replay required; no in-process native replay",
    }
    receipt_path = operation / (args.run_name + "-launch.json")
    with log_path.open("x") as log:
        process = subprocess.Popen(command, cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT)
        receipt["pid"] = process.pid
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(receipt), flush=True)
        code = process.wait()
    receipt.update({"exit_code": code, "finished_unix": time.time()})
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
