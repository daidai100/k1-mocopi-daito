#!/usr/bin/env python3
"""Supervise five fixed-contract K1 learners: two queued, three concurrent."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time


RUNS = {
    "desktop": (
        ("desktop_body_seed44", "world-body-v1", 44, 0, "0-15", 16),
        ("desktop_velocity_seed44", "world-velocity-v1", 44, 0, "0-15", 16),
    ),
    "server": (
        ("server_body_seed42", "world-body-v1", 42, 0, "0-7,32-39", 12),
        ("server_velocity_seed42", "world-velocity-v1", 42, 0, "8-15,40-47", 12),
        ("server_body_seed43", "world-body-v1", 43, 1, "16-31,48-63", 24),
    ),
}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", choices=RUNS, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    args = parser.parse_args()
    bundle, inputs, output = args.bundle.resolve(), args.inputs.resolve(), args.output.resolve()
    source_directories = list((bundle / "artifacts/source-snapshots").glob("*/k1_motion"))
    if len(source_directories) != 1:
        raise ValueError("Expected exactly one frozen source in the supplied bundle")
    source = source_directories[0]
    if source.parent.name != "ce28463096de4c8b4764cf41b7f031a2bdd292a4551db1a4c2aaeb7ff09275fe":
        raise ValueError("Unexpected frozen source revision")
    required = (bundle / "scripts/train_warp.py", bundle / "initialize.pt",
                bundle / "configs/controller.json", inputs / "library/index.jsonl",
                inputs / "reference-cache.pt", inputs / "reference-cache.json",
                inputs / "manifests/minimal-casual-curriculum-v1.json")
    if any(not path.is_file() for path in required):
        raise FileNotFoundError([str(path) for path in required if not path.is_file()])
    if output.exists():
        raise FileExistsError(f"Preserve existing campaign output: {output}")
    output.mkdir(parents=True)
    plan = dict(version="five-run-full-corpus-v1", host=args.host, bundle=str(bundle),
                inputs=str(inputs), source_revision=source.parent.name, initializer=str(bundle / "initialize.pt"),
                curriculum=str(inputs / "manifests/minimal-casual-curriculum-v1.json"),
                corpus="18,054 original train references; no mirrors", target_updates_per_run=4000,
                expected_transitions_per_run=4000 * 2048 * 32,
                variation="world-velocity-v1 adds only root-velocity reward weight 2/s to world-body-v1",
                runs=[dict(name=name, reward_profile=profile, seed=seed, physical_gpu=gpu,
                           cpu_affinity=affinity, cpu_workers=workers)
                      for name, profile, seed, gpu, affinity, workers in RUNS[args.host]])
    write_json(output / "plan.json", plan)
    for run in plan["runs"]:
        write_json(output / run["name"] / "status.json", dict(phase="queued", **run))

    def execute(run):
        directory = output / run["name"]
        status_path = directory / "status.json"
        env = {k: v for k, v in os.environ.items() if k not in
               ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")}
        env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   OMP_WAIT_POLICY="PASSIVE", GOMP_SPINCOUNT="0", K1_FROZEN_SOURCE=str(source))
        env["HIP_VISIBLE_DEVICES" if args.host == "server" else "CUDA_VISIBLE_DEVICES"] = str(run["physical_gpu"])
        if args.host == "server":
            compat = "/home/vivi/amd-physx/runtime-compat-gfx1200/lib"
            env["LD_LIBRARY_PATH"] = compat + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
        for phase, updates in (("preflight", 25), ("training", 4000)):
            training = directory / phase
            command = ["taskset", "-c", run["cpu_affinity"], str(args.python),
                       str(bundle / "scripts/train_warp.py"),
                       "--backend", "mujoco_cpp", "--library", str(inputs / "library"),
                       "--output", str(training), "--stage", "student", "--device", "cuda:0",
                       "--num-envs", "2048", "--iterations", str(updates), "--horizon", "32",
                       "--history", "10", "--hidden-sizes", "512", "256", "--sampling",
                       "take_transition_balanced", "--reference-storage", "packed", "--reference-cache",
                       str(inputs / "reference-cache.pt"), "--minibatch", "4096", "--epochs", "4",
                       "--learning-rate", "1e-5", "--min-learning-rate", "1e-6", "--kl-stop", ".02",
                       "--bc-weight", "0", "--evaluation-interval", "0", "--checkpoint-interval", "25",
                       "--milestone-interval", "125", "--threads", "1", "--seed", str(run["seed"]),
                       "--reward-profile", run["reward_profile"], "--observation-profile", "preview",
                       "--preview-horizon-s", ".3", "--safety-profile", "casual-safe-v1",
                       "--action-settings", str(bundle / "configs/controller.json"),
                       "--curriculum-manifest", str(inputs / "manifests/minimal-casual-curriculum-v1.json"),
                       "--arm-workers", "8", "--initialize", str(bundle / "initialize.pt"),
                       "--cpu-workers", str(run["cpu_workers"]), "--cpu-chunk-size", "4"]
            write_json(directory / f"{phase}-command.json", command)
            with (directory / f"{phase}.log").open("w") as log:
                process = subprocess.Popen(command, cwd=bundle, env=env, stdout=log, stderr=subprocess.STDOUT)
                write_json(status_path, dict(**run, phase=phase, trainer_pid=process.pid,
                                             supervisor_pid=os.getpid(), started_at=datetime.now(timezone.utc).isoformat(),
                                             command_file=str(directory / f"{phase}-command.json")))
                return_code = process.wait()
            report = read_json(training / "report.json")
            if return_code or not report or not report.get("finite_updates") or report.get("checkpoint_reload_max_error") != 0:
                write_json(status_path, dict(**run, phase="failed", stage=phase,
                                             trainer_exit_code=return_code, report_present=bool(report)))
                return False
            iteration = report["last_metrics"]["iteration"]
            if iteration != updates:
                write_json(status_path, dict(**run, phase="failed", stage=phase,
                                             reason=f"expected {updates} updates, got {iteration}"))
                return False
            write_json(status_path, dict(**run, phase=f"{phase}_complete", trainer_exit_code=0,
                                         updates=iteration, transitions=report["last_metrics"]["transitions"],
                                         adam_steps=report["last_metrics"]["optimizer_steps"]))
        return True

    if args.host == "desktop":
        for run in plan["runs"]:
            execute(run)
    else:
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(execute, run) for run in plan["runs"]]
            for future in futures:
                future.result()
    statuses = {run["name"]: read_json(output / run["name"] / "status.json") for run in plan["runs"]}
    write_json(output / "final-status.json", statuses)
    if any(value["phase"] != "training_complete" for value in statuses.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
