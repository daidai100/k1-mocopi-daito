#!/usr/bin/env python3
"""Checkpoint-preserving two-GPU continuation with a matched environment schedule."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time

from run_server_ablation import cpu_mask, read_metrics, write_json

ROOT = Path(__file__).resolve().parents[1]
NAMES = ("a_legacy_small", "b_beyondmimic_small", "c_beyondmimic_large")


def stage_plan(checkpoint, resize_at, resized_envs, target_transitions):
    """Switch all treatments after the same number of old-size transitions."""
    iteration, transitions = checkpoint["iteration"], checkpoint["transitions"]
    envs = checkpoint["num_envs"]
    if checkpoint["config"]["horizon"] != 32 or transitions >= target_transitions:
        raise ValueError("Unexpected rollout horizon or no remaining training budget")
    boundary = resize_at * 1024 * 32
    if transitions < boundary:
        if envs != 1024 or transitions != iteration*1024*32:
            raise ValueError("Pre-resize exposure does not match the original schedule")
        return {"directory": "training", "num_envs": 1024, "iterations": resize_at-iteration,
                "allow_env_resize": False, "milestone_interval": 500}
    if transitions == boundary:
        if iteration != resize_at or envs != 1024:
            raise ValueError("Environment switch must use the matched original-size checkpoint")
    elif envs != resized_envs or not any(
            row["start_transitions"] == boundary and row["num_envs"] == resized_envs
            for row in checkpoint.get("rollout_history", [])):
        raise ValueError("Checkpoint missed the matched environment switch")
    batch = resized_envs*32
    remaining = target_transitions-transitions
    if remaining % batch:
        raise ValueError("Final exposure must be exactly divisible by the resized rollout")
    return {"directory": "training-resized", "num_envs": resized_envs, "iterations": remaining//batch,
            "allow_env_resize": envs != resized_envs, "milestone_interval": 16_384_000//batch}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-cache", type=Path, required=True)
    parser.add_argument("--gpu-indices", type=int, nargs=3, default=[0, 1, 0], help="Unmasked PyTorch device indices, A/B/C")
    parser.add_argument("--workers", type=int, choices=(10, 15), default=15)
    parser.add_argument("--resize-at-iteration", type=int, default=500)
    parser.add_argument("--resize-num-envs", type=int, choices=(2048, 4096), default=2048)
    parser.add_argument("--target-transitions", type=int, default=2_097_152_000,
                        help="Safety ceiling, not the expected final exposure of time-budgeted runs")
    parser.add_argument("--hours", type=float, default=8., help="Wall-time budget per resumed run")
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.output.exists() or args.resize_at_iteration <= 0 or not 0 < args.hours <= 24:
        raise ValueError("New output and positive schedule boundary required")
    if "AMD EPYC 7452" not in Path("/proc/cpuinfo").read_text():
        raise ValueError("Unexpected server CPU")
    for core in range(30):
        path = Path(f"/sys/devices/system/cpu/cpu{core}/topology/thread_siblings_list")
        if path.read_text().strip() != f"{core},{core+32}":
            raise ValueError("Requalify the changed CPU topology")
    if any(os.environ.get(key) for key in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")):
        raise ValueError("Supervisor needs unmasked GPU indices; it isolates each child itself")
    import torch
    torch.set_num_threads(1)
    devices = {index: torch.cuda.get_device_name(index) for index in set(args.gpu_indices)}
    expected = ("R9700", "9060 XT", "R9700")
    if any(name not in devices[index] for name, index in zip(expected, args.gpu_indices)):
        raise ValueError("GPU assignment is not the qualified R9700 / 9060 XT / R9700 placement")
    previous = json.loads((args.resume_campaign/"status.json").read_text())
    if previous["phase"] not in ("interrupted", "completed", "failed"):
        raise ValueError("Stop the old supervisor before continuation; its checkpoint files must be stable")
    args.output.mkdir(parents=True)
    bundle = json.loads((ROOT/"bundle.json").read_text())
    state = {"phase": "launching", "started_unix": time.time(), "bundle": bundle,
             "continued_from": str(args.resume_campaign.resolve()), "workers_per_run": args.workers,
             "reserved_physical_cores": [30, 31], "target_transitions_each": args.target_transitions,
             "hours_per_run": args.hours, "budget_semantics": "Eight-hour per-run budget from continuation; compare shared exposure milestones",
             "environment_schedule": [{"start_transitions": 0, "num_envs": 1024},
                 {"start_transitions": args.resize_at_iteration*1024*32, "num_envs": args.resize_num_envs}],
             "behaviorally_accepted": False, "runs": {}}
    processes, logs = {}, []

    def stop(*_):
        for process in processes.values():
            if process.poll() is None:
                process.terminate()
        state.update(phase="interrupted", updated_unix=time.time())
        write_json(args.output/"status.json", state)
        raise SystemExit(130)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def launch(index, checkpoint_path):
        name = NAMES[index]
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        plan = stage_plan(checkpoint, args.resize_at_iteration, args.resize_num_envs, args.target_transitions)
        width = 4096 if index == 2 else 512
        profile = "legacy" if index == 0 else "beyondmimic-causal-v1"
        if (checkpoint["hidden_sizes"] != [width, width//2]
                or checkpoint["reward_settings"].get("profile", "legacy") != profile):
            raise ValueError("Checkpoint treatment differs")
        gpu = args.gpu_indices[index]
        mask = cpu_mask(index, args.workers)
        environment = {**os.environ, "HIP_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "1",
                       "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OMP_WAIT_POLICY": "PASSIVE",
                       "GOMP_SPINCOUNT": "0"}
        directory = args.output/name/plan["directory"]
        command = ["taskset", "-c", ",".join(map(str, mask)), sys.executable, str(ROOT/"scripts/train_cpu.py"),
                   "--library", str(ROOT/"library"), "--output", str(directory), "--stage", "student",
                   "--device", "cuda:0", "--cpu-workers", str(args.workers), "--cpu-chunk-size", "4",
                   "--iterations", str(plan["iterations"]), "--num-envs", str(plan["num_envs"]),
                   "--horizon", "32", "--history", "10", "--hidden-sizes", str(width), str(width//2),
                   "--sampling", "take_transition_balanced", "--reference-storage", "packed",
                   "--reference-cache", str(args.reference_cache.resolve()), "--minibatch", "4096",
                   "--epochs", "4", "--learning-rate", "1e-5", "--bc-weight", "0",
                   "--resume", str(checkpoint_path), "--evaluation-interval", "0", "--checkpoint-interval", "25",
                   "--milestone-interval", str(plan["milestone_interval"]), "--threads", "1", "--seed", "42",
                   "--self-collision-weight", "1", "--reward-profile", profile,
                   "--residual-scale", ".25", "--command-velocity-limit", "6"]
        if profile == "legacy":
            command += ["--root-velocity-weight", "2", "--root-velocity-sigma", ".5"]
        if plan["allow_env_resize"]:
            command.append("--allow-env-resize")
        run = state["runs"][name]
        run.setdefault("training_deadline_unix", time.time()+args.hours*3600)
        command += ["--max-seconds", str(max(0.001, run["training_deadline_unix"]-time.time()))]
        write_json(directory.parent/(plan["directory"]+"-command.json"),
                   {"command": command, "hip_visible_devices": str(gpu), "gpu": devices[gpu], "cpu_affinity": mask})
        log = (directory.parent/(plan["directory"]+".log")).open("w")
        logs.append(log)
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
        processes[name] = process
        run.update(pid=process.pid, phase="starting", profile=profile, hidden_sizes=[width, width//2],
                   gpu=devices[gpu], hip_visible_devices=str(gpu), cpu_affinity=mask,
                   training_directory=str(directory), stage=plan, start_iteration=checkpoint["iteration"],
                   start_transitions=checkpoint["transitions"], exit_code=None)
        run["training_directories"].append(str(directory))

    try:
        # Preserve all durable inputs before launching any child.
        for name in NAMES:
            source = Path(previous["runs"][name].get("training_directory", args.resume_campaign/name/"training"))
            destination = args.output/name/"resume-input"
            destination.mkdir(parents=True)
            selected = source/"checkpoint.pt"
            current = torch.load(selected, map_location="cpu", weights_only=True)
            if current["num_envs"] == 1024 and current["iteration"] > args.resize_at_iteration:
                selected = source/f"checkpoint-{args.resize_at_iteration:06d}.pt"
            shutil.copy2(selected, destination/"checkpoint.pt")
            shutil.copy2(source/"config.json", destination/"config.json")
            checkpoint = torch.load(destination/"checkpoint.pt", map_location="cpu", weights_only=True)
            stage_plan(checkpoint, args.resize_at_iteration, args.resize_num_envs, args.target_transitions)
            state["runs"][name] = {"training_directories": [], "completed_stages": [],
                                  "resumed_iteration": checkpoint["iteration"],
                                  "resumed_transitions": checkpoint["transitions"],
                                  "resume_source": str(selected),
                                  "prior_last_logged": previous["runs"][name].get("latest")}
        for index, name in enumerate(NAMES):
            launch(index, args.output/name/"resume-input/checkpoint.pt")
        while True:
            failed, running = False, False
            for index, name in enumerate(NAMES):
                run = state["runs"][name]
                if run["phase"] == "completed":
                    continue
                process = processes[name]
                code = process.poll()
                training = Path(run["training_directory"])
                metrics = read_metrics(training/"metrics.jsonl")
                run["exit_code"] = code
                if metrics:
                    run["latest"] = metrics[-1]
                    run["recent_transitions_per_second"] = statistics.median(r["transitions_per_second"] for r in metrics[-20:])
                if code is None:
                    running = True
                    run["phase"] = "learner_updates" if metrics else "starting"
                elif code != 0 or not (training/"report.json").exists():
                    failed = True
                    run.update(phase="failed", error="Trainer failed or exited without its completed report")
                else:
                    report = json.loads((training/"report.json").read_text())
                    if not report["finite_updates"] or report["checkpoint_reload_max_error"] != 0:
                        raise RuntimeError("Stage finite-update/reload gate failed")
                    run["completed_stages"].append({"directory": str(training), "transitions": report["transitions"],
                                                     "iteration": report["last_metrics"]["iteration"]})
                    if report["transitions"] == args.target_transitions or report.get("stop_reason") == "walltime_budget":
                        run["phase"] = "completed"
                        run["stop_reason"] = report.get("stop_reason", "iteration_budget")
                    else:
                        launch(index, training/"checkpoint.pt")
                        running = True
            state.update(phase="failed" if failed else "running" if running else "completed", updated_unix=time.time())
            write_json(args.output/"status.json", state)
            if failed:
                raise RuntimeError("An ablation failed; stopping peers and preserving checkpoints")
            if not running:
                break
            time.sleep(5)
    except Exception as error:
        state.update(phase="failed", error=f"{type(error).__name__}: {error}", updated_unix=time.time())
        write_json(args.output/"status.json", state)
        raise
    finally:
        for process in processes.values():
            if process.poll() is None:
                process.terminate()
        for process in processes.values():
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for log in logs:
            log.close()


if __name__ == "__main__":
    main()
