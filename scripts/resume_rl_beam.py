#!/usr/bin/env python3
"""Benchmark or continue four saved learners with explicit CPU/GPU placement."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time

from run_rl_beam import NAMES
from run_server_ablation import read_metrics, write_json

ROOT = Path(__file__).resolve().parents[1]


def cpu_layout(workers):
    if len(workers) != 4 or any(w < 2 or w % 2 for w in workers) or sum(workers) > 64:
        raise ValueError("Four even worker counts must fit the server's 32 physical cores")
    masks, first = [], 0
    for count in workers:
        cores = list(range(first, first + count//2))
        masks.append(cores + [c+32 for c in cores])
        first += count//2
    return masks, list(range(first, 32))


def continuation_command(original, bundle, executable, checkpoint, output, cache, mask,
                         iterations, remaining_seconds=None, unfused=False, ppo_lock=None):
    if original[:2] != ["taskset", "-c"] or "train_cpu.py" not in original[4]:
        raise ValueError("Unexpected original learner command")
    arguments = original[5:].copy()

    def replace(flag, value=None):
        if flag in arguments:
            index = arguments.index(flag)
            del arguments[index:index+2]
        if value is not None:
            arguments.extend([flag, str(value)])

    replace("--initialize")
    replace("--resume", checkpoint)
    replace("--library", bundle / "library")
    replace("--reference-cache", cache)
    replace("--output", output)
    replace("--iterations", iterations)
    replace("--cpu-workers", len(mask))
    replace("--max-seconds", remaining_seconds)
    replace("--ppo-update-lock", ppo_lock)
    if remaining_seconds is None:
        replace("--checkpoint-interval", 1000000)
    if unfused:
        arguments.append("--cpu-unfused-arm-feedback")
    return ["taskset", "-c", ",".join(map(str, mask)), executable, str(bundle / "scripts/train_cpu.py"),
            *arguments]


def concurrent_results(runs, warmup):
    """Exclude warmups and tails after any of the four learners finishes."""
    series = {}
    for name, run in runs.items():
        path = Path(run["training_directory"]) / "metrics.jsonl"
        rows = read_metrics(path)
        # elapsed_seconds is written with each line; mtime anchors the final line.
        origin = path.stat().st_mtime - rows[-1]["elapsed_seconds"]
        series[name] = [(origin+r["elapsed_seconds"], r) for r in rows]
    start = max(rows[warmup-1][0] for rows in series.values())
    end = min(rows[-1][0] for rows in series.values())
    measured = {}
    for name, rows in series.items():
        common = [r for timestamp, r in rows if start <= timestamp <= end]
        if len(common) < 3:
            raise ValueError("Insufficient common four-learner measurement window")
        measured[name] = {
            "first_iteration": common[0]["iteration"], "last_iteration": common[-1]["iteration"],
            "updates": len(common)-1,
            "transitions_per_second": (common[-1]["transitions"]-common[0]["transitions"])
                / (common[-1]["elapsed_seconds"]-common[0]["elapsed_seconds"]),
            "mean_rollout_seconds": statistics.mean(r["rollout_seconds"] for r in common[1:]),
            "mean_update_seconds": statistics.mean(r["update_seconds"] for r in common[1:]),
        }
    return {"start_unix": start, "end_unix": end, "runs": measured,
            "aggregate_transitions_per_second": sum(r["transitions_per_second"] for r in measured.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-campaign", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-cache", type=Path, required=True)
    parser.add_argument("--workers", type=int, nargs=4, default=[14, 16, 16, 16])
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--unfused", action="store_true")
    parser.add_argument("--serialize-ppo", action="store_true")
    parser.add_argument("--qualification", type=Path)
    parser.add_argument("--parity", type=Path)
    parser.add_argument("--live-status", type=Path)
    args = parser.parse_args()
    args.bundle = args.bundle.resolve()
    args.output = args.output.resolve()
    if args.output.exists() or args.iterations <= args.warmup:
        raise ValueError("A new destination and a measured post-warmup window are required")
    if args.benchmark and args.live_status:
        raise ValueError("Benchmarks cannot publish to production status")
    if "AMD EPYC 7452" not in Path("/proc/cpuinfo").read_text():
        raise ValueError("Requalify CPU topology")
    for core in range(32):
        if Path(f"/sys/devices/system/cpu/cpu{core}/topology/thread_siblings_list").read_text().strip() != f"{core},{core+32}":
            raise ValueError("CPU sibling mapping changed")
    masks, reserved = cpu_layout(args.workers)
    if any(os.environ.get(k) for k in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")):
        raise ValueError("Supervisor must see both unmasked GPUs")
    import torch
    torch.set_num_threads(1)
    devices = [torch.cuda.get_device_name(i) for i in (0, 1)]
    if "R9700" not in devices[0] or "9060 XT" not in devices[1]:
        raise ValueError("GPU placement changed")
    previous = json.loads((args.resume_campaign / "status.json").read_text())
    if previous["phase"] not in ("interrupted", "completed", "failed", "completed_with_failures"):
        raise ValueError("Stop the original supervisor before loading its checkpoints")
    for run in previous["runs"].values():
        if run.get("pid") and Path(f"/proc/{run['pid']}").exists():
            raise ValueError("An original learner is still present")
    bundle = json.loads((args.bundle / "bundle.json").read_text())
    if not args.benchmark:
        if args.qualification is None or args.parity is None:
            raise ValueError("Production requires a matched full-pool benchmark and exact parity")
        qualification = json.loads(args.qualification.read_text())
        parity = json.loads(args.parity.read_text())
        if (qualification["phase"] != "completed" or not qualification["benchmark"]
                or qualification["bundle"] != bundle or qualification["workers_per_run"] != args.workers
                or qualification["unfused"] != args.unfused or not parity["all_exact"]):
            raise ValueError("Qualification does not match this continuation")
        if qualification.get("serialize_ppo", False) != args.serialize_ppo:
            raise ValueError("PPO scheduling differs from qualification")
    args.output.mkdir(parents=True)
    state = {**copy.deepcopy(previous), "phase": "starting", "benchmark": args.benchmark,
             "bundle": bundle, "continued_from": str(args.resume_campaign.resolve()),
             "execution_started_unix": time.time(), "workers_per_run": args.workers,
             "reserved_physical_cores": reserved, "unfused": args.unfused, "warmup": args.warmup,
             "serialize_ppo": args.serialize_ppo,
             "behaviorally_accepted": False, "runs": {}}
    processes, logs = {}, []

    def publish():
        state["updated_unix"] = time.time()
        write_json(args.output / "status.json", state)
        if args.live_status:
            write_json(args.live_status, state)

    def stop(*_):
        state["phase"] = "interrupted"
        publish()
        raise SystemExit(130)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        # Freeze every input before any learner or benchmark can advance.
        for name in NAMES:
            old = previous["runs"][name]
            source = Path(old["training_directory"])
            destination = args.output / name / "resume-input"
            destination.mkdir(parents=True)
            shutil.copy2(source / "checkpoint.pt", destination / "checkpoint.pt")
            shutil.copy2(source / "config.json", destination / "config.json")
            checkpoint = torch.load(destination / "checkpoint.pt", map_location="cpu", weights_only=True)
            if checkpoint["num_envs"] != 2048 or checkpoint["config"]["horizon"] != 32:
                raise ValueError("Keep the established rollout and exposure schedule")
            adam_steps = int(max(float(v["step"]) for v in checkpoint["optimizer"]["state"].values()))
            if adam_steps != checkpoint["optimizer_steps"]:
                raise ValueError("Checkpoint Adam counter is inconsistent")
            state["runs"][name] = {
                "phase": "staged", "resumed_iteration": checkpoint["iteration"],
                "resumed_transitions": checkpoint["transitions"], "resumed_optimizer_steps": adam_steps,
                "resume_source": str(source / "checkpoint.pt"),
                "prior_last_logged": old.get("latest"),
                "training_directories": [] if args.benchmark else old["training_directories"].copy(),
                "training_deadline_unix": old.get("training_deadline_unix",
                    previous["started_unix"] + previous["hours_per_run"]*3600),
            }
        for index, name in enumerate(NAMES):
            run = state["runs"][name]
            gpu = 1 if index == 0 else 0
            remaining = None if args.benchmark else run["training_deadline_unix"] - time.time()
            if remaining is not None and remaining <= 0:
                raise ValueError("Original wall-time budget has expired")
            directory = args.output / name / "training"
            command = continuation_command(previous["runs"][name]["command"], args.bundle, sys.executable,
                directory.parent / "resume-input/checkpoint.pt", directory, args.reference_cache.resolve(),
                masks[index], args.iterations if args.benchmark else 500000, remaining, args.unfused,
                args.output / "gpu-0-ppo.lock" if args.serialize_ppo and gpu == 0 else None)
            environment = {**os.environ, "HIP_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "1",
                           "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OMP_WAIT_POLICY": "PASSIVE",
                           "GOMP_SPINCOUNT": "0"}
            run.update(gpu=devices[gpu], hip_visible_devices=str(gpu), cpu_affinity=masks[index],
                       training_directory=str(directory), command=command)
            run["training_directories"].append(str(directory))
            write_json(directory.parent / "command.json", run)
            log = (directory.parent / "training.log").open("w")
            logs.append(log)
            process = subprocess.Popen(command, cwd=args.bundle, env=environment, stdout=log, stderr=subprocess.STDOUT)
            processes[name] = process
            run.update(pid=process.pid, phase="starting")
            publish()
        while True:
            running = False
            for name, process in processes.items():
                run = state["runs"][name]
                if run["phase"] in ("completed", "failed"):
                    continue
                directory = Path(run["training_directory"])
                metrics = read_metrics(directory / "metrics.jsonl")
                code = process.poll()
                run["exit_code"] = code
                if metrics:
                    run["latest"] = metrics[-1]
                    run["recent_transitions_per_second"] = statistics.median(
                        r["transitions_per_second"] for r in metrics[-20:])
                if code is None:
                    running = True
                    run["phase"] = "learner_updates" if metrics else "starting"
                elif code or not (directory / "report.json").exists():
                    run.update(phase="failed", error="Learner did not produce a successful terminal report")
                else:
                    report = json.loads((directory / "report.json").read_text())
                    valid = (report["finite_updates"] and report["checkpoint_reload_max_error"] == 0
                             and report["start_iteration"] == run["resumed_iteration"]
                             and report["start_transitions"] == run["resumed_transitions"]
                             and report["start_optimizer_steps"] == run["resumed_optimizer_steps"])
                    run.update(phase="completed" if valid else "failed", stop_reason=report["stop_reason"],
                               finite_updates=report["finite_updates"],
                               checkpoint_reload_max_error=report["checkpoint_reload_max_error"])
                    if args.benchmark and len(metrics) > args.warmup:
                        rows = metrics[args.warmup:]
                        run["benchmark_result"] = {
                            "measured_updates": len(rows),
                            "transitions_per_second": 65536*len(rows)/sum(r["iteration_seconds"] for r in rows),
                            "mean_rollout_seconds": statistics.mean(r["rollout_seconds"] for r in rows),
                            "mean_update_seconds": statistics.mean(r["update_seconds"] for r in rows),
                            "median_iteration_seconds": statistics.median(r["iteration_seconds"] for r in rows),
                        }
            failed = any(r["phase"] == "failed" for r in state["runs"].values())
            state["phase"] = "running" if running else "completed_with_failures" if failed else "completed"
            if not running and not failed and args.benchmark:
                state["aggregate_transitions_per_second"] = sum(
                    r["benchmark_result"]["transitions_per_second"] for r in state["runs"].values())
                state["concurrent_result"] = concurrent_results(state["runs"], args.warmup)
            publish()
            if not running:
                break
            time.sleep(2)
    except Exception as error:
        state.update(phase="failed", error=f"{type(error).__name__}: {error}")
        publish()
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
