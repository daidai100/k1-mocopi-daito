#!/usr/bin/env python3
"""Supervise three independent, resource-partitioned CPU-sim/GPU-learner runs."""
import argparse
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def read_metrics(path):
    if not path.exists():
        return []
    result = []
    for line in path.read_text().splitlines():
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            # The last line may be in flight; completed earlier records remain authoritative.
            break
    return result


def cpu_mask(index, workers):
    if workers not in (10, 15):
        raise ValueError("This topology-qualified launcher supports 10 or 15 workers per run")
    cores = list(range(index*10, index*10+10))
    logical = cores + ([core+32 for core in cores[:5]] if workers == 15 else [])
    return logical


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--library", type=Path, default=ROOT/"library")
    parser.add_argument("--reference-cache", type=Path)
    parser.add_argument("--workers", type=int, choices=(10, 15), required=True)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--large-width", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--worker-tuning", action="store_true", help="All three use baseline weights/reward/size")
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.output.exists() or args.iterations <= args.warmup or not 1e-5 <= args.learning_rate <= 1e-3:
        raise ValueError("New output and measured post-warmup updates required")
    # This mask is specific to the verified EPYC7452, not a portable core-number guess.
    cpuinfo = Path("/proc/cpuinfo").read_text()
    if "AMD EPYC 7452" not in cpuinfo:
        raise ValueError("Unexpected CPU topology; requalify placement before running")
    for core in range(30):
        siblings = (Path("/sys/devices/system/cpu")/f"cpu{core}/topology/thread_siblings_list").read_text().strip()
        if siblings != f"{core},{core+32}":
            raise ValueError("CPU SMT mapping changed")
    import torch
    torch.set_num_threads(1)
    if "R9700" not in torch.cuda.get_device_name(0):
        raise ValueError("The training device is not the qualified R9700")
    args.output.mkdir(parents=True)
    bundle = json.loads((ROOT/"bundle.json").read_text())
    state = {"phase": "launching", "started_unix": time.time(), "bundle": bundle,
             "workers_per_run": args.workers, "physical_cores_per_run": 10,
             "reserved_physical_cores": [30, 31], "num_envs": args.num_envs,
             "target_iterations": args.iterations, "target_transitions_each": args.iterations*32*args.num_envs,
             "initial_learning_rate": args.learning_rate,
             "worker_tuning_only": args.worker_tuning, "behaviorally_accepted": False,
             "gpu": torch.cuda.get_device_name(0), "runs": {}}
    processes, logs = {}, []

    def stop(*_):
        for process in processes.values():
            if process.poll() is None:
                process.terminate()
        state["phase"] = "interrupted"
        write_json(args.output/"status.json", state)
        raise SystemExit(130)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        for index, name in enumerate(("a_legacy_small", "b_beyondmimic_small", "c_beyondmimic_large")):
            profile = "legacy" if index == 0 or args.worker_tuning else "beyondmimic-causal-v1"
            width = args.large_width if index == 2 and not args.worker_tuning else 512
            mask = cpu_mask(index, args.workers)
            environment = {**os.environ, "HIP_VISIBLE_DEVICES": "0", "OMP_NUM_THREADS": "1",
                           "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "OMP_WAIT_POLICY": "PASSIVE",
                           "GOMP_SPINCOUNT": "0", "K1_SOURCE_REVISION": bundle["source_revision"]}
            command = ["taskset", "-c", ",".join(map(str, mask)), sys.executable,
                       str(ROOT/"scripts/train_cpu.py"), "--library", str(args.library.resolve()),
                       "--output", str(args.output/name/"training"), "--stage", "student",
                       "--device", "cuda:0", "--cpu-workers", str(args.workers), "--cpu-chunk-size", "4",
                       "--iterations", str(args.iterations), "--num-envs", str(args.num_envs),
                       "--horizon", "32", "--history", "10", "--hidden-sizes", str(width), str(width//2),
                       "--sampling", "take_transition_balanced", "--reference-storage", "packed",
                       "--minibatch", "4096", "--epochs", "4", "--learning-rate", str(args.learning_rate), "--bc-weight", "0",
                       "--initialize", str(ROOT/"initialize.pt"), "--evaluation-interval", "0",
                       "--checkpoint-interval", "25", "--threads", "1", "--seed", "42",
                       "--self-collision-weight", "1", "--reward-profile", profile,
                       "--residual-scale", ".25", "--command-velocity-limit", "6"]
            if profile == "legacy":
                command += ["--root-velocity-weight", "2", "--root-velocity-sigma", ".5"]
            if args.reference_cache:
                command += ["--reference-cache", str(args.reference_cache.resolve())]
            destination = args.output/name
            destination.mkdir()
            write_json(destination/"command.json", {"command": command, "cpu_affinity": mask})
            log = (destination/"training.log").open("w")
            logs.append(log)
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
            processes[name] = process
            state["runs"][name] = {"pid": process.pid, "profile": profile, "hidden_sizes": [width, width//2],
                                    "cpu_affinity": mask, "phase": "starting"}
        while True:
            running, failed = False, False
            for name, process in processes.items():
                code = process.poll()
                running |= code is None
                failed |= code not in (None, 0)
                metrics = read_metrics(args.output/name/"training/metrics.jsonl")
                run = state["runs"][name]
                run.update(exit_code=code, phase="learner_updates" if code is None and metrics else
                           "starting" if code is None else "completed" if code == 0 else "failed")
                if metrics:
                    run["latest"] = metrics[-1]
                    run["recent_transitions_per_second"] = statistics.median(r["transitions_per_second"] for r in metrics[-20:])
                report = args.output/name/"training/report.json"
                if code == 0 and not report.exists():
                    failed = True
                    run.update(phase="failed", error="Trainer exited without a completed report")
            state.update(phase="failed" if failed else "running" if running else "completed", updated_unix=time.time())
            write_json(args.output/"status.json", state)
            if failed:
                for process in processes.values():
                    if process.poll() is None:
                        process.terminate()
                raise RuntimeError("An ablation failed; peers stopped and existing checkpoints preserved")
            if not running:
                break
            time.sleep(10)
        results = {}
        for name in processes:
            report = json.loads((args.output/name/"training/report.json").read_text())
            metrics = read_metrics(args.output/name/"training/metrics.jsonl")[args.warmup:]
            if not report["finite_updates"] or report["checkpoint_reload_max_error"] != 0:
                raise RuntimeError("Finite-update/reload gate failed")
            results[name] = {"aggregate_transitions_per_second": len(metrics)*args.num_envs*32/sum(r["iteration_seconds"] for r in metrics),
                             "median_rollout_seconds": statistics.median(r["rollout_seconds"] for r in metrics),
                             "median_update_seconds": statistics.median(r["update_seconds"] for r in metrics),
                             "peak_torch_vram_bytes": report["peak_vram_bytes"],
                             "finite_updates": report["finite_updates"],
                             "checkpoint_reload_max_error": report["checkpoint_reload_max_error"],
                             "physics_contract": report["physics_contract"],
                             "reference_storage_bytes": report["reference_storage_bytes"]}
        write_json(args.output/"comparison.json", {"runs": results, "workers": args.workers,
                   "sum_steady_run_rates": sum(r["aggregate_transitions_per_second"] for r in results.values()),
                   "wall_seconds_including_setup": time.time()-state["started_unix"],
                   "scope": "Matched exposure, separate samplers/optimizers. Rates exclude loading and saves; "
                            "sum is approximate when finish times differ. Reward scales are not behavioral scores.",
                   "behaviorally_accepted": False})
    finally:
        for process in processes.values():
            if process.poll() is None:
                process.terminate()
        for log in logs:
            log.close()


if __name__ == "__main__":
    main()
