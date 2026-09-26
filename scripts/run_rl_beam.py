#!/usr/bin/env python3
"""Four matched fresh-optimizer RL treatments on the qualified two-GPU server."""
import argparse
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

from run_server_ablation import read_metrics, write_json

ROOT = Path(__file__).resolve().parents[1]
NAMES = ("r0_control", "r1_pv_controller", "r2_curriculum", "r3_travel_reward")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-cache", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--preflight-report", type=Path)
    parser.add_argument("--hours", type=float, default=8)
    args = parser.parse_args()
    if args.output.exists() or not 0 < args.hours <= 8:
        raise ValueError("A new output and a positive at-most-eight-hour budget are required")
    if "AMD EPYC 7452" not in Path("/proc/cpuinfo").read_text():
        raise ValueError("Requalify the CPU topology")
    for core in range(32):
        if Path(f"/sys/devices/system/cpu/cpu{core}/topology/thread_siblings_list").read_text().strip() != f"{core},{core+32}":
            raise ValueError("CPU topology changed")
    if any(os.environ.get(k) for k in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")):
        raise ValueError("Supervisor must see the unmasked GPU mapping")
    import torch
    import mujoco
    torch.set_num_threads(1)
    devices = [torch.cuda.get_device_name(i) for i in (0, 1)]
    if "R9700" not in devices[0] or "9060 XT" not in devices[1] or mujoco.__version__ != "3.10.0":
        raise ValueError("GPU or physics dependency mapping changed")
    bundle = json.loads((ROOT / "bundle.json").read_text())
    if not args.preflight:
        if args.preflight_report is None:
            raise ValueError("Production requires a full-pool four-way preflight")
        preflight = json.loads(args.preflight_report.read_text())
        if (preflight["phase"] != "completed" or not preflight["preflight"]
                or preflight["bundle"] != bundle or set(preflight["runs"]) != set(NAMES)):
            raise ValueError("Preflight was unsuccessful or used different code/data")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True)
    state = {"phase": "starting", "started_unix": time.time(), "bundle": bundle,
             "hours_per_run": args.hours, "preflight": args.preflight, "seed": 42,
             "num_envs": 2048, "workers_per_run": 14, "reserved_physical_cores": [28, 29, 30, 31],
             "beam_width": 4, "behaviorally_accepted": False, "runs": {}}
    processes, logs = {}, []

    def publish():
        state["updated_unix"] = time.time()
        write_json(args.output / "status.json", state)

    def stop(*_):
        state["phase"] = "interrupted"
        publish()
        raise SystemExit(130)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        for index, name in enumerate(NAMES):
            gpu = 1 if index == 0 else 0
            cores = list(range(index*7, index*7+7))
            mask = cores + [c+32 for c in cores]
            directory = args.output / name / "training"
            directory.parent.mkdir()
            environment = {**os.environ, "HIP_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "1",
                           "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OMP_WAIT_POLICY": "PASSIVE",
                           "GOMP_SPINCOUNT": "0"}
            command = ["taskset", "-c", ",".join(map(str, mask)), sys.executable, str(ROOT/"scripts/train_cpu.py"),
                       "--library", str(ROOT/"library"), "--output", str(directory), "--stage", "student",
                       "--device", "cuda:0", "--cpu-workers", "14", "--cpu-chunk-size", "4",
                       "--iterations", "12" if args.preflight else "500000", "--num-envs", "2048",
                       "--horizon", "32", "--history", "10", "--hidden-sizes", "512", "256",
                       "--sampling", "take_transition_balanced", "--reference-storage", "packed",
                       "--reference-cache", str(args.reference_cache.resolve()), "--minibatch", "4096",
                       "--epochs", "4", "--learning-rate", "1e-5", "--min-learning-rate", "1e-6",
                       "--kl-stop", ".02", "--bc-weight", "0", "--initialize", str(ROOT/"initialize.pt"),
                       "--evaluation-interval", "0", "--checkpoint-interval", "25",
                       "--milestone-interval", "250", "--threads", "1", "--seed", "42",
                       "--self-collision-weight", "1", "--reward-profile", "legacy",
                       "--root-velocity-weight", "4" if index == 3 else "2", "--root-velocity-sigma", ".5",
                       "--residual-scale", ".25", "--command-velocity-limit", "6"]
            if index:
                command += ["--action-settings", str(ROOT/"configs/controller-pv-arm-feedback-v1.json")]
            if index >= 2:
                command += ["--curriculum-manifest", str(ROOT/"manifests/rl-beam-curriculum.json")]
            if not args.preflight:
                command += ["--max-seconds", str(args.hours*3600)]
            run = {"phase": "starting", "gpu": devices[gpu], "hip_visible_devices": str(gpu),
                   "cpu_affinity": mask, "training_directory": str(directory),
                   "training_directories": [str(directory)], "command": command}
            write_json(directory.parent/"command.json", run)
            log = (directory.parent/"training.log").open("w")
            logs.append(log)
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
            processes[name] = process
            run["pid"] = process.pid
            state["runs"][name] = run
            publish()
        while True:
            running = False
            for name, process in processes.items():
                run = state["runs"][name]
                if run["phase"] in ("completed", "failed"):
                    continue
                directory = Path(run["training_directory"])
                metrics = read_metrics(directory/"metrics.jsonl")
                code = process.poll()
                run["exit_code"] = code
                if metrics:
                    run["latest"] = metrics[-1]
                    run["recent_transitions_per_second"] = statistics.median(r["transitions_per_second"] for r in metrics[-10:])
                if code is None:
                    running = True
                    run["phase"] = "learner_updates" if metrics else "starting"
                elif code or not (directory/"report.json").exists():
                    run.update(phase="failed", error="Trainer exited without a successful terminal report; checkpoints retained")
                else:
                    report = json.loads((directory/"report.json").read_text())
                    valid = (report["finite_updates"] and report["checkpoint_reload_max_error"] == 0
                             and report["start_iteration"] == report["start_transitions"] == report["start_optimizer_steps"] == 0)
                    run.update(phase="completed" if valid else "failed", checkpoint_reload_max_error=report["checkpoint_reload_max_error"],
                               finite_updates=report["finite_updates"], stop_reason=report["stop_reason"])
            failed = any(r["phase"] == "failed" for r in state["runs"].values())
            state["phase"] = "running" if running else "completed_with_failures" if failed else "completed"
            publish()
            if not running:
                break
            time.sleep(5)
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
