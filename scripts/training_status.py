#!/usr/bin/env python3
"""Report a run from live process state, saved metrics and replay diagnostics."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def recent_rows(path, maximum=200):
    if not path.exists():
        return []
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 1024 * 1024))
        lines = stream.read().splitlines()
    rows = []
    for line in lines:
        try:
            rows.append(json.loads(line))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
    return rows[-maximum:]


def status(directory):
    directory = Path(directory).resolve()
    pids = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            args = (entry / "cmdline").read_bytes().decode().split("\0")
            if (
                not any(Path(arg).name in ("train_isaac.py", "train_warp.py") for arg in args)
                or "--output" not in args
            ):
                continue
            output = Path(args[args.index("--output") + 1])
            output = (Path(os.readlink(entry / "cwd")) / output).resolve()
            if output == directory:
                pids.append(int(entry.name))
        except (OSError, UnicodeDecodeError, ValueError, IndexError):
            continue
    config_path = directory / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    rows = recent_rows(directory / "metrics.jsonl")
    report = directory / "report.json"
    result = {
        "observed_unix_time": time.time(),
        "run": str(directory),
        "state": "running" if pids else "workload_finished" if report.exists() else "not_running",
        "live_pids": pids,
        "stage": config.get("config", {}).get("stage"),
        "num_envs": config.get("num_envs"),
        "planned_iterations": config.get("config", {}).get("iterations"),
        "target_iteration": config.get("target_iteration", config.get("config", {}).get("iterations")),
        "initialize_checkpoint": config.get("initialize_checkpoint"),
        "model_signature": config.get("model_signature"),
        "task_version": config.get("task_version"),
        "source_revision": config.get("source_revision"),
        "latest_metrics": rows[-1] if rows else None,
        "behaviorally_accepted": False,
    }
    metrics_path = directory / "metrics.jsonl"
    result["metrics_age_seconds"] = time.time() - metrics_path.stat().st_mtime if rows else None
    result["phase"] = result["state"]
    if pids:
        result["phase"] = "initializing" if not rows else "learner_updates"
        interval = config.get("config", {}).get("evaluation_interval", 0)
        if rows and interval and (
            rows[-1]["iteration"] % interval == 0 or rows[-1]["iteration"] == result["target_iteration"]
        ):
            pending = directory / "native-evaluations" / f"iteration-{rows[-1]['iteration']:06d}.json"
            if not pending.exists():
                result["phase"] = "native_replay_pending"
                result["phase_evidence"] = "Learner reached scheduled replay; its result is not yet saved"
    if len(rows) >= 2:
        seconds = rows[-1]["elapsed_seconds"] - rows[0]["elapsed_seconds"]
        iterations = rows[-1]["iteration"] - rows[0]["iteration"]
        if seconds > 0 and iterations > 0:
            result["recent_transitions_per_second"] = (
                rows[-1]["transitions"] - rows[0]["transitions"]
            ) / seconds
            result["rough_remaining_seconds"] = (
                max(0, result["target_iteration"] - rows[-1]["iteration"]) * seconds / iterations
            )
    evaluations = sorted((directory / "native-evaluations").glob("iteration-*.json"))
    if evaluations:
        try:
            data = json.loads(evaluations[-1].read_text())
            result["latest_native_evaluation"] = {
                "path": str(evaluations[-1]),
                "results": {k: {f: v[f] for f in ("completed", "total")} for k, v in data.items()},
                "scope": "Development diagnostic, not exported-student acceptance",
            }
        except json.JSONDecodeError:
            result["native_evaluation_write_in_progress"] = True
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
    )
    result["gpu_util_memory_power"] = gpu.stdout.strip() if gpu.returncode == 0 else gpu.stderr.strip()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", nargs="?", default="artifacts/teacher-isaac-v6-continuation")
    parser.add_argument("--output")
    args = parser.parse_args()
    rendered = json.dumps(status(args.run), indent=2) + "\n"
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered)
    print(rendered, end="")
