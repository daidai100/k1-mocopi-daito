#!/usr/bin/env python3
"""Sequential real-physics PPO benchmarks, retaining logs and GPU measurements."""

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from freeze_source import freeze_source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--initialize")
    source.add_argument("--teacher")
    parser.add_argument("--output", required=True)
    parser.add_argument("--sizes", nargs="+", type=int, default=[1024, 2048, 4096])
    parser.add_argument("--iterations", type=int, default=24)
    parser.add_argument("--isaac-python", default=".venv-isaac/bin/python")
    parser.add_argument("--backend", choices=("isaac", "warp"), default="isaac")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--corruption", action="store_true")
    parser.add_argument("--no-conditional-graphs", action="store_true")
    parser.add_argument(
        "--no-gpu-sampling", action="store_true", help="Use visible-device memory metrics on MIG"
    )
    args = parser.parse_args()
    if args.iterations < 6 or min(args.sizes) < 1:
        parser.error("Use at least six iterations and positive environment counts")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    frozen_source, revision = freeze_source(Path(__file__).resolve().parents[1])
    environment = {**os.environ, "K1_FROZEN_SOURCE": str(frozen_source / "k1_motion")}
    summaries = []
    for size in args.sizes:
        run = output / str(size)
        command = [
            args.isaac_python if args.backend == "isaac" else args.python,
            f"scripts/train_{args.backend}.py",
            "--library",
            args.library,
            "--output",
            str(run),
            "--num-envs",
            str(size),
            "--minibatch",
            str(size * 4),
            "--iterations",
            str(args.iterations),
            "--evaluation-interval",
            "0",
            "--history",
            "10",
            "--hidden-sizes",
            "512",
            "256",
        ]
        command += (
            ["--stage", "student", "--teacher", args.teacher]
            if args.teacher
            else ["--initialize", args.initialize]
        )
        if args.corruption:
            command.append("--corruption")
        if args.backend == "warp" and args.no_conditional_graphs:
            command.append("--no-conditional-graphs")
        gpu_samples = []
        with (output / f"{size}.log").open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
            while process.poll() is None:
                if args.no_gpu_sampling:
                    time.sleep(1)
                    continue
                gpu = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used,power.draw",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                )
                if gpu.returncode == 0:
                    values = [float(v) for v in gpu.stdout.strip().split(",")]
                    gpu_samples.append(
                        {
                            "unix_time": time.time(),
                            "utilization_pct": values[0],
                            "memory_mib": values[1],
                            "power_w": values[2],
                            "training_started": (run / "metrics.jsonl").exists(),
                        }
                    )
                time.sleep(1)
        (output / f"{size}-gpu.json").write_text(json.dumps(gpu_samples, indent=2) + "\n")
        summary = {
            "num_envs": size,
            "exit_code": process.returncode,
            "command": command,
            "source_revision": revision,
        }
        if process.returncode == 0 and (run / "report.json").exists():
            metrics = [json.loads(s) for s in (run / "metrics.jsonl").read_text().splitlines()][5:]
            live = [s for s in gpu_samples if s["training_started"]]
            summary.update(
                {
                    "warmed_iterations": len(metrics),
                    "median_transitions_per_second": statistics.median(
                        r["transitions_per_second"] for r in metrics
                    ),
                    "median_rollout_seconds": statistics.median(r["rollout_seconds"] for r in metrics),
                    "median_update_seconds": statistics.median(r["update_seconds"] for r in metrics),
                    "peak_gpu_memory_mib": max(s["memory_mib"] for s in gpu_samples) if gpu_samples else None,
                    "median_gpu_utilization_pct": statistics.median(s["utilization_pct"] for s in live)
                    if live
                    else None,
                    "median_gpu_power_w": statistics.median(s["power_w"] for s in live) if live else None,
                    "peak_visible_cuda_memory_bytes": max(
                        r.get("visible_cuda_memory_used_bytes", 0) for r in metrics
                    ),
                    "checkpoint_reload_max_error": json.loads((run / "report.json").read_text())[
                        "checkpoint_reload_max_error"
                    ],
                }
            )
        summaries.append(summary)
        (output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
        print(json.dumps(summary), flush=True)
        # A runtime failure requires inspection before attempting another size.
        if process.returncode:
            raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
