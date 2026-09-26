#!/usr/bin/env python3
"""Bounded full-PPO GPU/environment qualification; fresh state for each setting."""
import argparse
import json
import os
from pathlib import Path
import statistics
import shutil
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-cache", type=Path, required=True)
    parser.add_argument("--cpu-mask", required=True)
    parser.add_argument("--workers", type=int, default=15)
    parser.add_argument("--num-envs", type=int, nargs="+", default=[1024, 2048, 4096])
    parser.add_argument("--iterations", type=int, default=18)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--resume-checkpoint", type=Path, help="Qualify real optimizer continuation, preserving the input")
    parser.add_argument("--profile", choices=("legacy", "beyondmimic-causal-v1"), default="beyondmimic-causal-v1")
    args = parser.parse_args()
    if args.iterations <= args.warmup or min(args.num_envs) <= 0:
        raise ValueError("Positive environment counts and measured post-warmup iterations required")
    args.bundle = args.bundle.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    torch.set_num_threads(1)
    if torch.cuda.device_count() != 1:
        raise ValueError("Isolate exactly one GPU for qualification")
    summary = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
               "hip": torch.version.hip, "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
               "bundle": str(args.bundle), "cpu_mask": args.cpu_mask, "workers": args.workers,
               "hidden_sizes": [args.width, args.width//2], "profile": args.profile,
               "warmup": args.warmup, "results": {}, "behaviorally_accepted": False}
    if args.resume_checkpoint:
        preserved = args.output/"resume-input"
        preserved.mkdir()
        shutil.copy2(args.resume_checkpoint, preserved/"checkpoint.pt")
        if (args.resume_checkpoint.parent/"config.json").exists():
            shutil.copy2(args.resume_checkpoint.parent/"config.json", preserved/"config.json")
        args.resume_checkpoint = preserved/"checkpoint.pt"
    for count in args.num_envs:
        output = args.output / str(count)
        command = ["taskset", "-c", args.cpu_mask, sys.executable,
                   str(args.bundle / "scripts/train_cpu.py"), "--library", str(args.bundle/"library"),
                   "--output", str(output), "--stage", "student", "--device", "cuda:0",
                   "--cpu-workers", str(args.workers), "--cpu-chunk-size", "4",
                   "--iterations", str(args.iterations), "--num-envs", str(count),
                   "--horizon", "32", "--history", "10", "--hidden-sizes", str(args.width), str(args.width//2),
                   "--sampling", "take_transition_balanced", "--reference-storage", "packed",
                   "--reference-cache", str(args.reference_cache.resolve()), "--minibatch", "4096",
                   "--epochs", "4", "--learning-rate", "1e-5", "--bc-weight", "0",
                   "--evaluation-interval", "0",
                   "--checkpoint-interval", "1000000", "--threads", "1", "--seed", "42",
                   "--self-collision-weight", "1", "--reward-profile", args.profile,
                   "--residual-scale", ".25", "--command-velocity-limit", "6"]
        if args.resume_checkpoint:
            command += ["--resume", str(args.resume_checkpoint), "--allow-env-resize"]
        else:
            command += ["--initialize", str(args.bundle/"initialize.pt")]
        if args.profile == "legacy":
            command += ["--root-velocity-weight", "2", "--root-velocity-sigma", ".5"]
        (args.output/f"{count}-command.json").write_text(json.dumps(command, indent=2)+"\n")
        started = time.monotonic()
        with (args.output/f"{count}.log").open("w") as log:
            subprocess.run(command, cwd=args.bundle, stdout=log, stderr=subprocess.STDOUT, check=True)
        report = json.loads((output/"report.json").read_text())
        if not report["finite_updates"] or report["checkpoint_reload_max_error"] != 0:
            raise RuntimeError("PPO finite/reload qualification failed")
        metrics = [json.loads(line) for line in (output/"metrics.jsonl").read_text().splitlines()][args.warmup:]
        result = {"aggregate_transitions_per_second": len(metrics)*count*32/sum(r["iteration_seconds"] for r in metrics),
                  "median_rollout_seconds": statistics.median(r["rollout_seconds"] for r in metrics),
                  "median_update_seconds": statistics.median(r["update_seconds"] for r in metrics),
                  "peak_torch_vram_bytes": report["peak_vram_bytes"],
                  "peak_visible_vram_bytes": max(r["visible_cuda_memory_used_bytes"] for r in metrics),
                  "finite_updates": True, "checkpoint_reload_max_error": 0,
                  "wall_seconds_including_setup": time.monotonic()-started,
                  "source_revision": report["source_revision"],
                  "reference_fingerprint": report["reference_fingerprint"],
                  "reference_storage_bytes": report["reference_storage_bytes"]}
        if args.resume_checkpoint:
            result.update(start_iteration=report["start_iteration"], start_transitions=report["start_transitions"],
                          start_optimizer_steps=report["start_optimizer_steps"],
                          final_iteration=report["last_metrics"]["iteration"], final_transitions=report["transitions"],
                          final_optimizer_steps=report["optimizer_steps"])
        summary["results"][str(count)] = result
        (args.output/"comparison.json").write_text(json.dumps(summary, indent=2)+"\n")
        print(json.dumps({"num_envs": count, **result}), flush=True)


if __name__ == "__main__":
    main()
