#!/usr/bin/env python3
"""Check benchmark input equivalence and summarize measured PPO throughput."""
import argparse
import json
from pathlib import Path
import statistics


def summarize(root):
    runs = []
    anchor = None
    for path in sorted(root.glob("*/benchmark.json")):
        result = json.loads(path.read_text())
        training = json.loads((path.parent / "report.json").read_text())
        key = {k: training[k] for k in ("model_signature", "reference_fingerprint", "observation",
               "action_settings", "reward_settings", "task_version", "initialization_iteration")}
        key.update({k: result[k] for k in ("mujoco", "numpy", "scipy", "physics_dt", "control_dt")})
        key["source_revision"] = result["bundle"]["source_revision"]
        key["manifest_sha256"] = result["bundle"]["manifest_sha256"]
        key["initialize_sha256"] = result["bundle"]["initialize_sha256"]
        key["ppo"] = {k: training["config"][k] for k in ("stage", "horizon", "epochs", "minibatch",
                      "learning_rate", "gamma", "gae_lambda", "clip", "bc_weight", "hidden_sizes",
                      "sampling", "seed", "matmul_precision")}
        if anchor is None:
            anchor = key
        if key != anchor:
            differences = [k for k in anchor if key[k] != anchor[k]]
            raise ValueError(f"Unmatched benchmark contract {path}: {differences}")
        if not result["finite_updates"] or result["checkpoint_reload_max_error"] != 0:
            raise ValueError(f"Failed finite/reload gate: {path}")
        metrics = [json.loads(line) for line in (path.parent / "metrics.jsonl").read_text().splitlines()]
        measured = metrics[result["warmup_discarded"]:]
        rates = [r["transitions_per_second"] for r in measured]
        row = {"name": path.parent.name, **{k: result[k] for k in (
            "hostname", "backend", "device", "gpu", "workers", "num_envs", "threads", "python", "torch",
            "iterations", "measured_iterations", "median_transitions_per_second",
            "aggregate_transitions_per_second", "median_rollout_seconds", "median_update_seconds",
            "peak_torch_vram_bytes", "setup_seconds")},
            "warmed_min_transitions_per_second": min(rates),
            "warmed_max_transitions_per_second": max(rates),
            "warmed_std_transitions_per_second": statistics.stdev(rates) if len(rates) > 1 else 0.,
            "peak_visible_gpu_memory_bytes": max(r.get("visible_cuda_memory_used_bytes", 0) for r in metrics),
            "finite_updates": True, "checkpoint_reload_max_error": 0.,
            "primary_measurement": path.parent.name != "local-warp-128"}
        runs.append(row)
    if not runs:
        raise ValueError("No completed benchmarks")
    return {"complete": True, "matched_contract": anchor, "runs": runs,
            "units": "One transition is one world at 50 Hz, including policy, ten 500-Hz physics substeps and PPO.",
            "limitations": ["Short representative 114-training-clip, 17-family panel, not full-pool learning",
                            "CPU and Warp dynamics and floating-point arithmetic need not be identical",
                            "Python and PyTorch builds differ between the two machines",
                            "No behavioral comparison, controller promotion or long training campaign",
                            "First local-warp-128 calibration overlapped CPU tests; idle repeat is primary"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.directory)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps([{"run": r["name"], "transitions_s": round(r["median_transitions_per_second"], 1),
                       "rollout_s": round(r["median_rollout_seconds"], 3),
                       "update_s": round(r["median_update_seconds"], 3)} for r in report["runs"]], indent=2))


if __name__ == "__main__":
    main()
