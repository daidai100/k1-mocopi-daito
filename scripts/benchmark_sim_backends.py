#!/usr/bin/env python3
"""Bounded, matched K1 PPO/physics benchmarks, not a training campaign."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ["K1_MOTION_ROOT"] = str(ROOT)
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]


def prepare(args):
    from freeze_source import freeze_source
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source, revision = freeze_source(ROOT)
    shutil.copytree(source / "k1_motion", output / "src/k1_motion")
    (output / "scripts").mkdir()
    for name in ("benchmark_sim_backends.py", "bench_cpu_physics.py", "freeze_source.py",
                 "train_cpu.py", "train_warp.py"):
        shutil.copy2(ROOT / "scripts" / name, output / "scripts" / name)
    shutil.copytree(ROOT / "configs", output / "configs")
    robot = json.loads((ROOT / "configs/k1.json").read_text())["model"]
    shutil.copytree((ROOT / robot).parent, (output / robot).parent)
    library = output / "library"
    (library / "clips").mkdir(parents=True)
    rows = [json.loads(line) for line in (Path(args.library) / "index.jsonl").read_text().splitlines()]
    for row in rows:
        path = Path(args.library) / row["reference_path"]
        relative = Path("clips") / (row["id"] + ".npz")
        shutil.copy2(path, library / relative)
        row["reference_path"] = str(relative)
    manifest = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    (library / "index.jsonl").write_text(manifest)
    shutil.copy2(args.initialize, output / "initialize.pt")
    receipt = {"source_revision": revision,
               "parent_library": str(Path(args.library).resolve()),
               "manifest_sha256": hashlib.sha256(manifest.encode()).hexdigest(),
               "initialize_sha256": hashlib.sha256((output / "initialize.pt").read_bytes()).hexdigest(),
               "train_families": dict(Counter(r["family"] for r in rows if r["split"] == "train")),
               "splits": dict(Counter(r["split"] for r in rows)),
               "portable_paths_only": True, "trajectories_unchanged": True}
    (output / "bundle.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)


def run(args):
    import mujoco
    import numpy as np
    import scipy
    import torch
    from k1_motion.learning import TrainConfig, train
    from k1_motion.tracking_env import TrackerEnv

    output = Path(args.output)
    if output.exists():
        raise ValueError("Use a new benchmark output")
    torch.set_num_threads(args.threads)
    if args.device.startswith("cuda"):
        torch.cuda.set_device(args.device)
        if args.expected_gpu and args.expected_gpu not in torch.cuda.get_device_name():
            raise ValueError("Unexpected GPU identity")
    source = json.loads((ROOT / "bundle.json").read_text())
    os.environ["K1_SOURCE_REVISION"] = source["source_revision"]
    options = {"nconmax": 128, "njmax": 1024, "epa_horizon": 96, "conditional_graphs": False}
    if args.backend == "mujoco_parallel":
        import k1_motion.tracking_env as tracking
        from bench_cpu_physics import ParallelCpuPhysics
        # Scoped to this benchmark process. Production code and existing exports
        # remain untouched. Gold/parallel equivalence has a separate regression.
        tracking.MujocoPhysics = lambda spec, count, device, action_settings=None: ParallelCpuPhysics(
            spec, count, device, workers=args.workers, action_settings=action_settings)
    started = time.monotonic()
    selected_backend = "mujoco" if args.backend == "mujoco_parallel" else args.backend
    if args.backend == "mujoco_cpp":
        options = {"workers": args.workers, "chunk_size": args.chunk_size}
    env = TrackerEnv(str(ROOT / "library"), num_envs=args.num_envs, device=args.device,
                     backend=selected_backend, history=10,
                     self_collision_weight=1., root_velocity_weight=2., root_velocity_sigma=.5,
                     reference_storage="packed",
                     physics_options=options if args.backend in ("warp", "mujoco_cpp") else None)
    setup_seconds = time.monotonic() - started
    metadata = {"hostname": platform.node(), "python": platform.python_version(),
                "torch": torch.__version__, "torch_cuda": torch.version.cuda, "torch_hip": torch.version.hip,
                "mujoco": mujoco.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
                "gpu": torch.cuda.get_device_name() if args.device.startswith("cuda") else None,
                "cpu_count": os.cpu_count(), "cpu_affinity": sorted(os.sched_getaffinity(0)),
                "backend": args.backend, "device": args.device, "workers": args.workers,
                "chunk_size": args.chunk_size, "physics_contract": getattr(env.physics, "contract", None),
                "num_envs": args.num_envs, "setup_seconds": setup_seconds, "bundle": source,
                "model_signature": env.spec.signature, "physics_dt": float(env.spec.model.opt.timestep),
                "control_dt": env.spec.control_dt, "physics_substeps": env.spec.substeps,
                "threads": args.threads, "benchmark_only": True}
    print(json.dumps({"benchmark_start": metadata}), flush=True)
    try:
        report = train(env, output, TrainConfig(
            stage="student", iterations=args.iterations, horizon=32, epochs=4,
            minibatch=args.minibatch, learning_rate=5e-5, bc_weight=0.,
            evaluation_interval=0, checkpoint_interval=args.iterations,
            hidden_sizes=(512, 256), sampling="take_transition_balanced", seed=args.seed),
            initialize_checkpoint=ROOT / "initialize.pt")
    finally:
        physics_timing = getattr(env.physics, "timing", None)
        env.close()
    metrics = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    measured = metrics[args.warmup:]
    summary = {**metadata, "physics_timing": physics_timing,
               "iterations": len(metrics), "warmup_discarded": args.warmup,
               "measured_iterations": len(measured), "transitions": report["transitions"],
               "finite_updates": report["finite_updates"],
               "checkpoint_reload_max_error": report["checkpoint_reload_max_error"],
               "reference_storage_bytes": report["reference_storage_bytes"],
               "median_transitions_per_second": statistics.median(r["transitions_per_second"] for r in measured),
               "aggregate_transitions_per_second": args.num_envs * 32 * len(measured)
                   / sum(r["iteration_seconds"] for r in measured),
               "median_rollout_seconds": statistics.median(r["rollout_seconds"] for r in measured),
               "median_update_seconds": statistics.median(r["update_seconds"] for r in measured),
               "peak_torch_vram_bytes": report["peak_vram_bytes"],
               "family_transition_share": {f: statistics.mean(r["family_transition_share"][f]
                   for r in measured) for f in measured[0]["family_transition_share"]},
               "behaviorally_accepted": False,
               "scope": "Warm end-to-end PPO throughput; excludes setup, evaluation and final save/reload."}
    (output / "benchmark.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"benchmark_complete": summary}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("prepare", "run"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--library")
    parser.add_argument("--initialize")
    parser.add_argument("--backend", choices=("warp", "mujoco", "mujoco_parallel", "mujoco_cpp"), default="warp")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--expected-gpu")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=4, help="C++ dynamic chunk; zero selects static scheduling")
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=18)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--minibatch", type=int, default=4096)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.operation == "prepare":
        if not args.library or not args.initialize:
            parser.error("Preparation requires a library and initializer")
        prepare(args)
    else:
        if args.iterations <= args.warmup or args.warmup < 0 or min(args.num_envs, args.threads) < 1:
            parser.error("Require measured iterations, nonnegative warmup and positive sizes")
        run(args)


if __name__ == "__main__":
    main()
