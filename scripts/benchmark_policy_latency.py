#!/usr/bin/env python3
"""Batch-one exported CPU policy and controller timing, not hardware acceptance."""
import argparse
from dataclasses import replace
import gc
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def summary(samples):
    import numpy as np
    return {"samples": len(samples), **{f"p{p}_ms": float(np.percentile(samples, p))
                                       for p in (50, 95, 99)}, "max_ms": max(samples),
            "over_5ms": sum(x > 5 for x in samples), "over_20ms": sum(x > 20 for x in samples)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initialize", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--widths", nargs="+", type=int, default=[512, 1024, 2048, 4096, 8192, 16384])
    parser.add_argument("--threads", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    args = parser.parse_args()
    import numpy as np
    import torch
    from k1_motion.learning import Actor, ExportedActor, Policy
    from k1_motion.model_transfer import checkpoint_hidden_sizes, _widen_network
    from k1_motion.robot import K1Model
    from k1_motion.runtime import Controller, Mode

    if args.output.exists() or min(args.samples, args.warmup, *args.threads) < 1:
        raise ValueError("Use a new output and positive sample/thread counts")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    saved = torch.load(args.initialize, map_location="cpu", weights_only=True)
    source = Actor(saved["actor_size"], checkpoint_hidden_sizes(saved)).eval()
    source.load_state_dict({k.removeprefix("actor."): v for k, v in saved["model"].items()
                            if k.startswith("actor.")})
    robot = K1Model()
    result = {"hostname": platform.node(), "cpu_affinity": sorted(os.sched_getaffinity(0)),
              "platform": platform.platform(), "torch": torch.__version__,
              "cpu_model": next((s.split(":", 1)[1].strip() for s in Path("/proc/cpuinfo").read_text().splitlines()
                                  if s.startswith("model name")), "unknown"),
              "initialize": str(args.initialize), "actor_input": saved["actor_size"],
              "critic_input": saved["critic_size"], "actor_output": 22, "dtype": "float32",
              "policy_period_ms": 20, "provisional_policy_p99_budget_ms": 5,
              "scope": "Hot batch-one TorchScript CPU; full tick excludes retargeting, transport and physics. "
                       "This host is a proxy, not the unspecified deployment laptop or Orin.",
              "rows": [], "hardware_accepted": False}
    rng = np.random.default_rng(42)
    samples = (source.normalizer.mean.numpy() + rng.normal(size=(64, saved["actor_size"]))
               * np.sqrt(source.normalizer.variance.numpy()+1e-5)).astype(np.float32)
    for width in args.widths:
        torch.set_num_threads(1)
        torch.manual_seed(42)
        actor = Actor(saved["actor_size"], (width, width//2)).eval()
        with torch.no_grad():
            _widen_network(actor.network, source.network, torch.arange(saved["actor_size"]))
            actor.normalizer.load_state_dict(source.normalizer.state_dict())
            x = torch.from_numpy(samples)
            transfer_error = float((actor(x) - source(x)).abs().max())
        if transfer_error > 2e-5:
            raise ValueError(f"Function-preserving widening failed: {transfer_error}")
        parameters = sum(p.numel() for p in actor.parameters())
        critic_parameters = ((saved["critic_size"]+1)*width + (width+1)*(width//2) + width//2+1)
        with tempfile.TemporaryDirectory(prefix="k1-latency-") as temporary:
            path = Path(temporary)/"actor.pt"
            torch.jit.script(ExportedActor(actor)).save(str(path))
            policy = Policy.__new__(Policy)
            policy.actor = torch.jit.load(str(path), map_location="cpu").eval()
            policy.metadata = {"observation": saved["observation"],
                               "action_settings": saved.get("action_settings", {})}
            export_bytes = path.stat().st_size
        del actor
        gc.collect()
        for threads in args.threads:
            torch.set_num_threads(threads)
            controller = Controller(robot, policy)
            neutral = robot.neutral_reference()
            state = robot.state(0)
            controller.calibrate(neutral, state, True, 0)
            controller.arm(state, 0)
            policy_ms, tick_ms = [], []
            for i in range(args.samples + args.warmup):
                before = time.perf_counter_ns()
                action = policy(samples[i % len(samples)])
                elapsed = (time.perf_counter_ns() - before)/1e6
                if not np.isfinite(action).all():
                    raise FloatingPointError("Nonfinite policy")
                now = (i+1)*robot.control_dt
                controller.set_reference(replace(neutral, source_time=now, received_time=now))
                current = replace(state, time=now)
                before = time.perf_counter_ns()
                command = controller.tick(current, now, True)
                tick_elapsed = (time.perf_counter_ns() - before)/1e6
                if command.mode != Mode.ACTIVE or command.damping_only:
                    raise ValueError("Benchmark did not execute the active controller")
                if i >= args.warmup:
                    policy_ms.append(elapsed)
                    tick_ms.append(tick_elapsed)
            row = {"hidden_sizes": [width, width//2], "threads": threads,
                   "actor_parameters": parameters, "training_parameters": parameters+critic_parameters+22,
                   "actor_fp32_bytes": parameters*4, "export_bytes": export_bytes,
                   "transfer_max_error": transfer_error, "policy": summary(policy_ms),
                   "controller_tick": summary(tick_ms)}
            row["provisional_cpu_budget_pass"] = row["policy"]["p99_ms"] <= 5 and row["controller_tick"]["p99_ms"] <= 20
            result["rows"].append(row)
            args.output.write_text(json.dumps(result, indent=2)+"\n")
            print(json.dumps(row), flush=True)
        del policy, controller
        gc.collect()


if __name__ == "__main__":
    main()
