#!/usr/bin/env python3
"""Exact trajectory parity against an immutable complete pre-change package."""
import argparse
import copy
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("K1_MOTION_ROOT", str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def baseline_module(package):
    name = "k1_motion_pipeline_baseline"
    spec = importlib.util.spec_from_file_location(name, package / "__init__.py",
                                                 submodule_search_locations=[str(package)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return importlib.import_module(name + ".tracking_env")


def compare(left, right):
    import torch
    groups = {}

    def collect(a, b):
        if torch.is_tensor(a):
            assert a.shape == b.shape and a.dtype == b.dtype
            pair = groups.setdefault(a.dtype, ([], []))
            pair[0].append(a.flatten())
            pair[1].append(b.flatten())
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for key in a:
                collect(a[key], b[key])
        elif isinstance(a, (tuple, list)):
            assert len(a) == len(b)
            for x, y in zip(a, b):
                collect(x, y)
        else:
            assert a == b

    collect(left, right)
    for a, b in groups.values():
        torch.testing.assert_close(torch.cat(a).cpu(), torch.cat(b).cpu(), atol=0, rtol=0)


def clone_sampler(library):
    import torch
    shared = copy.copy(library)
    shared.__dict__.update({k: v.clone() for k, v in library.__dict__.items() if torch.is_tensor(v)})
    return shared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-package", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--reference-cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--action-settings", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    import torch
    from k1_motion.tracking_env import TrackerEnv
    torch.set_num_threads(1)
    original = baseline_module(args.baseline_package.resolve()).TrackerEnv
    settings = json.loads(args.action_settings.read_text())
    results = []
    prototype = None
    started = time.monotonic()
    for feedback in (False, True):
        for corruption in (False, True):
            options = dict(directory=args.library, num_envs=args.num_envs, device=args.device,
                           backend="mujoco_cpp", physics_options={"workers": args.workers},
                           history=10, reference_storage="packed", reward_profile="legacy",
                           root_velocity_weight=2., root_velocity_sigma=.5, self_collision_weight=1.,
                           corruption=corruption, reference_cache=args.reference_cache,
                           action_settings=settings if feedback else {})
            if prototype is None:
                before = original(**options)
                prototype = clone_sampler(before.library)
                options.pop("reference_cache")
            else:
                options.pop("reference_cache")
                before = original(**options, library=clone_sampler(prototype))
            shared = clone_sampler(before.library)
            after = TrackerEnv(**options, library=shared)
            try:
                torch.manual_seed(321)
                first = before.reset()
                torch.manual_seed(321)
                second = after.reset()
                compare(first, second)
                generator = torch.Generator().manual_seed(123)
                ended = 0
                masked_steps = 0
                for step in range(args.steps):
                    action = (torch.rand(args.num_envs, 22, generator=generator)*2-1).to(args.device)
                    mask = None
                    if step % 17 == 16:
                        mask = torch.arange(args.num_envs, device=args.device) % 2 == 0
                        masked_steps += 1
                    torch.manual_seed(1000+step)
                    first = before.step(action, actuation_mask=mask)
                    torch.manual_seed(1000+step)
                    second = after.step(action, actuation_mask=mask)
                    keys = ("clips", "frames", "command_frames", "episode_steps", "previous_target")
                    compare((first, before.physics.state(), before.last_step, [getattr(before, k) for k in keys]),
                            (second, after.physics.state(), after.last_step, [getattr(after, k) for k in keys]))
                    ended += int(first[3].sum())
                results.append({"feedback": feedback, "corruption": corruption, "steps": args.steps,
                                "environments": args.num_envs, "episode_ends": ended,
                                "masked_steps": masked_steps, "all_exact": True})
                print(json.dumps(results[-1]), flush=True)
            finally:
                before.close()
                after.close()
    report = {"baseline_package": str(args.baseline_package), "device": args.device,
              "results": results, "all_exact": True, "elapsed_seconds": time.monotonic()-started}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
