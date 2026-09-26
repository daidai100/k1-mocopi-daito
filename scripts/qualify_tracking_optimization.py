#!/usr/bin/env python3
"""Compare frozen pre-optimization and current environments, including resets."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("K1_MOTION_ROOT", str(ROOT))
sys.path.insert(0, str(ROOT/"src"))


def load_baseline(package, name):
    spec = importlib.util.spec_from_file_location(f"k1_motion._baseline_{name}", package/f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-package", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--reference-cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    import torch
    from k1_motion.tracking_env import TrackerEnv
    torch.set_num_threads(1)
    original = load_baseline(args.baseline_package, "tracking_env").TrackerEnv
    original_frames = load_baseline(args.baseline_package, "learning").MotionLibrary.frames

    def compare(left, right):
        if torch.is_tensor(left):
            torch.testing.assert_close(left, right, atol=0, rtol=0)
        elif isinstance(left, dict):
            assert left.keys() == right.keys()
            for key in left:
                compare(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            assert len(left) == len(right)
            for a, b in zip(left, right):
                compare(a, b)
        else:
            assert left == right

    results = []
    for profile in ("legacy", "beyondmimic-causal-v1"):
        for corruption in (False, True):
            options = dict(directory=args.library, num_envs=args.num_envs, device=args.device,
                           backend="mujoco_cpp", physics_options={"workers": args.workers},
                           history=10, reference_storage="packed", reward_profile=profile,
                           corruption=corruption, reference_cache=args.reference_cache, self_collision_weight=1.)
            if profile == "legacy":
                options.update(root_velocity_weight=2., root_velocity_sigma=.5)
            before = original(**options)
            # Independent samplers; immutable reference tensors can be shared.
            options.pop("reference_cache")
            import copy
            shared = copy.copy(before.library)
            shared.__dict__.update({k: v.clone() for k, v in before.library.__dict__.items()
                                    if torch.is_tensor(v)})
            after = TrackerEnv(**options, library=shared)
            before.library.frames = types.MethodType(original_frames, before.library)
            try:
                torch.manual_seed(321)
                first = before.reset()
                torch.manual_seed(321)
                second = after.reset()
                compare(first, second)
                generator = torch.Generator().manual_seed(123)
                ended = 0
                for step in range(args.steps):
                    action = (torch.rand(args.num_envs, 22, generator=generator)*2-1).to(args.device)
                    torch.manual_seed(1000+step)
                    first = before.step(action)
                    torch.manual_seed(1000+step)
                    second = after.step(action)
                    compare(first, second)
                    compare(before.physics.state(), after.physics.state())
                    compare(before.last_step, after.last_step)
                    for key in ("clips", "frames", "command_frames", "episode_steps", "previous_target"):
                        compare(getattr(before, key), getattr(after, key))
                    ended += int(first[3].sum())
                results.append({"profile": profile, "corruption": corruption, "steps": args.steps,
                                "environments": args.num_envs, "episode_ends": ended,
                                "reward_state_observation_termination_exact": True})
            finally:
                before.close()
                after.close()
            print(json.dumps(results[-1]), flush=True)
    receipt = {"baseline_package": str(args.baseline_package), "device": args.device,
               "gpu": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None,
               "results": results, "all_exact": True}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2)+"\n")


if __name__ == "__main__":
    main()
