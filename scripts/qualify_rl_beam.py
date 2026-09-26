#!/usr/bin/env python3
"""Verify native arm projection parity on frozen training poses and bounded stress inputs."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ["K1_MOTION_ROOT"] = str(ROOT)
sys.path.insert(0, str(ROOT/"src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import torch
    from k1_motion.robot import K1Model
    from k1_motion.collision_feedback import ArmCollisionFeedback
    from k1_motion.parallel_collision_feedback import ParallelArmCollisionFeedback
    torch.set_num_threads(1)
    robot = K1Model()
    payload = torch.load(args.cache, weights_only=True, map_location="cpu", mmap=True)
    values = payload["state"]["values"]
    ids = torch.linspace(0, len(values["joint_position"])-1, 384).long()
    rng = np.random.default_rng(92)
    q = np.concatenate((values["joint_position"][ids].numpy(),
                        rng.uniform(robot.limits[:, 0], robot.limits[:, 1], (384, 22)))).astype(np.float32)
    dq = np.concatenate((values["joint_velocity"][ids].numpy(), rng.normal(0, 2, (384, 22)))).astype(np.float32)
    target = q + rng.uniform(-.12, .12, q.shape).astype(np.float32)
    inputs = [torch.tensor(x) for x in (q, dq, target)]
    original = ArmCollisionFeedback(robot, .025)
    native = ParallelArmCollisionFeedback(robot, .025, 14)
    try:
        start = time.monotonic()
        expected = original.project_batch(*inputs)
        python_seconds = time.monotonic()-start
        native.project_batch(*inputs)
        start = time.monotonic()
        actual = native.project_batch(*inputs)
        native_seconds = time.monotonic()-start
        error = float((actual-expected).abs().max())
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0)
        torch.testing.assert_close(actual[:, :2], inputs[2][:, :2], atol=0, rtol=0)
        torch.testing.assert_close(actual[:, 10:], inputs[2][:, 10:], atol=0, rtol=0)
        report = {"passed": True, "poses": len(q), "training_poses": 384, "stress_poses": 384,
                  "max_error_rad": error, "tolerance_rad": 1e-6, "python_seconds": python_seconds,
                  "native_seconds": native_seconds, "speedup": python_seconds/native_seconds,
                  "physics_changed": False, "native_build": native.build}
        args.output.write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps(report), flush=True)
    finally:
        native.close()


if __name__ == "__main__":
    main()
