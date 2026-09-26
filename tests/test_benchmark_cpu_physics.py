"""Benchmark worker parallelism must not change the native physics contract."""
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from bench_cpu_physics import ParallelCpuPhysics
from k1_motion.robot import K1Model
from k1_motion.tracking_env import MujocoPhysics


def test_parallel_cpu_matches_native_step_and_partial_reset():
    torch.set_num_threads(1)
    robot = K1Model()
    settings = {"residual_scale": .25, "command_velocity_limit": 6.}
    native = MujocoPhysics(robot, 4, "cpu", settings)
    parallel = ParallelCpuPhysics(robot, 4, "cpu", workers=2, action_settings=settings)
    try:
        ref = robot.neutral_reference()
        keys = ("root_position", "root_orientation", "joint_position", "root_velocity", "joint_velocity")
        reset = {k: torch.tensor(np.tile(getattr(ref, k), (4, 1)), dtype=torch.float32) for k in keys}
        reset["root_velocity"][:, :3] = torch.tensor([.07, -.03, .02])
        reset["root_velocity"][:, 3:] = torch.tensor([.02, .03, -.04])
        reset["joint_velocity"][:, 2] = .1
        for physics in (native, parallel):
            physics.reset(torch.arange(4), reset)
        for tick in range(8):
            target = reset["joint_position"].clone()
            target[:, 2] += .025 * np.sin(tick * .3)
            velocity = reset["joint_velocity"].clone() * .2
            for physics in (native, parallel):
                physics.step(target, velocity)
            for key, value in native.state().items():
                torch.testing.assert_close(parallel.state()[key], value, rtol=0, atol=0)
            torch.testing.assert_close(parallel.effort_per_env, native.effort_per_env, rtol=0, atol=0)
            torch.testing.assert_close(parallel.self_collision_per_env, native.self_collision_per_env,
                                       rtol=0, atol=0)
            if tick == 3:
                ids = torch.tensor([3, 0])
                for physics in (native, parallel):
                    physics.reset(ids, {k: v[ids] for k, v in reset.items()})
        parallel.validate()
    finally:
        native.close()
        parallel.close()
    assert all(not p.is_alive() for p in parallel.processes)
