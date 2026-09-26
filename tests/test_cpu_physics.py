"""Compiled stepping preserves the scalar MuJoCo/PD task, not Warp numerics."""
import numpy as np
import pytest
import torch

from k1_motion.cpu_physics import CpuParallelPhysics
from k1_motion.math3d import quaternion
from k1_motion.robot import K1Model
from k1_motion.tracking_env import MujocoPhysics
from scipy.spatial.transform import Rotation


@pytest.mark.parametrize("chunk", [0, 1, 4])
def test_cpp_matches_native_contacts_velocities_and_partial_resets(chunk):
    torch.set_num_threads(1)
    robot = K1Model()
    settings = {"kp_scale": [1.05]*22, "kd_scale": [1.1]*22}
    native = MujocoPhysics(robot, 4, "cpu", settings)
    cpp = CpuParallelPhysics(robot, 4, "cpu", settings, workers=2, chunk_size=chunk)
    try:
        ref = robot.neutral_reference()
        keys = ("root_position", "root_orientation", "joint_position", "root_velocity", "joint_velocity")
        reset = {k: torch.tensor(np.tile(getattr(ref, k), (4, 1)), dtype=torch.float32) for k in keys}
        reset["root_orientation"][:] = torch.tensor(quaternion(Rotation.from_euler("xyz", [.05, -.04, .5])))
        reset["root_velocity"][:] = torch.tensor([.07, -.03, .02, .02, .03, -.04])
        reset["joint_velocity"][:, 2] = .1
        reset["root_position"][3, 2] -= .12
        for physics in (native, cpp):
            physics.reset(torch.arange(4), reset)
        saved = {k: v.clone() for k, v in cpp.state().items()}
        retained = cpp.state()
        collision_seen = False
        for tick in range(25):
            target = reset["joint_position"].clone()
            target[:, 2] += .25*np.sin(tick*.3)
            target[2, 3:6] = 10.  # Joint and torque clipping, and self-contact.
            velocity = reset["joint_velocity"]*.2 if tick % 2 else None
            for physics in (native, cpp):
                physics.step(target, velocity)
            for key, value in native.state().items():
                torch.testing.assert_close(cpp.state()[key], value, rtol=0, atol=0)
            torch.testing.assert_close(cpp.effort_per_env, native.effort_per_env, rtol=1e-7, atol=1e-8)
            torch.testing.assert_close(cpp.self_collision_per_env, native.self_collision_per_env, rtol=0, atol=0)
            assert cpp.saturation == pytest.approx(native.saturation, abs=1e-7)
            collision_seen |= bool(cpp.self_collision_per_env.any())
            if tick == 10:
                ids = torch.tensor([3, 0])
                for physics in (native, cpp):
                    physics.reset(ids, {k: v[ids] for k, v in reset.items()})
        assert collision_seen
        for key in saved:
            torch.testing.assert_close(saved[key], retained[key], rtol=0, atol=0)
        cpp.validate()
        with pytest.raises(ValueError, match="unique"):
            cpp.reset(torch.tensor([0, 0]), {k: v[:2] for k, v in reset.items()})
        with pytest.raises(ValueError, match="Nonfinite"):
            cpp.step(torch.full((4, 22), float("nan")))
    finally:
        native.close()
        cpp.close()
        cpp.close()
    with pytest.raises(RuntimeError, match="closed"):
        cpp.state()


@pytest.mark.parametrize("chunk", [0, 4])
def test_fused_feedback_preserves_clamps_physics_and_partial_resets(tmp_path, chunk):
    from test_training import make_library
    from k1_motion.tracking_env import TrackerEnv
    _, directory = make_library(tmp_path)
    settings = {"arm_collision_clearance": .025, "target_velocity_scale": .25,
                "upper_body_residual_scale": 0., "command_velocity_limit": 6.}
    envs = [TrackerEnv(directory, 8, "cpu", "mujoco_cpp", action_settings=settings,
                      physics_options={"workers": 2, "chunk_size": chunk, "fuse_arm_feedback": fused})
            for fused in (False, True)]
    torch.set_num_threads(1)
    try:
        for env in envs:
            torch.manual_seed(100)
            env.reset()
        for tick in range(70):
            action = torch.sin(torch.arange(8*22).reshape(8, 22)*.7+tick)
            results = []
            for env in envs:
                torch.manual_seed(1000+tick)
                mask = torch.arange(8) % 2 == 0 if tick % 11 == 10 else None
                results.append(env.step(action, actuation_mask=mask))
            for index in range(4):
                torch.testing.assert_close(results[0][index], results[1][index], rtol=0, atol=0)
            for key in envs[0].physics.state():
                torch.testing.assert_close(envs[0].physics.state()[key], envs[1].physics.state()[key],
                                           rtol=0, atol=0)
            torch.testing.assert_close(envs[0].previous_target, envs[1].previous_target, rtol=0, atol=0)
            if tick == 30:
                for env in envs:
                    torch.manual_seed(55)
                    env.reset(torch.tensor([1, 5, 7]))
        with pytest.raises(ValueError, match="Nonfinite"):
            envs[1].physics.step_projected(torch.full((8, 22), float("nan")), torch.zeros(8, 22),
                                           envs[1].previous_target, envs[1].arm_feedback)
    finally:
        for env in envs:
            env.close()
