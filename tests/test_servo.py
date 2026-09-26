"""Behavioral contracts for causal velocity commands and simulation candidates."""

from dataclasses import replace
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

from k1_motion.actuation import action_settings
from k1_motion.contracts import MotionClip
from k1_motion.learning import MotionLibrary
from k1_motion.observations import reference_tensor, targets_tensor
from k1_motion.robot import K1Model
from k1_motion.runtime import Controller, Mode
from k1_motion.servo import step_command, step_pd
from k1_motion.tracking_env import TrackerEnv
from test_training import make_library


def test_position_velocity_matches_training_and_runtime(tmp_path):
    robot, directory = make_library(tmp_path)
    path = directory / "standing.npz"
    clip = MotionClip.load(path)
    refs = []
    for i in range(50):
        ref = robot.neutral_reference(i * 0.02)
        q, dq = ref.joint_position.copy(), np.zeros(22)
        q[2] += 0.1 * i * 0.02
        dq[2] = 0.1
        refs.append(replace(ref, joint_position=q, joint_velocity=dq))
    MotionClip.from_references(refs, clip.metadata).save(path)
    settings = {
        "target_velocity_scale": 0.5,
        "kp_scale": 1.1,
        "kd_scale": 0.9,
        "upper_body_residual_scale": 0.2,
        "imu_reference_rate_scale": 1.0,
        "arm_collision_clearance": 0.025,
    }

    class Policy:
        metadata = {"action_settings": settings}

        def __call__(self, observation):
            return np.full(22, 0.03, dtype=np.float32)

    env = TrackerEnv(directory, num_envs=1, device="cpu", action_settings=settings)
    zero = torch.zeros(1, dtype=torch.long)
    env.reset(clips=zero, frames=zero)
    robot.reset(refs[0])
    robot.data.qvel[6:] = refs[0].joint_velocity
    mujoco.mj_forward(robot.model, robot.data)
    controller = Controller(robot, Policy())
    controller.calibrate(refs[0], robot.state(0), True, 0)
    controller.arm(robot.state(0), 0)
    for i in range(5):
        now = i * robot.control_dt
        if i:
            controller.set_reference(refs[i])
        command = controller.tick(robot.state(now), now)
        assert command.velocities[2] == pytest.approx(0.05)
        env.step(torch.full((1, 22), 0.03), auto_reset=False)
        step_command(robot, command, controller.action_settings)
        np.testing.assert_allclose(env.physics.robots[0].data.qpos, robot.data.qpos, atol=2e-7)
        np.testing.assert_allclose(env.physics.robots[0].data.qvel, robot.data.qvel, atol=2e-6)
    env.close()


def test_stale_reference_pause_and_fault_clear_velocity():
    robot = K1Model()
    robot.reset()

    class Policy:
        metadata = {"action_settings": {"target_velocity_scale": 1.0}}

        def __call__(self, observation):
            return np.zeros(22)

    ref = replace(robot.neutral_reference(), joint_velocity=np.ones(22) * 2)
    c = Controller(robot, Policy())
    c.calibrate(ref, robot.state(0), True, 0)
    c.arm(robot.state(0), 0)
    assert np.all(c.tick(robot.state(0), 0, True).velocities == 2)
    c.tick(robot.state(0.04), 0.04, True)
    assert np.all(c.tick(robot.state(0.06), 0.06, True).velocities == 0)
    c.pause(0.07)
    assert np.all(c.tick(robot.state(0.08), 0.08, True).velocities == 0)
    c.stop(0.09)
    command = c.tick(robot.state(0.1), 0.1, True)
    assert command.damping_only and command.mode == Mode.STOPPED
    # A nonzero velocity mistakenly supplied to a damping command cannot inject energy.
    robot.data.qvel[6:] = 0.1
    a = robot.data.qpos.copy()
    step_pd(robot, a[7:], np.ones(22) * 100, damping_only=True)
    assert np.isfinite(robot.data.qpos).all()
    assert np.max(np.abs(robot.data.ctrl)) < 2


def test_imu_uses_reference_angular_velocity_in_body_frame():
    robot = K1Model()
    orientation = torch.tensor([[np.cos(0.4), 0.0, 0.0, np.sin(0.4)]], dtype=torch.float32)
    ref = reference_tensor(
        replace(
            robot.neutral_reference(),
            root_orientation=orientation[0].numpy(),
            root_velocity=np.array([0, 0, 0, 0.2, -0.1, 0.0]),
        )
    )
    from k1_motion.observations import quat_apply, quat_inv

    matching = quat_apply(quat_inv(orientation), ref["root_velocity"][:, 3:])
    target = targets_tensor(
        torch.zeros((1, 22)),
        ref,
        orientation,
        matching,
        torch.tensor(robot.limits, dtype=torch.float32),
        imu_reference_rate_scale=1.0,
    )
    torch.testing.assert_close(target, ref["joint_position"])
    disturbed = targets_tensor(
        torch.zeros((1, 22)),
        ref,
        orientation,
        matching + torch.tensor([[0.0, 0.2, 0.0]]),
        torch.tensor(robot.limits, dtype=torch.float32),
        imu_reference_rate_scale=1.0,
    )
    assert float(disturbed[0, 14] - target[0, 14]) == pytest.approx(0.03, abs=1e-6)


def test_unqualified_references_require_explicit_audited_candidate_mode(tmp_path):
    robot, directory = make_library(tmp_path)
    index = directory / "index.jsonl"
    row = json.loads(index.read_text())
    row.update(
        training_eligible=False,
        physics_qualified=False,
        simulation_candidate=True,
        recovery_audit={"accepted": True},
    )
    index.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="No accepted"):
        MotionLibrary(directory, robot, "cpu")
    candidates = MotionLibrary(directory, robot, "cpu", candidate_training=True)
    assert len(candidates.rows) == 1 and not candidates.rows[0]["physics_qualified"]
    row["recovery_audit"]["accepted"] = False
    index.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="No accepted"):
        MotionLibrary(directory, robot, "cpu", candidate_training=True)
    row["recovery_audit"]["accepted"] = True
    row["split"] = "test"
    index.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="No accepted"):
        MotionLibrary(directory, robot, "cpu", candidate_training=True)


def test_invalid_servo_settings_rejected():
    robot = K1Model()
    for settings in (
        {"kp_scale": 0},
        {"kd_scale": [1] * 21},
        {"target_velocity_scale": float("nan")},
        {"upper_body_residual_scale": -1},
    ):
        with pytest.raises(ValueError):
            action_settings(robot, settings)


def test_arm_feedback_increases_clearance_without_changing_leg_balance():
    from k1_motion.collision_feedback import ArmCollisionFeedback

    robot = K1Model()
    fixture = json.loads((Path(__file__).parent / "fixtures/k1_forearm_feedback.json").read_text())
    robot.data.qpos[:] = fixture["qpos"]
    mujoco.mj_forward(robot.model, robot.data)
    g1, g2 = fixture["geom_pair"]
    before = mujoco.mj_geomDistance(robot.model, robot.data, g1, g2, 0.1, None)
    assert before <= 0
    q, dq = np.asarray(fixture["qpos"])[7:], np.asarray(fixture["qvel"])[6:]
    feedback = ArmCollisionFeedback(robot)
    np.testing.assert_array_equal(robot.model.geom_margin, 0)
    corrected = feedback.project(q, dq, q)
    np.testing.assert_array_equal(corrected[10:], q[10:])
    np.testing.assert_array_equal(corrected[:2], q[:2])
    assert np.max(abs(corrected - q)) <= 0.15 + 1e-12
    robot.data.qpos[7:] = corrected
    mujoco.mj_forward(robot.model, robot.data)
    after = mujoco.mj_geomDistance(robot.model, robot.data, g1, g2, 0.1, None)
    assert after > before + 0.005
    # Intervening worlds and rigid world translations cannot contaminate feedback state.
    feedback.project(robot.neutral, np.zeros(22), robot.neutral)
    np.testing.assert_array_equal(feedback.project(q, dq, q), corrected)
    np.testing.assert_array_equal(feedback.project(q, dq, q)[10:], q[10:])


def test_position_velocity_feedback_is_causal_and_resets_between_runs():
    robot = K1Model()

    class Policy:
        metadata = {"action_settings": {"target_velocity_scale": 0.25, "arm_collision_clearance": 0.025}}

        def __call__(self, observation):
            return np.zeros(22)

    def run(future):
        robot.reset()
        controller = Controller(robot, Policy())
        controller.calibrate(robot.neutral_reference(), robot.state(0), True, 0)
        controller.arm(robot.state(0), 0)
        output = []
        for i in range(10):
            t = i * 0.02
            reference = robot.neutral_reference(t)
            if i >= 7:
                q, v = reference.joint_position.copy(), np.zeros(22)
                q[2] += future
                v[2] = future * 10
                reference = replace(reference, joint_position=q, joint_velocity=v)
            if i:
                controller.set_reference(reference)
            command = controller.tick(robot.state(t), t)
            output.append(np.r_[command.targets, command.velocities])
            step_command(robot, command, controller.action_settings)
        return np.array(output)

    a, b = run(0.1), run(-0.1)
    np.testing.assert_array_equal(a[:7], b[:7])
    assert not np.allclose(a[7:], b[7:])


def test_motion_fidelity_rejects_stationary_and_slow_base_with_matching_joint_poses():
    from k1_motion.control_validation import motion_fidelity

    robot = K1Model()
    refs = []
    for i in range(101):
        ref = robot.neutral_reference(i * 0.02)
        position = ref.root_position.copy()
        position[0] += i * 0.01
        refs.append(replace(ref, root_position=position, root_velocity=np.array([0.5, 0, 0, 0, 0, 0])))
    clip = MotionClip.from_references(refs, {"model_signature": robot.signature})
    qpos = np.stack([np.r_[r.root_position, r.root_orientation, r.joint_position] for r in refs[1:]])
    qvel = np.zeros((100, 28))
    qvel[:, 0] = 0.5
    assert motion_fidelity(clip, qpos, qvel)["motion_fidelity_passed"]
    for fraction in (0.0, 0.25):
        wrong = qpos.copy()
        wrong[:, 0] *= fraction
        wrong_velocity = qvel * fraction
        result = motion_fidelity(clip, wrong, wrong_velocity)
        assert not result["motion_fidelity_passed"]
        assert result["root_progress_ratio"] == pytest.approx(fraction)
