import json

import numpy as np
import pytest
import torch
import mujoco

from k1_motion.contracts import MotionClip
from k1_motion.learning import Policy, TrainConfig, train
from k1_motion.observations import ObservationBuilder
from k1_motion.robot import K1Model
from k1_motion.ros_replay import LEGACY_NAMES, decode_joint_command
from k1_motion.tracking_env import TrackerEnv
from k1_motion.runtime import Controller


def make_library(tmp_path):
    robot = K1Model()
    directory = tmp_path / "library"
    directory.mkdir()
    refs = [robot.neutral_reference(i * 0.02) for i in range(50)]
    metadata = {
        "capture_group": "synthetic/standing",
        "family": "stance",
        "model_signature": robot.signature,
        "source_kind": "synthetic_debug",
        "split": "train",
        "kinematics_accepted": True,
        "reference_path": "standing.npz",
    }
    MotionClip.from_references(refs, metadata).save(directory / "standing.npz")
    (directory / "index.jsonl").write_text(json.dumps(metadata) + "\n")
    return robot, directory


def test_cpp_backend_trains_and_reloads(tmp_path):
    _, directory = make_library(tmp_path)
    torch.set_num_threads(1)
    env = TrackerEnv(directory, 4, "cpu", "mujoco_cpp", reference_storage="packed",
                     physics_options={"workers": 2, "chunk_size": 1}, self_collision_weight=1.)
    try:
        config = TrainConfig(iterations=2, horizon=4, epochs=1, minibatch=16,
                             evaluation_interval=0, hidden_sizes=(32, 16))
        report = train(env, tmp_path / "cpp-training", config)
        assert report["finite_updates"]
        assert report["checkpoint_reload_max_error"] == 0
        assert report["backend"] == "mujoco_cpp"
        assert report["physics_contract"]["actual_workers"] == 2
    finally:
        env.close()


def test_self_contact_cost_excludes_ground_and_preserves_physics(tmp_path):
    _, directory = make_library(tmp_path)
    rewards, states = [], []
    for weight in (0.0, 2.0):
        env = TrackerEnv(directory, num_envs=2, device="cpu", self_collision_weight=weight)
        env.reset(clips=torch.zeros(2, dtype=torch.long), frames=torch.zeros(2, dtype=torch.long))
        # World 0 has ordinary feet on ground. World 1 has a right arm/trunk
        # intersection above the floor, within the joint's declared limits.
        robot = env.physics.robots[1]
        robot.data.qpos[2] += 1
        robot.data.qpos[14] = 1.629
        mujoco.mj_forward(robot.model, robot.data)
        _, _, reward, _, _ = env.step(torch.zeros(2, 22), auto_reset=False)
        torch.testing.assert_close(env.physics.self_collision_per_env, torch.tensor([0.0, 1.0]))
        rewards.append(reward)
        states.append(env.physics.state()["q"])
        env.close()
    torch.testing.assert_close(states[0], states[1])
    torch.testing.assert_close(rewards[1], rewards[0] - torch.tensor([0.0, 0.04]))
    with pytest.raises(ValueError, match="MuJoCo backends"):
        TrackerEnv(directory, backend="isaac", self_collision_weight=1)


def test_resume_rejects_changed_collision_reward(tmp_path):
    _, directory = make_library(tmp_path)
    config = TrainConfig(iterations=1, horizon=4, epochs=1, minibatch=8, evaluation_interval=0)
    original = TrackerEnv(directory, num_envs=2, device="cpu")
    train(original, tmp_path / "original", config)
    changed = TrackerEnv(directory, num_envs=2, device="cpu", self_collision_weight=1)
    with pytest.raises(ValueError, match="reward settings"):
        train(changed, tmp_path / "changed", config, resume_checkpoint=tmp_path / "original/checkpoint.pt")
    original.close()
    changed.close()


def test_experimental_action_settings_match_runtime_and_replay(tmp_path):
    robot, directory = make_library(tmp_path)
    settings = {"residual_scale": 0.5, "command_velocity_limit": 10.0}
    env = TrackerEnv(directory, num_envs=1, device="cpu", action_settings=settings)
    zeros = torch.zeros(1, dtype=torch.long)
    env.reset(clips=zeros, frames=zeros)
    robot.reset()

    class ConstantPolicy:
        metadata = {"action_settings": settings}

        def __call__(self, _):
            return np.ones(22)

    controller = Controller(robot, ConstantPolicy())
    controller.calibrate(robot.neutral_reference(), robot.state(0), True, 0)
    controller.arm(robot.state(0), 0)
    command = controller.tick(robot.state(0), 0, True)
    env.step(torch.ones(1, 22), auto_reset=False)
    np.testing.assert_allclose(env.previous_target[0], command.targets, atol=1e-7)
    robot.step(command.targets)
    np.testing.assert_allclose(env.physics.robots[0].data.qpos, robot.data.qpos, atol=1e-7)
    assert np.isclose(command.targets[0] - robot.neutral[0], robot.velocity_limit[0] * 0.02)
    assert np.isclose(command.targets[10] - robot.neutral[10], 0.2)
    assert robot.config["command_velocity_limit"] == 6
    env.close()


def test_changed_action_contract_requires_initialization_and_explicit_refinement(tmp_path):
    from dataclasses import replace
    from k1_motion.export import export_checkpoint

    robot, directory = make_library(tmp_path)
    config = TrainConfig(iterations=1, horizon=4, epochs=1, minibatch=8, evaluation_interval=0)
    original = TrackerEnv(directory, num_envs=2, device="cpu")
    train(original, tmp_path / "teacher", config)
    checkpoint = tmp_path / "teacher/checkpoint.pt"
    settings = {"residual_scale": 0.5, "command_velocity_limit": 10.0}
    changed = TrackerEnv(directory, num_envs=2, device="cpu", action_settings=settings)
    with pytest.raises(ValueError, match="Resume action settings"):
        train(changed, tmp_path / "resume", config, resume_checkpoint=checkpoint)
    with pytest.raises(ValueError, match="Teacher action settings"):
        train(changed, tmp_path / "imitation", replace(config, stage="student"), checkpoint)
    report = train(changed, tmp_path / "refined", replace(config, stage="student", bc_weight=0), checkpoint)
    assert report["action_settings"] == settings
    saved = torch.load(tmp_path / "refined/checkpoint.pt", weights_only=True)
    assert saved["action_settings"] == settings and report["finite_updates"]
    export_checkpoint(tmp_path / "refined/checkpoint.pt", tmp_path / "export/actor.pt")
    policy = Policy(tmp_path / "export/actor.pt", robot.signature)
    assert Controller(robot, policy).action_settings == settings
    original.close()
    changed.close()


def test_shared_replay_library_still_checks_robot_contract(tmp_path):
    from k1_motion.learning import MotionLibrary

    spec, directory = make_library(tmp_path)
    library = MotionLibrary(directory, spec, "cpu")
    env = TrackerEnv(directory, num_envs=2, device="cpu", library=library)
    assert env.library is library
    env.close()
    library.rows[0]["model_signature"] = "different_robot"
    with pytest.raises(ValueError, match="robot contract"):
        TrackerEnv(directory, num_envs=2, device="cpu", library=library)


def test_training_scope_exclusion_cannot_hide_held_out_recordings(tmp_path):
    from k1_motion.learning import MotionLibrary

    spec, directory = make_library(tmp_path)
    index = directory / "index.jsonl"
    original = json.loads(index.read_text())
    excluded = {**original, "capture_group": "synthetic/excluded", "training_eligible": False}
    validation = {**excluded, "split": "validation", "capture_group": "synthetic/held-out"}
    index.write_text("".join(json.dumps(r) + "\n" for r in (original, excluded, validation)))
    training = MotionLibrary(directory, spec, "cpu")
    held_out = MotionLibrary(directory, spec, "cpu", "validation")
    assert len(training.rows) == len(held_out.rows) == 1
    assert training.rows[0]["capture_group"] == "synthetic/standing"
    assert held_out.rows[0]["capture_group"] == "synthetic/held-out"


def test_teacher_student_update_export_and_reload(tmp_path):
    torch.set_num_threads(1)
    robot, directory = make_library(tmp_path)
    env = TrackerEnv(directory, num_envs=2, device="cpu")
    teacher_path = tmp_path / "teacher"
    report = train(
        env, teacher_path, TrainConfig(iterations=1, horizon=4, epochs=1, minibatch=8, evaluation_interval=0)
    )
    assert report["finite_updates"] and report["checkpoint_reload_max_error"] == 0
    with pytest.raises(ValueError, match="causal student"):
        Policy(teacher_path / "actor.pt", robot.signature)
    student_path = tmp_path / "student"
    report = train(
        env,
        student_path,
        TrainConfig(stage="student", iterations=1, horizon=4, epochs=1, minibatch=8, evaluation_interval=0),
        teacher_path / "checkpoint.pt",
    )
    assert report["checkpoint_reload_max_error"] == 0
    policy = Policy(student_path / "actor.pt", robot.signature)
    obs = ObservationBuilder(robot.neutral).build(robot.state(0), robot.neutral_reference(), np.zeros(22), 0)
    result = policy(obs)
    assert result.shape == (22,) and np.isfinite(result).all() and np.max(abs(result)) <= 1
    with pytest.raises(ValueError, match="contract mismatch"):
        Policy(student_path / "actor.pt", "wrong_model")


def test_actor_cannot_observe_teacher_future(tmp_path):
    torch.set_num_threads(1)
    _, directory = make_library(tmp_path)
    env = TrackerEnv(directory, num_envs=2, device="cpu")
    env.reset()
    env.frames[:] = 0
    env.command_frames[:] = 0
    env.history.reset()
    a, pa = env.observe()
    env.library.values["joint_position"][:, 5:] += 0.3
    env.history.reset()
    b, pb = env.observe()
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert not torch.equal(pa, pb)


def test_wider_longer_history_preserves_policy_and_value():
    from k1_motion.learning import ActorCritic
    from k1_motion.model_transfer import initialize_model
    from k1_motion.observations import observation_contract

    torch.manual_seed(42)
    torch.set_num_threads(1)
    original = ActorCritic(684, 684)
    original.actor.normalizer.update(torch.randn(32, 684))
    original.critic_norm.update(torch.randn(32, 684))
    saved = {
        "model": original.state_dict(),
        "actor_size": 684,
        "critic_size": 684,
        "observation": observation_contract(4),
    }
    widened = ActorCritic(1500, 1500, (512, 256))
    transfer = initialize_model(widened, saved, observation_contract(10))
    obs = torch.randn(8, 1500)
    torch.testing.assert_close(original.actor(obs[:, 816:]), widened.actor(obs), atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(original.value(obs[:, 816:]), widened.value(obs), atol=2e-6, rtol=1e-5)
    obs[:, :816] *= 100
    torch.testing.assert_close(original.actor(obs[:, 816:]), widened.actor(obs), atol=2e-6, rtol=1e-5)
    assert transfer["new_history"] == 10
    # Symmetry is broken in outgoing weights while the initial function is kept.
    assert not torch.equal(widened.actor.network[2].weight[:, :256], widened.actor.network[2].weight[:, 256:])
    with pytest.raises(ValueError, match="causal observation contract"):
        initialize_model(widened, saved, {**observation_contract(10), "future_frames": 1})


def test_extended_history_student_deploys_with_matching_runtime(tmp_path):
    from k1_motion.runtime import Controller
    from k1_motion.export import export_checkpoint

    _, directory = make_library(tmp_path)
    env = TrackerEnv(directory, num_envs=2, device="cpu", history=10)
    config = TrainConfig(
        iterations=1, horizon=4, epochs=1, minibatch=8, evaluation_interval=0, hidden_sizes=(512, 256)
    )
    teacher = tmp_path / "teacher-long"
    train(env, teacher, config)
    config.stage = "student"
    student = tmp_path / "student-long"
    train(env, student, config, teacher / "checkpoint.pt")
    export_checkpoint(student / "checkpoint.pt", student / "reexport.pt")
    policy = Policy(student / "reexport.pt", env.spec.signature)
    controller = Controller(env.spec, policy)
    assert controller.builder.history.length == 10
    obs = controller.builder.build(env.spec.state(0), env.spec.neutral_reference(), np.zeros(22), 0)
    assert np.isfinite(policy(obs)).all()


def test_transition_sampling_balances_durations_and_keeps_every_clip(tmp_path):
    from k1_motion.learning import MotionLibrary

    robot, directory = make_library(tmp_path)
    metadata = json.loads((directory / "index.jsonl").read_text())
    MotionClip.from_references([robot.neutral_reference(i * 0.02) for i in range(100)], metadata).save(
        directory / "standing.npz"
    )
    short = {**metadata, "family": "walk", "capture_group": "synthetic/short", "reference_path": "short.npz"}
    MotionClip.from_references([robot.neutral_reference(i * 0.02) for i in range(40)], short).save(
        directory / "short.npz"
    )
    (directory / "index.jsonl").write_text(json.dumps(metadata) + "\n" + json.dumps(short) + "\n")
    library = MotionLibrary(directory, robot, "cpu")
    torch.manual_seed(37)
    library.configure_sampling("transition_balanced")
    selections = torch.bincount(library.sample(20000), minlength=2)
    exposure = selections * library.episode_duration_ema
    assert 0.9 < float(exposure[0] / exposure[1]) < 1.1
    library.episode_duration_ema[:] = torch.tensor([1.0, 10000.0])
    library.configure_sampling("transition_balanced")
    assert torch.bincount(library.sample(20000), minlength=2).min() > 500


def test_moving_reference_reset_retains_world_linear_and_body_angular_velocity(tmp_path):
    from k1_motion.math3d import rotation

    _, directory = make_library(tmp_path)
    env = TrackerEnv(directory, num_envs=2, device="cpu")
    env.library.values["root_velocity"][:] = torch.tensor([0.2, -0.1, 0.05, 0.3, 0.1, -0.2])
    env.library.values["joint_velocity"][:] = 0.12
    env.reset()
    for robot in env.physics.robots:
        np.testing.assert_allclose(robot.data.qvel[:3], [0.2, -0.1, 0.05], atol=1e-7)
        np.testing.assert_allclose(
            rotation(robot.data.qpos[3:7]).apply(robot.data.qvel[3:6]), [0.3, 0.1, -0.2], atol=1e-7
        )
        np.testing.assert_allclose(robot.data.qvel[6:], 0.12, atol=1e-7)


def test_continuation_keeps_optimizer_steps_and_rejects_changed_rollout_size(tmp_path):
    torch.set_num_threads(1)
    _, directory = make_library(tmp_path)
    env = TrackerEnv(directory, num_envs=2, device="cpu")
    config = TrainConfig(iterations=1, horizon=4, epochs=1, minibatch=8, evaluation_interval=0)
    first = tmp_path / "first"
    train(env, first, config)
    previous = torch.load(first / "checkpoint.pt", weights_only=True)
    previous["train_parents"].append("synthetic/earlier-library")
    torch.save(previous, first / "checkpoint.pt")
    second = tmp_path / "continued"
    result = train(env, second, config, resume_checkpoint=first / "checkpoint.pt")
    saved = torch.load(second / "checkpoint.pt", weights_only=True)
    assert saved["iteration"] == 2 and result["transitions"] == 16
    assert "synthetic/earlier-library" in saved["train_parents"]
    assert all(float(v["step"]) == 2 for v in saved["optimizer"]["state"].values())
    changed = TrackerEnv(directory, num_envs=3, device="cpu")
    with pytest.raises(ValueError, match="same rollout size"):
        train(changed, tmp_path / "bad-rollout", config, resume_checkpoint=first / "checkpoint.pt")
    initialized = tmp_path / "initialized"
    result = train(changed, initialized, config, initialize_checkpoint=first / "checkpoint.pt")
    assert result["initialization_iteration"] == 1 and result["transitions"] == 12


def test_explicit_environment_resize_preserves_optimizer_and_exposure(tmp_path):
    from dataclasses import replace
    torch.set_num_threads(1)
    _, directory = make_library(tmp_path)
    config = TrainConfig(iterations=1, horizon=4, epochs=1, minibatch=8, evaluation_interval=0,
                         hidden_sizes=(32, 16), milestone_interval=1)
    first = TrackerEnv(directory, num_envs=2, device="cpu")
    train(first, tmp_path/"first", config)
    first.close()
    changed = TrackerEnv(directory, num_envs=3, device="cpu")
    try:
        with pytest.raises(ValueError, match="same rollout horizon"):
            train(changed, tmp_path/"bad-horizon", replace(config, horizon=5, allow_env_resize=True),
                  resume_checkpoint=tmp_path/"first/checkpoint.pt")
        result = train(changed, tmp_path/"resized", replace(config, allow_env_resize=True),
                       resume_checkpoint=tmp_path/"first/checkpoint.pt")
        saved = torch.load(tmp_path/"resized/checkpoint.pt", weights_only=True)
        assert saved["iteration"] == 2 and saved["transitions"] == 8+12
        assert saved["optimizer_steps"] == 3
        assert result["start_transitions"] == 8 and result["target_transitions"] == 20
        assert saved["rollout_history"] == [
            {"start_iteration": 0, "start_transitions": 0, "num_envs": 2, "horizon": 4},
            {"start_iteration": 1, "start_transitions": 8, "num_envs": 3, "horizon": 4},
        ]
        assert all(float(v["step"]) == 3 for v in saved["optimizer"]["state"].values())
        assert (tmp_path/"resized/checkpoint-000002.pt").exists()
        again = train(changed, tmp_path/"again", config, resume_checkpoint=tmp_path/"resized/checkpoint.pt")
        assert again["transitions"] == 32 and again["last_metrics"]["iteration"] == 3
    finally:
        changed.close()


def test_walltime_budget_saves_final_checkpoint_and_reports_actual_updates(tmp_path):
    _, directory = make_library(tmp_path)
    env = TrackerEnv(directory, num_envs=2, device="cpu")
    try:
        config = TrainConfig(iterations=100, horizon=4, epochs=1, minibatch=8, evaluation_interval=0,
                             hidden_sizes=(32, 16), max_seconds=1e-12)
        result = train(env, tmp_path/"bounded", config)
        assert result["stop_reason"] == "walltime_budget" and result["iterations"] == 1
        assert result["transitions"] == 8 and result["optimizer_steps"] == 1
        checkpoint = torch.load(tmp_path/"bounded/checkpoint.pt", weights_only=True)
        assert checkpoint["iteration"] == 1 and checkpoint["optimizer_steps"] == 1
        assert result["checkpoint_reload_max_error"] == 0
    finally:
        env.close()


def test_archive_joint_names_reorder_instead_of_assuming_indices():
    q = np.arange(22) * 0.01
    actual = decode_joint_command(list(reversed(LEGACY_NAMES)), q[::-1])
    np.testing.assert_array_equal(actual, q)
    with pytest.raises(ValueError):
        decode_joint_command(list(LEGACY_NAMES[:-1]), q[:-1])


def test_training_and_evaluation_use_the_same_sole_geometry_and_material():
    import xml.etree.ElementTree as ET
    from k1_motion.robot import ROOT
    from k1_motion.simulator_assets import shared_collision_urdf

    robot = K1Model()
    urdf = ET.parse(shared_collision_urdf(ROOT, robot.model_path))
    for side in ("left", "right"):
        collision = urdf.find(f".//link[@name='{side}_ankle_roll_link']/collision")
        np.testing.assert_allclose(
            np.fromstring(collision.find("geometry/box").get("size"), sep=" "), [0.18, 0.07, 0.036]
        )
        np.testing.assert_allclose(
            np.fromstring(collision.find("origin").get("xyz"), sep=" "), [0.026, 0, -0.02]
        )
    np.testing.assert_allclose(robot.model.geom_friction[:, 0], robot.config["ground_friction"])
