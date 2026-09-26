"""Failure cases: ablations discard guards; dead channels perturb PPO/commands;
command history leaks across reset or differs between training and replay;
new observations alter transferred actor outputs; queue changes multiple axes.
Artifacts are real PPO checkpoints, exported actors and native replay traces.
"""

from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from k1_motion.robot import K1Model
from test_training import make_library

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def settings():
    return {
        **json.loads(
            (
                Path(__file__).resolve().parents[1] / "configs/controller-pv-official80-guard03-speed-v1.json"
            ).read_text()
        ),
        "mask_inactive_actions": True,
        "ankle_prior_scale": 0.5,
    }


def test_ablation_preserves_settings_and_rebinds_actuator_contract():
    from ablate_action_settings import ablation_metadata
    from k1_motion.actuators import actuator_contract

    robot = K1Model()
    parent = dict(
        action_settings=settings(), sha256="parent", actuator_contract=actuator_contract(robot, settings())
    )
    result = ablation_metadata(robot, parent, {"residual_scale": 0.4}, "actor.pt", "source")
    assert result["action_settings"] == {**settings(), "residual_scale": 0.4}
    assert result["actuator_contract"] == actuator_contract(robot, result["action_settings"])
    assert parent["action_settings"]["residual_scale"] == 0.25
    assert result["control_ablation"]["changed_settings"] == ["residual_scale"]
    assert not result["behaviorally_accepted"]


def test_disabled_channels_cannot_change_statistics_or_gradients():
    from k1_motion.actuation import active_action_mask, action_statistics
    from k1_motion.action_chunks import masked_statistics

    mask = active_action_mask(settings())
    assert mask == [0.0] * 10 + [1.0] * 12
    mean = torch.zeros(3, 4, 22, requires_grad=True)
    dist = torch.distributions.Normal(mean, torch.ones_like(mean))
    action = torch.randn_like(mean)
    a, e = action_statistics(dist, action, torch.tensor(mask))
    changed = action.clone()
    changed[..., :10] = 100
    b, f = action_statistics(dist, changed, torch.tensor(mask))
    torch.testing.assert_close(a, b)
    torch.testing.assert_close(e, f)
    a.sum().backward()
    assert torch.count_nonzero(mean.grad[..., :10]) == 0
    lengths = torch.tensor([1, 2, 4])
    c, g = masked_statistics(dist, action, lengths, torch.tensor(mask))
    d, h = masked_statistics(dist, changed, lengths, torch.tensor(mask))
    torch.testing.assert_close(c, d)
    torch.testing.assert_close(g, h)
    assert active_action_mask({"upper_body_residual_scale": 0}) == [1.0] * 22


@pytest.mark.parametrize("chunk", [1, 2])
def test_native_ppo_export_replay_resume_and_command_feedback(tmp_path, chunk):
    from k1_motion.learning import TrainConfig, train, Policy
    from k1_motion.tracking_env import TrackerEnv
    from k1_motion.training_validation import checkpoint_environment_settings
    from k1_motion.export import export_checkpoint
    from k1_motion.contracts import MotionClip
    from k1_motion.control_validation import replay_clip
    from k1_motion.runtime import Controller

    torch.set_num_threads(1)
    robot, library = make_library(tmp_path)
    options = dict(
        history=3,
        observation_profile="preview",
        command_feedback=True,
        action_settings=settings(),
        physics_options={"workers": 2},
    )
    cfg = TrainConfig(
        stage="student",
        iterations=2,
        horizon=4,
        epochs=1,
        minibatch=8,
        hidden_sizes=(16, 8),
        action_chunk_size=chunk,
        bc_weight=0,
        evaluation_interval=0,
        checkpoint_interval=1,
        milestone_interval=2,
    )
    env = TrackerEnv(library, 2, "cpu", "mujoco_cpp", **options)
    try:
        report = train(env, tmp_path / "train", cfg)
        assert report["finite_updates"] and report["checkpoint_reload_max_error"] == 0
        metrics = report["last_metrics"]["command_metrics"]
        assert "command/leg_action_bound_fraction" in metrics
        assert "command/guard_delta_rms_rad" in metrics
        checkpoint = tmp_path / "train/checkpoint.pt"
        saved = torch.load(checkpoint, weights_only=True)
        opts = checkpoint_environment_settings(saved)
        assert opts["command_feedback"] is True
        export_checkpoint(checkpoint, tmp_path / "export/actor.pt")
        policy = Policy(tmp_path / "export/actor.pt", robot.signature)
        action = policy(np.zeros(saved["actor_size"], dtype=np.float32))
        assert np.count_nonzero(action[..., :10]) == 0
        result = replay_clip(
            robot,
            policy,
            MotionClip.load(library / "standing.npz"),
            tmp_path / "replay.npz",
            authority_trace=True,
        )
        assert result["ticks"] > 0 if "ticks" in result else result["simulated_s"] > 0
        trace = np.load(tmp_path / "replay.npz")
        assert "command_raw_target" in trace.files
        assert "substep_requested_torque" in trace.files
        assert len(trace["substep_requested_torque"]) == 10 * len(trace["command_raw_target"])
        env.reset(clips=torch.zeros(2, dtype=torch.long), frames=torch.zeros(2, dtype=torch.long))
        assert torch.count_nonzero(env.previous_velocity) == 0
        assert torch.count_nonzero(env.previous_guard_delta) == 0
        action = torch.zeros(2, 22)
        action[0, :10] = 1
        env.step(action, auto_reset=False)
        torch.testing.assert_close(env.previous_action[0], env.previous_action[1])
        controller = Controller(robot, policy)
        assert controller.builder.command_feedback
    finally:
        env.close()
    restored = TrackerEnv(library, 2, "cpu", "mujoco_cpp", **opts, physics_options={"workers": 2})
    try:
        train(restored, tmp_path / "resume", replace(cfg, iterations=1), resume_checkpoint=checkpoint)
        restored.action_settings["mask_inactive_actions"] = False
        with pytest.raises(ValueError, match="action settings"):
            train(restored, tmp_path / "bad", cfg, resume_checkpoint=checkpoint)
    finally:
        restored.close()


def test_command_observation_extension_preserves_initialized_actor():
    from k1_motion.learning import ActorCritic
    from k1_motion.observations import observation_contract
    from k1_motion.model_transfer import initialize_model

    old = observation_contract(3, "preview")
    new = observation_contract(3, "preview", command_feedback=True)
    source = ActorCritic(old["size"], old["size"] + 140, (16, 8))
    target = ActorCritic(new["size"], new["size"] + 140, (16, 8))
    report = initialize_model(
        target,
        dict(
            model=source.state_dict(), actor_size=old["size"], critic_size=old["size"] + 140, observation=old
        ),
        new,
    )
    assert report["max_output_difference"]["actor_float64_accumulation"] < 2e-5


def test_queue_axes_are_matched_and_keep_all_guards(tmp_path):
    from prepare_authority_campaign import controller_variants, build_runs

    variants = controller_variants(settings())
    assert len(variants) == 8
    by_name = {r["name"]: r for r in variants}
    assert by_name["legacy_baseline"]["settings"]["mask_inactive_actions"] is False
    assert by_name["masked_baseline"]["settings"]["ankle_prior_scale"] == 1.0
    assert by_name["combined"]["settings"]["residual_scale"] == 0.4
    assert by_name["combined"]["settings"]["target_velocity_scale"] == 1.0
    assert by_name["prior_off"]["settings"]["ankle_prior_scale"] == 0.0
    assert by_name["command_feedback"]["command_feedback"] is True
    runs = build_runs(
        tmp_path,
        variants,
        library=tmp_path / "library",
        cache=tmp_path / "cache.pt",
        initializer=tmp_path / "init.pt",
        updates=2000,
        milestone=250,
        num_envs=2048,
    )
    for run in runs:
        assert run["settings"]["operating_speed_guard"]
        assert run["settings"]["command_position_margin_rad"] == 0.03
        command = run["command"]
        assert command[command.index("--iterations") + 1] == "2000"
        assert "--initialize" in command and "--resume" not in command
        assert command[command.index("--reward-profile") + 1] == "survival-position-v2"
        assert command[command.index("--backend") + 1] == "mujoco_cpp"


def test_native_commands_and_observations_match_exported_runtime(tmp_path):
    import mujoco
    from k1_motion.tracking_env import TrackerEnv
    from k1_motion.runtime import Controller
    from k1_motion.observations import observation_contract
    from k1_motion.servo import step_command

    robot, library = make_library(tmp_path)
    torch.set_num_threads(1)
    env = TrackerEnv(
        library, 1, "cpu", "mujoco", history=3, action_settings=settings(), command_feedback=True
    )
    zero = torch.zeros(1, dtype=torch.long)
    env.reset(clips=zero, frames=zero)

    class Policy:
        metadata = {
            "action_settings": settings(),
            "observation": observation_contract(3, command_feedback=True),
        }

        def __call__(self, obs):
            self.observation = obs.copy()
            return np.linspace(-1, 1, 22, dtype=np.float32)

    # Explicit contract required by the physical actuator profile.
    from k1_motion.actuators import actuator_contract

    Policy.metadata["actuator_contract"] = actuator_contract(robot, settings())
    policy = Policy()
    controller = Controller(robot, policy)
    robot.reset()
    mujoco.mj_forward(robot.model, robot.data)
    controller.calibrate(robot.neutral_reference(), robot.state(0), True, 0)
    controller.arm(robot.state(0), 0)
    try:
        for i in range(8):
            now = i * 0.02
            if i:
                controller.set_reference(robot.neutral_reference(now))
            # Reset/step returns observe(), so reset its history for aligned comparison.
            env.history.reset()
            controller.builder.reset()
            obs, _ = env.observe()
            command = controller.tick(robot.state(now), now)
            np.testing.assert_allclose(obs[0].numpy(), policy.observation, atol=3e-6)
            env.step(torch.linspace(-1, 1, 22)[None], auto_reset=False)
            step_command(robot, command, controller.action_settings)
            mujoco.mj_forward(robot.model, robot.data)
            for key in controller.last_command:
                np.testing.assert_allclose(
                    env.last_command[key][0].numpy(), controller.last_command[key], atol=3e-6
                )
            np.testing.assert_allclose(env.physics.robots[0].data.qpos, robot.data.qpos, atol=3e-6)
        controller.pause(0.17)
        assert not controller.previous_velocity.any() and not controller.previous_guard_delta.any()
    finally:
        env.close()


def test_ankle_prior_scale_changes_only_the_prior():
    from k1_motion.observations import targets_tensor, reference_tensor

    robot = K1Model()
    ref = reference_tensor(robot.neutral_reference())
    q = torch.tensor([[np.cos(0.05), 0, np.sin(0.05), 0]], dtype=torch.float32)
    args = (torch.zeros(1, 22), ref, q, torch.zeros(1, 3), torch.tensor(robot.limits))
    full = targets_tensor(*args)
    off = targets_tensor(*args, ankle_prior_scale=0)
    half = targets_tensor(*args, ankle_prior_scale=0.5)
    torch.testing.assert_close(half - off, (full - off) * 0.5)
    assert torch.count_nonzero(full - off) == 2
    torch.testing.assert_close(off, ref["joint_position"].to(off.dtype))


def test_resume_in_place_retains_metrics_and_cumulative_budget(tmp_path):
    from k1_motion.learning import TrainConfig, train
    from k1_motion.tracking_env import TrackerEnv

    _, library = make_library(tmp_path)
    torch.set_num_threads(1)
    config = TrainConfig(
        bc_weight=0,
        stage="student",
        iterations=1,
        horizon=4,
        epochs=1,
        minibatch=8,
        evaluation_interval=0,
        hidden_sizes=(16, 8),
        checkpoint_interval=1,
    )
    env = TrackerEnv(library, 2, "cpu", "mujoco_cpp", physics_options={"workers": 2})
    try:
        train(env, tmp_path / "training", config)
        checkpoint = tmp_path / "training/checkpoint.pt"
        resumed = train(env, tmp_path / "training", config, resume_checkpoint=checkpoint)
        metrics = [json.loads(x) for x in (tmp_path / "training/metrics.jsonl").read_text().splitlines()]
        assert [r["iteration"] for r in metrics] == [1, 2]
        assert resumed["transitions"] == 16
    finally:
        env.close()
