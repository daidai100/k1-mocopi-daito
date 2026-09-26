import json

import pytest
import torch

from k1_motion.observations import quat_apply, quat_mul
from k1_motion.tracking_rewards import angular_velocity, beyondmimic_causal_tracking
from k1_motion.tracking_env import TrackerEnv
from test_training import make_library


def test_backward_world_angular_velocity_sign_and_identity():
    identity = torch.tensor([[1., 0, 0, 0]])
    angle = torch.tensor(0.04)
    rotated = torch.tensor([[torch.cos(angle/2), 0, 0, torch.sin(angle/2)]])
    expected = torch.tensor([[0., 0, 2.]])
    torch.testing.assert_close(angular_velocity(rotated, identity, .02), expected)
    torch.testing.assert_close(angular_velocity(-rotated, identity, .02), expected)
    torch.testing.assert_close(angular_velocity(identity, identity, .02), torch.zeros_like(expected))


def test_tracking_peak_translation_and_yaw_alignment(tmp_path):
    _, path = make_library(tmp_path)
    env = TrackerEnv(path, 2, "cpu", reference_storage="packed")
    ids = torch.zeros(2, dtype=torch.long)
    env.reset(clips=ids, frames=ids)
    reference = env.library.frames(ids, ids)
    state = {"position": reference["root_position"].clone(),
             "orientation": reference["root_orientation"].clone(),
             "landmarks": reference["landmarks"].clone(),
             "body_orientation": reference["body_orientation"].clone()}
    previous = state["body_orientation"].clone()
    velocity = reference["landmark_velocity"]
    tracking, components = beyondmimic_causal_tracking(state, reference, previous, previous, velocity, .02)
    torch.testing.assert_close(tracking, torch.full((2,), 7.5), atol=1e-6, rtol=0)
    assert all(torch.allclose(x, torch.ones(2)) for x in components.values())
    state["position"][:, 0] += 5
    state["landmarks"][:, :, 0] += 5
    shifted, _ = beyondmimic_causal_tracking(state, reference, previous, previous, velocity, .02)
    torch.testing.assert_close(shifted, tracking)
    yaw = torch.tensor([0.70710678, 0., 0., 0.70710678]).expand(2, 4)
    state["landmarks"] = state["position"][:, None]+quat_apply(
        yaw[:, None].expand(2, 17, 4), reference["landmarks"]-reference["root_position"][:, None])
    state["orientation"] = quat_mul(yaw, state["orientation"])
    state["body_orientation"] = quat_mul(yaw[:, None].expand(2, 17, 4), previous)
    _, components = beyondmimic_causal_tracking(state, reference, state["body_orientation"], previous, velocity, .02)
    torch.testing.assert_close(components["relative_body_position"], torch.ones(2))
    torch.testing.assert_close(components["relative_body_orientation"], torch.ones(2))
    assert (components["anchor_orientation"] < 0.01).all()
    env.close()


def test_profile_preserves_causal_actor_and_uses_no_future_reward(tmp_path):
    _, path = make_library(tmp_path)
    outputs = []
    for profile, perturb_future in (("legacy", False), ("beyondmimic-causal-v1", False),
                                     ("beyondmimic-causal-v1", True)):
        env = TrackerEnv(path, 2, "cpu", history=10, reference_storage="packed", reward_profile=profile)
        ids = torch.zeros(2, dtype=torch.long)
        causal, _ = env.reset(clips=ids, frames=ids)
        if perturb_future:
            for name in ("joint_position", "body_orientation", "landmarks", "landmark_velocity"):
                env.library.values[name][2:] += .3
        next_causal, _, reward, done, info = env.step(torch.zeros(2, 22), auto_reset=False)
        outputs.append((causal, next_causal, reward, done, info["reward_components"]))
        env.close()
    for item in (0, 1, 3):
        torch.testing.assert_close(outputs[0][item], outputs[1][item], rtol=0, atol=0)
    for item in (0, 1, 2, 3):
        torch.testing.assert_close(outputs[1][item], outputs[2][item], rtol=0, atol=0)
    expected_components = {
        "anchor_height", "anchor_orientation", "relative_body_position",
        "relative_body_orientation", "global_body_linear_velocity",
        "global_body_angular_velocity", "diagnostic/world_body_rmse_m",
    }
    assert set(outputs[1][4]) == set(outputs[2][4]) == expected_components
    for name in expected_components:
        torch.testing.assert_close(outputs[1][4][name], outputs[2][4][name], rtol=0, atol=0)
    with pytest.raises(ValueError, match="only to the legacy"):
        TrackerEnv(path, reward_profile="beyondmimic-causal-v1", root_velocity_weight=2.)


def test_reference_cache_equivalence_independent_sampler_and_contract(tmp_path):
    from k1_motion.learning import MotionLibrary
    from k1_motion.reference_cache import build_reference_cache, load_reference_cache
    spec, directory = make_library(tmp_path)
    path = tmp_path/"cache.pt"
    report = build_reference_cache(directory, path, spec)
    fresh = MotionLibrary(directory, spec, "cpu", storage="packed")
    loaded = load_reference_cache(path, directory, spec, "cpu")
    assert report["clips"] == 1 and fresh.fingerprint == loaded.fingerprint
    for key in fresh.values:
        torch.testing.assert_close(fresh.values[key], loaded.values[key], rtol=0, atol=0)
    loaded.episode_count[:] = 3
    second = load_reference_cache(path, directory, spec, "cpu")
    assert second.episode_count.sum() == 0
    with pytest.raises(ValueError, match="contract differs"):
        load_reference_cache(path, directory, spec, "cpu", storage="padded")
    manifest = directory/"index.jsonl"
    row = json.loads(manifest.read_text())
    row["training_eligible"] = False
    manifest.write_text(json.dumps(row)+"\n")
    with pytest.raises(ValueError, match="contract differs"):
        load_reference_cache(path, directory, spec, "cpu")


def test_beyondmimic_cpp_train_reload_and_resume_contract(tmp_path):
    from k1_motion.learning import TrainConfig, train
    _, path = make_library(tmp_path)
    config = TrainConfig(stage="student", iterations=1, horizon=4, epochs=1,
                         minibatch=8, evaluation_interval=0, bc_weight=0, hidden_sizes=(32, 16))
    original = TrackerEnv(path, 2, "cpu", "mujoco_cpp", reward_profile="beyondmimic-causal-v1",
                          physics_options={"workers": 2}, reference_storage="packed", self_collision_weight=1.)
    report = train(original, tmp_path/"original", config)
    assert report["finite_updates"] and report["checkpoint_reload_max_error"] == 0
    assert report["reward_settings"]["profile"] == "beyondmimic-causal-v1"
    original.close()
    changed = TrackerEnv(path, 2, "cpu", "mujoco_cpp", physics_options={"workers": 2},
                         reference_storage="packed", self_collision_weight=1.)
    with pytest.raises(ValueError, match="reward settings"):
        train(changed, tmp_path/"changed", config, resume_checkpoint=tmp_path/"original/checkpoint.pt")
    changed.close()


def test_transfer_rejects_real_changes_with_high_precision_check(monkeypatch):
    from k1_motion.learning import ActorCritic
    import k1_motion.model_transfer as transfer
    from k1_motion.observations import observation_contract
    torch.set_num_threads(1)
    original = ActorCritic(544, 544, (32, 16))
    checkpoint = {"model": original.state_dict(), "actor_size": 544, "critic_size": 544,
                  "observation": observation_contract(4)}
    widened = ActorCritic(544, 544, (64, 32))
    result = transfer.initialize_model(widened, checkpoint, observation_contract(4))
    assert result["max_output_difference"]["critic_float64_accumulation"] < 2e-5
    widen = transfer._widen_network

    def broken(destination, source, mapping):
        widen(destination, source, mapping)
        destination[-1].bias.add_(0.001)

    monkeypatch.setattr(transfer, "_widen_network", broken)
    with pytest.raises(ValueError, match="changed the trained function"):
        transfer.initialize_model(widened, checkpoint, observation_contract(4))


@pytest.mark.parametrize("profile,exponentials", [("legacy", 8), ("beyondmimic-causal-v1", 6)])
def test_only_selected_reward_is_calculated(tmp_path, profile, exponentials):
    _, path = make_library(tmp_path)
    env = TrackerEnv(path, 2, "cpu", reward_profile=profile, reference_storage="packed")
    try:
        zeros = torch.zeros(2, dtype=torch.long)
        env.reset(clips=zeros, frames=zeros)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile_result:
            env.step(torch.zeros(2, 22), auto_reset=False)
        assert sum(item.count for item in profile_result.key_averages() if item.key == "aten::exp") == exponentials
    finally:
        env.close()


def test_reference_cache_rebind_requires_identical_preprocessing(tmp_path):
    import shutil
    from pathlib import Path
    import k1_motion.reference_cache as cache
    spec, directory = make_library(tmp_path)
    original = tmp_path/"original.pt"
    cache.build_reference_cache(directory, original, spec)
    package = Path(cache.__file__).parent
    receipt = cache.rebind_reference_cache(original, tmp_path/"rebound.pt", package, directory, spec)
    before = cache.load_reference_cache(original, directory, spec, "cpu")
    after = cache.load_reference_cache(tmp_path/"rebound.pt", directory, spec, "cpu")
    assert receipt["tensor_transformations"] == 0
    for key in before.values:
        torch.testing.assert_close(before.values[key], after.values[key], atol=0, rtol=0)
    baseline = tmp_path/"old-package"
    baseline.mkdir()
    for name in cache.PREPROCESSING_FILES:
        shutil.copy2(package/name, baseline/name)
    source = (baseline/"learning.py").read_text()
    assert cache._preprocessing_ast(source.replace("indices = self.offsets[clips] + frames",
                                                 "indices = self.offsets[clips] + frames + 0")) == cache._preprocessing_ast(source)
    changed = source.replace('self.sampling_mode = "episode_balanced"', 'self.sampling_mode = "changed"')
    (baseline/"learning.py").write_text(changed)
    with pytest.raises(ValueError, match="preprocessing changed"):
        cache.rebind_reference_cache(original, tmp_path/"bad.pt", baseline, directory, spec)
