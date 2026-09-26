"""Failures first: missing travel errors, frame leakage, repeated event inflation,
reset leakage, shifted exposure, reward changes, and incompatible resumes.
"""

import copy
import json
from types import SimpleNamespace

import pytest
import torch

from k1_motion.training_curriculum import TrainingCurriculum
from test_minimal_casual_curriculum import rows_and_features
from test_training import make_library


def contracts(tmp_path):
    from prepare_minimal_casual_curriculum import build_contract
    from k1_motion.training_curriculum import fidelity_curriculum_contract

    rows, features = rows_and_features()
    old = build_contract(rows, features)
    new = fidelity_curriculum_contract(old)
    a, b = tmp_path / "v1.json", tmp_path / "v2.json"
    a.write_text(json.dumps(old))
    b.write_text(json.dumps(new))
    library = SimpleNamespace(
        rows=rows,
        device="cpu",
        lengths=torch.full((7,), 201),
        take_weights=torch.ones(7),
        episode_duration_ema=torch.full((7,), 50.0),
    )
    return library, a, b, old, new


def signals(n=2, velocity=0.0, world=0.0, collision=0.0, position=0.0, speed=0.0):
    return dict(
        root_velocity_error_squared=torch.full((n,), velocity**2),
        world_position_error_squared=torch.full((n,), world**2),
        collision=torch.full((n,), collision),
        joint_limit_fraction=torch.full((n,), position),
        operating_speed_fraction=torch.full((n,), speed),
    )


def test_contract_preserves_originals_weights_and_rejects_unversioned_thresholds(tmp_path):
    library, a, b, old, new = contracts(tmp_path)
    assert old["version"] == "minimal-casual-curriculum-v1"
    assert new["version"] == "minimal-casual-curriculum-v2"
    for key in ["train_ids", "locomotion_ids", "target_transition_weights", "reset_mix", "sampling"]:
        assert old[key] == new[key]
    legacy = TrainingCurriculum(library, a)
    expected = library.weights.clone()
    current = TrainingCurriculum(library, b)
    torch.testing.assert_close(library.weights, expected, atol=0, rtol=0)
    bad = copy.deepcopy(new)
    bad["failure_observation"]["window_steps"] = 0
    b.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="failure observation"):
        TrainingCurriculum(library, b)
    with pytest.raises(ValueError, match="Resume curriculum"):
        current.load_state_dict(legacy.state_dict())


def test_sustained_errors_enter_ledger_without_terminal_and_do_not_inflate(tmp_path):
    library, _, b, _, _ = contracts(tmp_path)
    sampler = TrainingCurriculum(library, b)
    clips = torch.tensor([0, 1])
    failed = torch.zeros(2, dtype=torch.bool)
    for tick in range(24):
        sampler.record(clips, torch.full((2,), 101 + tick), failed, signals=signals(velocity=0.4, world=0.2))
    assert sampler.failure_counts.sum() == 0
    sampler.record(clips, torch.full((2,), 125), failed, signals=signals(velocity=0.4, world=0.2))
    # Two signals in the same transition count once, in the preceding second.
    assert sampler.failure_counts[0, 1] == 1 and sampler.failure_counts[1, 1] == 1
    for tick in range(25, 70):
        sampler.record(
            clips, torch.full((2,), min(199, 101 + tick)), failed, signals=signals(velocity=0.4, world=0.2)
        )
    assert sampler.failure_counts.sum() == 2
    metrics = sampler.update()
    assert metrics["fidelity_events"]["root_velocity"] == 2
    assert metrics["fidelity_events"]["world_position"] == 2
    assert metrics["target_locomotion_transition_share"] == 0.5
    assert sampler.failure_counts.max() <= 1


def test_safety_events_are_immediate_and_reset_only_selected_environment(tmp_path):
    library, _, b, _, _ = contracts(tmp_path)
    sampler = TrainingCurriculum(library, b)
    clips = torch.tensor([0, 1])
    frames = torch.tensor([100, 100])
    failed = torch.zeros(2, dtype=torch.bool)
    sampler.record(clips, frames, failed, signals=signals(collision=1, position=0.001, speed=0.001))
    assert sampler.failure_counts.sum() == 2
    sampler.record(clips, frames, failed, signals=signals(collision=1, position=0.001, speed=0.001))
    assert sampler.failure_counts.sum() == 2
    sampler.reset_environments(torch.tensor([0]))
    sampler.record(clips, frames, failed, signals=signals(collision=1, position=0.001, speed=0.001))
    assert sampler.failure_counts[0, 1] == 2 and sampler.failure_counts[1, 1] == 1
    sampler.reset_environments(torch.tensor([0, 1]))
    for tick in range(24):
        sampler.record(clips, frames, failed, signals=signals(velocity=0.4))
    before = sampler.failure_counts.clone()
    sampler.reset_environments(torch.tensor([0]))
    sampler.record(clips, frames, failed, signals=signals(velocity=0.4))
    assert sampler.failure_counts[0, 1] == before[0, 1]
    assert sampler.failure_counts[1, 1] == before[1, 1] + 1


def test_signal_contract_fails_closed_and_legacy_record_is_unchanged(tmp_path):
    library, a, b, _, _ = contracts(tmp_path)
    sampler = TrainingCurriculum(library, b)
    ids = torch.tensor([0, 1])
    failed = torch.tensor([False, True])
    with pytest.raises(ValueError, match="signals"):
        sampler.record(ids, ids, failed)
    with pytest.raises(ValueError, match="signals"):
        sampler.record(ids, ids, failed, signals={"collision": torch.zeros(2)})
    legacy = TrainingCurriculum(library, a)
    legacy.record(ids, ids, failed)
    assert legacy.failure_counts.sum() == 1


def test_native_fidelity_events_do_not_change_physics_reward_or_observation(tmp_path):
    from k1_motion.contracts import MotionClip
    from k1_motion.tracking_env import TrackerEnv
    from k1_motion.training_curriculum import fidelity_curriculum_contract
    from prepare_minimal_casual_curriculum import build_contract

    robot, library = make_library(tmp_path)
    original = json.loads((library / "index.jsonl").read_text())
    rows = []
    for key, family, name in [("w", "walk", "walk_forward_001"), ("g", "gesture", "wave_001")]:
        row = dict(
            original,
            id=key,
            family=family,
            take_name=name,
            capture_group="synthetic/" + name,
            reference_path=key + ".npz",
        )
        clip = MotionClip.load(library / "standing.npz")
        # Synthetic sustained world lag, with unchanged relative pose and safety.
        clip.values = {key: value.copy() for key, value in clip.values.items()}
        clip.values["root_position"][1:, 0] += 0.25
        clip.values["landmarks"][1:, :, 0] += 0.25
        clip.metadata.update(row)
        clip.save(library / row["reference_path"])
        rows.append(row)
    (library / "index.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    v1 = build_contract(
        rows, {"w": {"event_longest_s": {"forward_travel": 1.0}}, "g": {"event_longest_s": {}}}
    )
    manifests = []
    for name, contract in [("v1", v1), ("v2", fidelity_curriculum_contract(v1))]:
        p = tmp_path / (name + ".json")
        p.write_text(json.dumps(contract))
        manifests.append(p)
    torch.set_num_threads(1)
    envs = [
        TrackerEnv(
            library,
            2,
            "cpu",
            "mujoco_cpp",
            history=3,
            observation_profile="preview",
            reward_profile="world-body-v1",
            safety_profile="casual-safe-v1",
            physics_options={"workers": 2},
        )
        for _ in manifests
    ]
    try:
        for env, p in zip(envs, manifests):
            env.curriculum = TrainingCurriculum(env.library, p)
            env.reset(clips=torch.zeros(2, dtype=torch.long), frames=torch.zeros(2, dtype=torch.long))
        for _ in range(30):
            outputs = [env.step(torch.zeros(2, 22)) for env in envs]
            for k in range(4):
                torch.testing.assert_close(outputs[0][k], outputs[1][k], atol=0, rtol=0)
            for k in ["q", "dq", "position", "velocity"]:
                torch.testing.assert_close(
                    envs[0].physics.state()[k], envs[1].physics.state()[k], atol=0, rtol=0
                )
        assert envs[0].curriculum.failure_counts.sum() == 0
        assert envs[1].curriculum.failure_counts.sum() > 0
        assert envs[1].curriculum.update()["fidelity_events"]["world_position"] > 0
        # The new ledger persists through a real learner checkpoint, while its
        # episode-local window is rebuilt by the declared physical reset.
        from dataclasses import replace
        from k1_motion.learning import TrainConfig, train, Policy
        from k1_motion.export import export_checkpoint
        from k1_motion.control_validation import replay_clip

        cfg = TrainConfig(
            stage="student",
            iterations=2,
            horizon=32,
            epochs=1,
            minibatch=64,
            hidden_sizes=(32, 16),
            bc_weight=0,
            evaluation_interval=0,
            checkpoint_interval=1,
            curriculum_manifest=str(manifests[1]),
        )
        report = train(envs[1], tmp_path / "training", cfg)
        assert report["finite_updates"] and report["checkpoint_reload_max_error"] == 0
        saved = torch.load(tmp_path / "training/checkpoint.pt", map_location="cpu", weights_only=True)
        assert saved["curriculum_state"]["transitions"] == 128
        assert (
            report["last_metrics"]["curriculum"]["failure_observation"]["version"]
            == "train-fidelity-phase-v1"
        )
        export_checkpoint(tmp_path / "training/checkpoint.pt", tmp_path / "export/actor.pt")
        replay = replay_clip(
            robot, Policy(tmp_path / "export/actor.pt", robot.signature), MotionClip.load(library / "g.npz")
        )
        assert replay["resets_during_trial"] == 0
        resumed = train(
            envs[1],
            tmp_path / "resumed",
            replace(cfg, iterations=1),
            resume_checkpoint=tmp_path / "training/checkpoint.pt",
        )
        assert resumed["optimizer_steps"] > report["optimizer_steps"]
        with pytest.raises(ValueError, match="curriculum"):
            train(
                envs[1],
                tmp_path / "invalid",
                replace(cfg, iterations=1, curriculum_manifest=str(manifests[0])),
                resume_checkpoint=tmp_path / "training/checkpoint.pt",
            )
    finally:
        for env in envs:
            env.close()
