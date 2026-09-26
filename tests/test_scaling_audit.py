"""Audit risks: lost export scale, double scaling, unscaled validation targets,
wrong preview/safety factory settings, silent legacy actor metadata, changed resume.
Reproduce: PYTHONPATH=src .venv/bin/python -m pytest tests/test_scaling_audit.py
"""

import json
from dataclasses import replace
import numpy as np
import pytest
import torch
from test_training import make_library


def test_scaled_training_export_replay_and_validation(tmp_path):
    from k1_motion.tracking_env import TrackerEnv
    from k1_motion.learning import TrainConfig, train, Policy
    from k1_motion.export import export_checkpoint
    from k1_motion.reference_scale import scale_clip
    from k1_motion.contracts import MotionClip
    from k1_motion.control_validation import replay_clip
    from k1_motion.training_validation import replay_panel

    robot, directory = make_library(tmp_path)
    clip = MotionClip.load(directory / "standing.npz")
    shift = np.zeros_like(clip.values["root_position"])
    shift[:, 0] = np.arange(len(clip.times)) * 0.01
    shift[15:30, 2] = 0.15
    values = {k: v.copy() for k, v in clip.values.items()}
    values["root_position"] += shift
    values["landmarks"] += shift[:, None]
    clip = MotionClip(clip.times, values, {**clip.metadata, "family": "jump"})
    clip.save(directory / "standing.npz")
    row = json.loads((directory / "index.jsonl").read_text())
    row["family"] = "jump"
    (directory / "index.jsonl").write_text(json.dumps(row) + "\n")
    torch.set_num_threads(1)
    settings = dict(
        history=3,
        observation_profile="preview",
        preview_horizon_s=0.0,
        reward_profile="world-capped-root-v1",
        reference_scale=0.9,
        reference_storage="packed",
        safety_profile="casual-safe-v1",
        physics_options={"workers": 2},
    )
    env = TrackerEnv(directory, 2, "cpu", "mujoco_cpp", **settings)
    cfg = TrainConfig(
        stage="student",
        iterations=2,
        horizon=8,
        epochs=1,
        minibatch=16,
        hidden_sizes=(32, 16),
        evaluation_interval=0,
        checkpoint_interval=1,
        bc_weight=0,
    )
    try:
        train(env, tmp_path / "training", cfg)
        auto = json.loads((tmp_path / "training/actor.json").read_text())
        assert auto["reference_scale"] == env.reward_settings["reference_scale"]
        for treatment in ("missing", None, {**auto["reference_scale"], "scale": 0.85}):
            incomplete = {**auto, "reference_scale": treatment}
            if treatment == "missing":
                incomplete.pop("reference_scale")
            (tmp_path / "training/actor.json").write_text(json.dumps(incomplete))
            with pytest.raises(ValueError, match="re-export"):
                Policy(tmp_path / "training/actor.pt", robot.signature)
        (tmp_path / "training/actor.json").write_text(json.dumps(auto))
        export_checkpoint(tmp_path / "training/checkpoint.pt", tmp_path / "export/actor.pt")
        policy = Policy(tmp_path / "export/actor.pt", robot.signature)
        assert policy.metadata["reference_scale"] == auto["reference_scale"]
        original = MotionClip.load(directory / "standing.npz")
        scaled = scale_clip(original, robot, auto["reference_scale"])
        np.testing.assert_allclose(
            scaled.values["root_position"], env.library.values["root_position"].numpy(), atol=1e-7
        )
        with pytest.raises(ValueError, match="already"):
            scale_clip(scaled, robot, auto["reference_scale"])
        result = replay_clip(robot, policy, original, tmp_path / "trace.npz")
        assert result["reference_scale"] == auto["reference_scale"]
        assert result["preview"]["preview_horizon_s"] == 0
        # Same data but a distinct directory forces the held-out loader branch.
        other = tmp_path / "other"
        other.mkdir()
        rows = [json.loads(x) for x in (directory / "index.jsonl").read_text().splitlines()]
        for row in rows:
            row["reference_path"] = str((directory / row["reference_path"]).resolve())
        (other / "index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        from k1_motion.learning import ActorCritic

        saved = torch.load(tmp_path / "training/checkpoint.pt", weights_only=True)
        model = ActorCritic(saved["actor_size"], saved["critic_size"], tuple(saved["hidden_sizes"]))
        model.load_state_dict(saved["model"])
        model.eval()
        report = replay_panel(env, model, "student", other, "train")
        assert report["reference_scale"] == auto["reference_scale"]
        lib = env.evaluation_libraries[(str(other), "train")]
        assert lib.reference_scale_contract == auto["reference_scale"]
        for key in ("root_position", "root_velocity", "landmarks", "landmark_velocity"):
            torch.testing.assert_close(lib.values[key], env.library.values[key])
        import subprocess
        import sys

        native = subprocess.run(
            [
                sys.executable,
                "scripts/replay_checkpoint.py",
                "--checkpoint",
                str(tmp_path / "training/checkpoint.pt"),
                "--library",
                str(directory),
                "--split",
                "train",
                "--num-envs",
                "2",
                "--output",
                str(tmp_path / "native.json"),
            ],
            capture_output=True,
            text=True,
        )
        assert native.returncode == 0, native.stderr
        assert (
            json.loads((tmp_path / "native.json").read_text())["reference_scale"] == auto["reference_scale"]
        )
        # Same numeric scale with different semantics must not be silently replayed.
        import copy
        changed = copy.deepcopy(saved)
        changed["reward_settings"]["reference_scale"]["version"] = "unknown-scale-v2"
        torch.save(changed, tmp_path / "changed.pt")
        rejected = subprocess.run(
            [sys.executable, "scripts/replay_checkpoint.py", "--checkpoint", str(tmp_path / "changed.pt"),
             "--library", str(directory), "--split", "train", "--num-envs", "2",
             "--output", str(tmp_path / "changed.json")], capture_output=True, text=True)
        assert rejected.returncode != 0
        assert "reward" in rejected.stderr.lower()
        assert not (tmp_path / "changed.json").exists()
        panel = tmp_path / "bad-panel.json"
        panel.write_text(
            json.dumps(
                [
                    dict(
                        id="missing",
                        capture_group="heldout/other",
                        family="jump",
                        cohort="validation",
                        reference_path=str(tmp_path / "missing.npz"),
                    )
                ]
            )
        )
        failed = subprocess.run(
            [
                sys.executable,
                "scripts/evaluate_rl_reference_pilot.py",
                "--panel",
                str(panel),
                "--policy",
                str(tmp_path / "export/actor.pt"),
                "--output",
                str(tmp_path / "failed-replay"),
                "--workers",
                "1",
            ],
            capture_output=True,
            text=True,
        )
        assert failed.returncode != 0
        failure = json.loads((tmp_path / "failed-replay/summary.json").read_text())
        assert failure["execution_errors"] == 1 and failure["all"]["trials"] == 1
        assert failure["all"]["clean"] == 0 and not failure["execution_passed"]
        # A worker initializer failure must still account for every panel row.
        metadata_path = tmp_path / "export/actor.json"
        valid_metadata = metadata_path.read_text()
        invalid_metadata = json.loads(valid_metadata)
        invalid_metadata["sha256"] = "incorrect"
        metadata_path.write_text(json.dumps(invalid_metadata))
        broken_worker = subprocess.run(
            [sys.executable, "scripts/evaluate_rl_reference_pilot.py", "--panel", str(panel),
             "--policy", str(tmp_path / "export/actor.pt"),
             "--output", str(tmp_path / "worker-failure"), "--workers", "1"],
            capture_output=True, text=True)
        assert broken_worker.returncode != 0
        worker_summary = json.loads((tmp_path / "worker-failure/summary.json").read_text())
        assert worker_summary["execution_errors"] == 1
        assert worker_summary["all"]["trials"] == 1
        assert worker_summary["all"]["fell"] == 0
        metadata_path.write_text(valid_metadata)
        # A different scale may initialize weights, but cannot resume optimizer state.
        wrong = TrackerEnv(directory, 2, "cpu", "mujoco_cpp", **{**settings, "reference_scale": 0.85})
        try:
            with pytest.raises(ValueError, match="reward settings"):
                train(
                    wrong,
                    tmp_path / "wrong",
                    replace(cfg, iterations=1),
                    resume_checkpoint=tmp_path / "training/checkpoint.pt",
                )
        finally:
            wrong.close()
    finally:
        env.close()


def test_checkpoint_environment_arguments_preserve_treatment():
    from k1_motion.training_validation import checkpoint_environment_settings
    from k1_motion.observations import observation_contract

    saved = dict(
        observation=observation_contract(10, "preview", 0.0),
        action_settings={"residual_scale": 0.25},
        reward_settings={
            "profile": "world-capped-root-v1",
            "self_collision_weight": 0.0,
            "safety": {"version": "casual-safe-v1"},
            "reference_scale": {"version": "k1-travel-sole-scale-v1", "scale": 0.85},
        },
    )
    got = checkpoint_environment_settings(saved)
    assert got["safety_profile"] == "casual-safe-v1"
    assert got["reference_scale"] == 0.85
    assert got["observation_profile"] == "preview"
    assert got["preview_horizon_s"] == 0.0
    assert got["reference_storage"] == "packed"


def test_capped_reward_overrides_and_metadata():
    from k1_motion.world_objective import WorldBodyTracking

    c = WorldBodyTracking(
        "world-capped-root-v1",
        settings={
            "weights": {"root_xy_position": 8.0, "body_relative": 2.0},
            "scales": {"body_distance_cap_m": 0.2},
        },
    ).contract
    assert c["weights"]["root_xy_position"] == 8.0
    assert c["weights"]["body_relative"] == 2.0
    assert c["body_distance_cap_m"] == 0.2
    assert c["point_aggregation"] == "mean_of_capped_linear_distance_scores"


def test_legacy_human_stream_panel_rejects_scaled_policy():
    from types import SimpleNamespace
    from k1_motion.evaluation_panel import _trial
    with pytest.raises(ValueError, match='scaled'):
        _trial(None, SimpleNamespace(metadata={'reference_scale': {'scale': .9}}),
               {}, {}, None, {}, {}, None)


@pytest.mark.parametrize('bad_id', ['../outside', '/tmp/outside', 'nested/trial', 'summary', 'contract'])
def test_panel_rejects_unsafe_or_reserved_ids_before_output(tmp_path, bad_id):
    import subprocess
    import sys
    panel = tmp_path / 'panel.json'
    panel.write_text(json.dumps([dict(id=bad_id, cohort='validation', family='walk',
                                     capture_group='heldout', reference_path='missing.npz')]))
    output = tmp_path / 'output'
    result = subprocess.run([sys.executable, 'scripts/evaluate_rl_reference_pilot.py',
                             '--panel', str(panel), '--policy', str(tmp_path / 'missing.pt'),
                             '--output', str(output)], capture_output=True, text=True)
    assert result.returncode != 0
    assert 'panel id' in result.stderr.lower()
    assert not output.exists()


def test_failed_heldout_scale_is_not_cached(tmp_path, monkeypatch):
    # Failure risks: a half-transformed cache survives validation and is reused
    # on retry; the original training library must remain selected on failure.
    from types import SimpleNamespace
    import k1_motion.training_validation as validation
    import k1_motion.reference_scale as scaling
    original = SimpleNamespace(storage='packed', candidate_training=False)
    env = SimpleNamespace(library=original, corruption=True, library_directory='train',
                          spec=None, device='cpu', reward_settings={'reference_scale': {'scale': .9}})
    monkeypatch.setattr(validation, 'MotionLibrary', lambda *a, **kw: SimpleNamespace())
    calls = []
    def fail(*args):
        calls.append(1)
        raise ValueError('invalid reference')
    monkeypatch.setattr(scaling, 'scale_library', fail)
    for _ in range(2):
        with pytest.raises(ValueError, match='invalid reference'):
            validation.replay_panel(env, None, 'student', tmp_path, 'validation')
        assert env.library is original
        assert not env.evaluation_libraries
    assert len(calls) == 2
