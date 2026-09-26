import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from k1_motion.collision_feedback import ArmCollisionFeedback
from k1_motion.parallel_collision_feedback import ParallelArmCollisionFeedback
from k1_motion.robot import K1Model
from k1_motion.training_curriculum import TrainingCurriculum


def test_parallel_arm_feedback_matches_original():
    robot = K1Model()
    generator = np.random.default_rng(92)
    q = generator.uniform(robot.limits[:, 0], robot.limits[:, 1], (384, 22)).astype(np.float32)
    dq = generator.normal(0, 2, q.shape).astype(np.float32)
    target = q + generator.uniform(-.12, .12, q.shape).astype(np.float32)
    inputs = [torch.from_numpy(x) for x in (q, dq, target)]
    original = ArmCollisionFeedback(robot, .025)
    parallel = ParallelArmCollisionFeedback(robot, .025, 2)
    try:
        expected = original.project_batch(*inputs)
        actual = parallel.project_batch(*inputs)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0)
        assert (actual - inputs[2]).abs().max() <= .150001
        torch.testing.assert_close(actual[:, :2], inputs[2][:, :2], atol=0, rtol=0)
        torch.testing.assert_close(actual[:, 10:], inputs[2][:, 10:], atol=0, rtol=0)
        assert (actual - inputs[2]).abs().max() > .01
    finally:
        parallel.close()


def curriculum(tmp_path):
    library = SimpleNamespace(rows=[{"id": "walk", "split": "train"}, {"id": "stance", "split": "train"}],
                              lengths=torch.tensor([200, 100]), device="cpu", take_weights=torch.ones(2),
                              episode_duration_ema=torch.tensor([20., 40.]))
    path = tmp_path / "curriculum.json"
    path.write_text(json.dumps({"train_ids": ["walk", "stance"], "locomotion_ids": ["walk"]}))
    return library, path


def test_curriculum_bounds_balance_and_resume(tmp_path):
    library, path = curriculum(tmp_path)
    sampler = TrainingCurriculum(library, path)
    exposure = library.weights * library.episode_duration_ema
    torch.testing.assert_close(exposure[0], exposure[1])
    clips = torch.zeros(12000, dtype=torch.long)
    sampler.record(clips, torch.full_like(clips, 125), torch.ones_like(clips, dtype=torch.bool))
    frames = sampler.sample_frames(clips)
    assert frames.min() >= 0 and frames.max() <= 198
    assert .24 < float((frames == 0).float().mean()) < .28
    assert float(((frames >= 50) & (frames < 100)).float().mean()) > .30
    restored = TrainingCurriculum(library, path)
    restored.load_state_dict(sampler.state_dict())
    torch.testing.assert_close(restored.failure_counts, sampler.failure_counts)
    library.rows[0]["split"] = "validation"
    with pytest.raises(ValueError, match="training set"):
        TrainingCurriculum(library, path)


def test_kl_guard_counts_actual_adam_steps(tmp_path):
    from test_training import make_library
    from k1_motion.learning import TrainConfig, train
    from k1_motion.tracking_env import TrackerEnv
    _, directory = make_library(tmp_path)
    torch.set_num_threads(1)
    env = TrackerEnv(directory, 4, "cpu", "mujoco_cpp", reference_storage="packed",
                     physics_options={"workers": 2})
    try:
        config = TrainConfig(iterations=2, horizon=4, epochs=4, minibatch=4, hidden_sizes=(32, 16),
                             evaluation_interval=0, learning_rate=1e-5, min_learning_rate=1e-6, kl_stop=1e-12)
        result = train(env, tmp_path / "training", config)
        saved = torch.load(tmp_path / "training/checkpoint.pt", weights_only=True)
        actual = max(float(v["step"]) for v in saved["optimizer"]["state"].values())
        assert result["optimizer_steps"] == actual < 32
        assert result["last_metrics"]["learning_rate"] < 1e-5
        assert (tmp_path / "training/checkpoint-000002.pt").exists()
    finally:
        env.close()


def test_beam_preserves_early_champions_and_four_candidates():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
    from candidate_beam import candidate_beam
    comparisons = {}
    for run, iteration, clean, collisions in (("r0", 250, 16, 22), ("r0", 500, 10, 30),
                                             ("r1", 250, 17, 7), ("r2", 250, 18, 6), ("r3", 250, 15, 10)):
        comparisons[f"{run}/{iteration}"] = {
            "all": {"clean": clean, "collision_trials": collisions, "completed": 24},
            "by_family": {"walk": {"clean": 1}}, "training_exposure": {"checkpoint_transitions": iteration*65536}}
    beam = candidate_beam(comparisons)
    assert len(beam["selected"]) == 4
    assert beam["best_per_run"]["r0"] == "r0/250"
    assert "r0/500" not in beam["selected"]
