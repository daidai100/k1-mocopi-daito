from dataclasses import replace
import fcntl

import pytest
import torch

from k1_motion.gpu_scheduling import gpu_update_slot
from k1_motion.learning import TrainConfig, train
from k1_motion.tracking_env import TrackerEnv


def test_update_slot_releases_after_exception(tmp_path):
    path = tmp_path / "gpu.lock"
    with pytest.raises(RuntimeError):
        with gpu_update_slot(path, "cpu") as slot:
            assert slot["wait_seconds"] >= 0
            raise RuntimeError("test update failure")
    with path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(lock, fcntl.LOCK_UN)


def test_update_scheduling_preserves_model_and_adam(tmp_path):
    from test_training import make_library
    _, library = make_library(tmp_path)
    torch.set_num_threads(1)
    config = TrainConfig(iterations=3, horizon=4, epochs=2, minibatch=8,
                         hidden_sizes=(32, 16), evaluation_interval=0)
    saved = []
    for name, path in (("original", None), ("scheduled", str(tmp_path / "gpu.lock"))):
        env = TrackerEnv(library, 4, "cpu", "mujoco_cpp", physics_options={"workers": 2})
        try:
            report = train(env, tmp_path / name, replace(config, ppo_update_lock=path))
            assert report["finite_updates"] and report["checkpoint_reload_max_error"] == 0
            saved.append(torch.load(tmp_path / name / "checkpoint.pt", weights_only=True))
        finally:
            env.close()
    for name in saved[0]["model"]:
        torch.testing.assert_close(saved[0]["model"][name], saved[1]["model"][name], atol=0, rtol=0)
    for key, values in saved[0]["optimizer"]["state"].items():
        for name, value in values.items():
            torch.testing.assert_close(value, saved[1]["optimizer"]["state"][key][name], atol=0, rtol=0)
