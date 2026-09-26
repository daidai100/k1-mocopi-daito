import json

import torch

from k1_motion.learning import MotionLibrary, TrainConfig, train
from k1_motion.contracts import MotionClip
from k1_motion.reference_admission import admit_reference, reference_rejections, take_family
from k1_motion.tracking_env import TrackerEnv
from test_training import make_library


def row():
    return {"capture_group": "bones_seed/jog_003", "split": "train", "family": "run",
            "kinematics_accepted": True, "recovery_audit": {"accepted": True},
            "rejected_ticks": 0, "frames": 100, "reference_path": "run.npz", "is_mirror": False,
            "physics_qualified": False, "controller_replay_passed": False}


def test_controller_failure_does_not_reject_rl_reference():
    failed = row()
    assert reference_rejections(failed) == []
    assert reference_rejections({**failed, "controller_replay_passed": True}) == []
    admitted = admit_reference(failed)
    assert admitted["training_eligible"] and admitted["rl_training_eligible"]
    assert not admitted["physics_qualified"]
    assert not admit_reference({**failed, "split": "test"})["training_eligible"]
    assert reference_rejections({**failed, "rejected_ticks": 1}) == ["invalid_reference_ticks"]
    assert "retargeting_audit_failed" in reference_rejections({**failed, "recovery_audit": {"accepted": False}})


def test_related_takes_and_unconfigured_contacts_stay_out_of_flat_ground_training():
    assert take_family("bones_seed/jog_003") == take_family("bones_seed/jog_002")
    assert reference_rejections(row(), {"bones_seed/jog"}) == ["related_held_out_take"]
    assert reference_rejections({**row(), "family": "climb"}) == ["contact_task_not_configured"]
    assert reference_rejections({**row(), "annotations": ["walking with a crutch"]}) == [
        "external_support_not_configured"]
    for description in ("crouching while supporting their hand on the wall",
                        "standing from a crouch, using a wall for support"):
        assert reference_rejections({**row(), "family": "squat", "annotations": [description]}) == [
            "external_support_not_configured"]
    assert not reference_rejections({**row(), "annotations": ["walking beside a wall"]})


def test_rl_admitted_reference_loads_and_trains_without_a_teacher_or_previous_controller_pass(tmp_path):
    torch.set_num_threads(1)
    spec, library = make_library(tmp_path)
    index = library / "index.jsonl"
    original = json.loads(index.read_text())
    admitted = admit_reference({**row(), **original, "controller_replay_passed": False})
    index.write_text(json.dumps(admitted) + "\n")
    data = MotionLibrary(library, spec, "cpu")
    assert len(data.rows) == 1 and not data.rows[0]["physics_qualified"]
    env = TrackerEnv(library, num_envs=2, device="cpu", root_velocity_weight=2, root_velocity_sigma=0.5)
    report = train(env, tmp_path / "ppo", TrainConfig(stage="student", iterations=1, horizon=4,
                   epochs=1, minibatch=8, evaluation_interval=0, bc_weight=0))
    assert report["finite_updates"] and report["teacher_checkpoint"] is None
    assert report["checkpoint_reload_max_error"] == 0
    assert report["reward_settings"]["root_velocity_weight"] == 2
    env.close()


def test_packed_references_preserve_frame_lookup_without_padding(tmp_path):
    spec, library = make_library(tmp_path)
    original = json.loads((library / "index.jsonl").read_text())
    short_metadata = {**original, "capture_group": "synthetic/short", "reference_path": "short.npz"}
    refs = [spec.neutral_reference(i * 0.02) for i in range(13)]
    MotionClip.from_references(refs, short_metadata).save(library / "short.npz")
    (library / "index.jsonl").write_text(json.dumps(original) + "\n" + json.dumps(short_metadata) + "\n")
    padded = MotionLibrary(library, spec, "cpu", storage="padded")
    packed = MotionLibrary(library, spec, "cpu", storage="packed")
    assert packed.fingerprint == padded.fingerprint
    indices = torch.tensor([0, 1, 1, 0, 1, 0])
    frames = torch.tensor([49, 12, 13, -1, 100, 25])
    a, b = padded.frames(indices, frames), packed.frames(indices, frames)
    for key in a:
        torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)
    assert packed.values["joint_position"].shape[0] == 63
    assert packed.values["joint_position"].numel() < padded.values["joint_position"].numel()


def test_velocity_reward_change_does_not_change_physics(tmp_path):
    spec, library = make_library(tmp_path)
    outputs = []
    for weight in (0.5, 2.0):
        env = TrackerEnv(library, num_envs=2, device="cpu", root_velocity_weight=weight)
        env.reset(clips=torch.zeros(2, dtype=torch.long), frames=torch.zeros(2, dtype=torch.long))
        _, _, reward, _, _ = env.step(torch.zeros(2, 22), auto_reset=False)
        state = env.physics.state()
        target = env.library.frames(env.clips, env.frames)["root_velocity"][:, :3]
        velocity_reward = (-((state["velocity"] - target).square().sum(-1)) / 0.75**2).exp()
        outputs.append((reward, state, velocity_reward))
        env.close()
    for key in outputs[0][1]:
        torch.testing.assert_close(outputs[0][1][key], outputs[1][1][key], atol=0, rtol=0)
    torch.testing.assert_close(outputs[1][0] - outputs[0][0], 1.5 * outputs[0][2] * spec.control_dt)
