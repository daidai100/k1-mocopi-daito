from dataclasses import replace

import numpy as np
import pytest

from k1_motion.contracts import MotionClip
from k1_motion.ground_reference import correct_walking_ground
from k1_motion.recovery_validation import audit_recovery
from k1_motion.reference_admission import (GROUNDED_RL_FAMILIES, ground_rl_accepted,
                                            grounded_reference_audit, reference_rejections,
                                            walking_reference_audit, walking_rl_accepted)
from k1_motion.robot import K1Model


def buried_walk(depths):
    robot = K1Model()
    frames = []
    for i, depth in enumerate(depths):
        reference = robot.neutral_reference(i * robot.control_dt)
        delta = np.array([i * 0.001, 0, -depth])
        frames.append(replace(reference, root_position=reference.root_position + delta,
                              landmarks=reference.landmarks + delta))
    return MotionClip.from_references(frames, {"model_signature": robot.signature})


def test_corrected_ground_is_audited_and_preserves_motion():
    original = buried_walk([.02] * 8)
    corrected, receipt = correct_walking_ground(original)
    report = [{"rms_landmark_error_m": receipt["mean_landmark_error_increase_bound_m"]}]
    before = audit_recovery(original, original, [{"rms_landmark_error_m": 0}],
                            {"rms_landmark_error_m": 0})
    after = audit_recovery(corrected, original, report, {"rms_landmark_error_m": 0})
    assert "ground_penetration_on_command_path" in before["rejection_reasons"]
    assert after["accepted"] and after["max_ground_penetration_m"] < .005
    for key in ("joint_position", "joint_velocity", "root_orientation", "contacts", "valid"):
        np.testing.assert_array_equal(corrected.values[key], original.values[key])
    np.testing.assert_array_equal(corrected.values["root_position"][:, :2],
                                  original.values["root_position"][:, :2])
    assert not after["physics_qualified"]


def test_ground_correction_is_causal_and_reset_between_clips():
    original = buried_walk([.01, .012, .014, .016, .02, .018, .016, .01])
    full, _ = correct_walking_ground(original)
    prefix = MotionClip(original.times[:4], {k: v[:4] for k, v in original.values.items()},
                        original.metadata)
    short, _ = correct_walking_ground(prefix)
    again, _ = correct_walking_ground(original)
    for key in full.values:
        np.testing.assert_array_equal(full.values[key][:4], short.values[key])
        np.testing.assert_array_equal(full.values[key], again.values[key])
    height = full.values["root_position"][:, 2] - original.values["root_position"][:, 2]
    np.testing.assert_allclose(full.values["root_velocity"][1:, 2], np.diff(height) / .02,
                               atol=1e-12)


def test_severe_penetration_is_not_hidden_by_unbounded_lift():
    original = buried_walk([.20] * 5)
    corrected, receipt = correct_walking_ground(original)
    assert receipt["max_lift_m"] <= .05 and receipt["height_limited_ticks"] > 0
    audit = audit_recovery(corrected, original, [{"rms_landmark_error_m": 0}],
                           {"rms_landmark_error_m": 0})
    assert not audit["accepted"]
    assert "ground_penetration_on_command_path" in audit["rejection_reasons"]


def test_nonfinite_settings_and_wrong_model_fail():
    original = buried_walk([.01] * 3)
    with pytest.raises(ValueError, match="settings"):
        correct_walking_ground(original, {"max_lift_m": float("nan")})
    original.metadata["model_signature"] = "wrong"
    with pytest.raises(ValueError, match="model mismatch"):
        correct_walking_ground(original)


def test_rl_ground_admission_allows_brief_foot_error_but_not_sustained_or_severe_error():
    depths = np.zeros(200)
    depths[80] = .015
    original = buried_walk(depths)
    geometry = audit_recovery(original, original, [{"rms_landmark_error_m": 0}],
                               {"rms_landmark_error_m": 0}, ground_profile=True)
    assert not geometry["accepted"]
    audit = walking_reference_audit(geometry)
    assert audit["accepted"] and audit["reset_height_correction_required"]
    row = {"family": "walk", "rejected_ticks": 0, "kinematics_accepted": False,
           "recovery_audit": geometry, "rl_reference_audit": audit,
           "capture_group": "bones_seed/walk_001", "split": "train", "frames": 200,
           "reference_path": "walk.npz"}
    assert walking_rl_accepted(row) and not reference_rejections(row)
    assert not walking_rl_accepted({**row, "family": "crawl"})
    assert not walking_rl_accepted({**row, "rejected_ticks": 1})
    assert reference_rejections(row, {"bones_seed/walk"}) == ["related_held_out_take"]
    for key, value in (("max_penetration_m", .026), ("fraction_over_5mm", .021),
                       ("longest_over_5mm_s", .062), ("max_nonfoot_penetration_m", .006),
                       ("mean_penetration_m", .0006)):
        bad = {**geometry, "ground_profile": {**geometry["ground_profile"], key: value}}
        assert not walking_reference_audit(bad)["accepted"]
        assert not walking_rl_accepted({**row, "recovery_audit": bad})
    bad = {**geometry, "rejection_reasons": ["self_collision_on_command_path"]}
    assert not walking_reference_audit(bad)["accepted"]


@pytest.mark.parametrize("family", ["walk", "turn", "squat", "run", "transition", "dance", "idle_stance"])
def test_ground_rl_library_reset_is_above_floor_without_changing_reference(tmp_path, family):
    import json
    import torch
    from k1_motion.reference_admission import admit_reference
    from k1_motion.tracking_env import TrackerEnv

    clip = buried_walk([.0] * 80 + [.015] + [.0] * 119)
    geometry = audit_recovery(clip, clip, [{"rms_landmark_error_m": 0}],
                               {"rms_landmark_error_m": 0}, ground_profile=True)
    row = admit_reference({"id": "ground-test", "capture_group": "fixture/walk", "split": "train",
                           "model_signature": clip.metadata["model_signature"],
                           "family": family, "rejected_ticks": 0, "kinematics_accepted": False,
                           "recovery_audit": geometry, "rl_reference_audit":
                           walking_reference_audit(geometry) if family == "walk"
                           else grounded_reference_audit(geometry, family),
                           "reference_path": "walk.npz", "frames": len(clip.times)})
    assert ground_rl_accepted(row)
    clip.metadata.update(row)
    clip.save(tmp_path / "walk.npz")
    (tmp_path / "index.jsonl").write_text(json.dumps(row) + "\n")
    env = TrackerEnv(tmp_path, num_envs=1, device="cpu", reference_storage="packed")
    env.reset(clips=torch.tensor([0]), frames=torch.tensor([80]))
    robot = env.physics.robots[0]
    depth = max((-float(c.dist) for c in robot.data.contact[:robot.data.ncon]
                 if 0 in robot.model.geom_bodyid[[c.geom1, c.geom2]]), default=0)
    assert depth < .00001
    assert robot.data.qpos[2] > clip.values["root_position"][80, 2] + .01
    np.testing.assert_allclose(env.library.frames(torch.tensor([0]), torch.tensor([80]))["root_position"][0],
                               clip.values["root_position"][80], atol=1e-7)
    env.close()


def test_grounded_family_contract_is_explicit_and_fail_closed():
    geometry = {"geometry_audited": True, "audit_hz": 500, "accepted": False,
                "rejection_reasons": ["ground_penetration_on_command_path"],
                "ground_profile": {"max_penetration_m": .015, "mean_penetration_m": .0001,
                                   "fraction_over_5mm": .01, "longest_over_5mm_s": .02,
                                   "max_nonfoot_penetration_m": 0}}
    for family in GROUNDED_RL_FAMILIES:
        row = {"family": family, "rejected_ticks": 0, "recovery_audit": geometry,
               "rl_reference_audit": grounded_reference_audit(geometry, family)}
        assert ground_rl_accepted(row) and not walking_rl_accepted(row)
        assert not ground_rl_accepted({**row, "family": "walk"})
        assert not ground_rl_accepted({**row, "rejected_ticks": 1})
        assert not ground_rl_accepted({**row, "recovery_audit": {
            **geometry, "rejection_reasons": ["stance_foot_slip"]}})
        for bad_family in ("climb", "crawl", "sit_or_kneel", "jump", "unknown"):
            assert not grounded_reference_audit(geometry, bad_family)["accepted"]
            assert not ground_rl_accepted({**row, "family": bad_family})
        bad_audit = {**row["rl_reference_audit"], "ground_limits": {"max_penetration_m": 1}}
        assert not ground_rl_accepted({**row, "rl_reference_audit": bad_audit})


def test_ground_repair_keeps_cumulative_human_tracking_budget(tmp_path):
    import importlib.util
    from pathlib import Path

    path = Path(__file__).parents[1] / "scripts/prepare_walking_references.py"
    spec = importlib.util.spec_from_file_location("prepare_grounded_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    clip = buried_walk([.02] * 10)
    identity = {"id": "tracking-budget", "family": "squat", "capture_group": "fixture/squat",
                "split": "train", "source_motion_id": "fixture/squat", "is_mirror": False,
                "model_signature": clip.metadata["model_signature"]}
    clip.metadata.update(identity, rms_landmark_error_m=.01)
    source, output = tmp_path / "source", tmp_path / "output"
    clip.save(source / "attempt.npz")
    original = {**identity, "kinematics_accepted": False, "rejected_ticks": 0,
                "rms_landmark_error_m": .02, "attempt_reference_path": "attempt.npz",
                "recovery_audit": {"accepted": False,
                                   "rejection_reasons": ["ground_penetration_on_command_path"]}}
    row = module.repair_one((str(source), original, str(output)))
    assert row["ground_correction"]["mean_landmark_error_increase_bound_m"] < .015
    assert "human_tracking_regression" in row["recovery_audit"]["rejection_reasons"]
    assert row["recovery_audit"]["original_human_tracking_baseline_m"] == .01
    assert not row["kinematics_accepted"] and not ground_rl_accepted(row)
    assert "reference_path" not in row
