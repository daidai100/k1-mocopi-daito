import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from k1_motion.contracts import MotionClip
from k1_motion.motion_coverage import (CoverageAccumulator, measure_motion, movement_tags,
                                     sustained_seconds)
from k1_motion.recovery_validation import audit_recovery
from k1_motion.robot import K1Model
from k1_motion.tracking_env import TrackerEnv
from k1_motion.training_validation import replay_panel
from test_ground_reference import buried_walk
from test_training import make_library


def preparation_module():
    path = Path(__file__).parents[1] / "scripts/prepare_broad_references.py"
    spec = importlib.util.spec_from_file_location("broad_preparation_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_multi_label_intent_does_not_turn_bumping_into_clearance():
    tags = movement_tags({"take_name": "dance_kick_boxing_R_001", "annotations": ["punching while dancing"]})
    assert {"dance", "kick", "boxing_striking"} <= set(tags)
    bump = movement_tags({"annotations": ["bumps into an obstacle"]})
    assert bump == ["obstacle_mention"]
    assert "step_over" in movement_tags({"take_name": "stepping_over_a_bar_R_001"})
    assert "step_over" in movement_tags({"annotations": ["A person walks over a low obstacle"]})
    assert "boxing_striking" not in movement_tags({"annotations": ["striking a dramatic dance pose"]})
    assert "boxing_striking" not in movement_tags({"annotations": ["punch yourself in the face"]})
    assert "shadowboxing_explicit" in movement_tags({"annotations": ["alternating boxing jabs"]})


def test_measured_span_keeps_durations_and_related_take_counts():
    robot = K1Model()
    clip = buried_walk([0.] * 21)
    values = {k: v.copy() for k, v in clip.values.items()}
    values["joint_position"][:, 2] += np.arange(21) * .05
    clip = MotionClip(clip.times, values, clip.metadata)
    features = measure_motion(clip, robot.limits)
    assert features["duration_s"] == pytest.approx(.4)
    assert features["event_longest_s"]["fast_arm_motion"] == pytest.approx(.4)
    assert features["event_longest_s"]["arms_and_legs_active"] == 0
    np.testing.assert_allclose(np.sum(features["joint_bin_seconds"], axis=1), .4)
    assert not features["obstacle_clearance_validated"]
    accumulator = CoverageAccumulator(robot.limits)
    for take in (1, 2):
        accumulator.add({"family": "gesture", "capture_group": f"bones_seed/wave_{take:03d}"}, features)
    report = accumulator.report()
    assert report["originals"] == 2 and report["related_take_families"] == 1
    assert report["measured_events"]["fast_arm_motion"]["take_families"] == 1
    assert sustained_seconds([True, False, True], .02) == .02
    assert sustained_seconds([], .02) == 0


def test_ground_only_candidate_keeps_split_and_invalid_tick_rejections():
    from test_reference_admission import row
    module = preparation_module()
    candidate = {**row(), "kinematics_accepted": False, "attempt_reference_path": "attempt.npz",
                 "recovery_audit": {"accepted": False,
                                    "rejection_reasons": ["ground_penetration_on_command_path"]}}
    assert module.candidate_rejections(candidate, set()) == []
    assert module.candidate_rejections(candidate, {"bones_seed/jog"}) == ["related_held_out_take"]
    assert "invalid_reference_ticks" in module.candidate_rejections({**candidate, "rejected_ticks": 1}, set())
    assert "external_support_not_configured" in module.candidate_rejections(
        {**candidate, "annotations": ["leaning against a wall"]}, set())


def test_expanded_payload_is_reaudited_and_preserves_clocks(tmp_path):
    module = preparation_module()
    robot = K1Model()
    metadata = {"id": "gesture-test", "capture_group": "fixture/wave", "split": "train",
                "family": "gesture", "source_motion_id": "fixture/wave", "is_mirror": False,
                "model_signature": robot.signature, "rejected_ticks": 0, "frames": 12,
                "rms_landmark_error_m": .01, "kinematics_accepted": True,
                "reference_path": "clip.npz"}
    refs = [robot.neutral_reference(i * .02) for i in range(12)]
    clip = MotionClip.from_references(refs, metadata, sample_times=np.arange(12) * .02)
    metadata["recovery_audit"] = audit_recovery(clip, clip, [{"rms_landmark_error_m": .01}], metadata)
    clip.metadata.update(metadata)
    clip.save(tmp_path / "clip.npz")
    result = module.prepare_one((str(tmp_path), metadata, str(tmp_path / "result")))
    assert result["kinematics_accepted"] and result["span_reference_audit"]["accepted"]
    saved = MotionClip.load(result["reference_path"])
    np.testing.assert_array_equal(saved.values["joint_position"], clip.values["joint_position"])
    np.testing.assert_array_equal(saved.source_times, clip.source_times)
    assert not result["physics_qualified"]
    with pytest.raises(ValueError, match="mismatch"):
        module.prepare_one((str(tmp_path), {**metadata, "split": "test"}, str(tmp_path / "bad")))


def test_periodic_validation_preserves_packed_storage(tmp_path):
    torch.set_num_threads(1)
    _, library = make_library(tmp_path)
    train = json.loads((library / "index.jsonl").read_text())
    validation = {**train, "split": "validation", "capture_group": "fixture/heldout",
                  "reference_path": "validation.npz"}
    clip = MotionClip.load(library / train["reference_path"])
    clip.metadata.update(validation)
    clip.save(library / validation["reference_path"])
    (library / "index.jsonl").write_text(json.dumps(train) + "\n" + json.dumps(validation) + "\n")
    env = TrackerEnv(library, num_envs=1, device="cpu", reference_storage="packed")
    actor = SimpleNamespace(actor=lambda obs: torch.zeros((len(obs), 22)))
    result = replay_panel(env, actor, "student", library, "validation")
    assert result["total"] == 1
    assert env.evaluation_libraries[str(library), "validation"].storage == "packed"
    assert result["trials"][0]["resets_during_trial"] == 0
    env.close()


def test_take_balancing_preserves_every_family_without_actor_replication_bias(tmp_path):
    from k1_motion.learning import MotionLibrary
    _, directory = make_library(tmp_path)
    original = json.loads((directory / "index.jsonl").read_text())
    # Three actor/take variants of A versus one of B in the same family.
    rows = [{**original, "capture_group": f"bones_seed/{name}"} for name in ("A_001", "A_002", "A_003", "B_001")]
    rows.append({**original, "family": "kick", "capture_group": "bones_seed/kick_001"})
    (directory / "index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    library = MotionLibrary(directory, K1Model(), "cpu", storage="packed")
    library.configure_sampling("take_transition_balanced")
    weights = library.weights / library.weights.sum()
    torch.testing.assert_close(weights[:3].sum(), weights[3])
    torch.testing.assert_close(weights[:4].sum(), weights[4])
    assert (weights > 0).all()
    library.configure_sampling("transition_balanced")
    torch.testing.assert_close(library.weights[0], library.weights[3])


def test_complementary_selection_distinguishes_motion_from_unconfigured_support():
    path = Path(__file__).parents[1] / "scripts/prepare_complementary_references.py"
    spec = importlib.util.spec_from_file_location("complementary_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.gap_family({"annotations": ["strikes a dance pose"]}) is None
    assert module.gap_family({"annotations": ["boxing with alternating punches"]}) == "punch"
    assert module.gap_family({"annotations": ["walks over a low obstacle"]}) == "step_over"
    assert module.gap_family({"annotations": ["dodges to the left"]}) == "avoidance"
    assert module.unconfigured_support({"annotations": ["walks on stepping stones"]})
    assert not module.unconfigured_support({"annotations": ["steps over a low obstacle"]})


def test_source_label_correction_preserves_motion_and_audits(tmp_path):
    path = Path(__file__).parents[1] / "scripts/curate_motion_labels.py"
    spec = importlib.util.spec_from_file_location("label_curation_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    robot = K1Model()
    metadata = {"id": "dance-pose", "capture_group": "fixture/pose", "split": "train",
                "family": "punch", "source_motion_id": "fixture/pose", "is_mirror": False,
                "model_signature": robot.signature, "rejected_ticks": 0, "frames": 12,
                "annotations": ["striking a dance pose"], "movement_tags": ["boxing_striking"],
                "rms_landmark_error_m": .01, "kinematics_accepted": True,
                "reference_path": str(tmp_path / "original.npz")}
    refs = [robot.neutral_reference(i * .02) for i in range(12)]
    clip = MotionClip.from_references(refs, metadata, sample_times=np.arange(12) * .02)
    metadata["recovery_audit"] = audit_recovery(clip, clip, [{"rms_landmark_error_m": .01}], metadata)
    clip.metadata.update(metadata)
    clip.save(tmp_path / "original.npz")
    (tmp_path / "clips").mkdir()
    corrected = module.curate_row(metadata, tmp_path, "fixture-revision")
    assert corrected["family"] == "dance" and corrected["source_family"] == "punch"
    assert corrected["movement_tags"] == ["dance"]
    saved = MotionClip.load(corrected["reference_path"])
    original = MotionClip.load(tmp_path / "original.npz")
    assert original.metadata["family"] == "punch"
    assert saved.metadata["recovery_audit"] == original.metadata["recovery_audit"]
    for key in saved.values:
        np.testing.assert_array_equal(saved.values[key], original.values[key])
    np.testing.assert_array_equal(saved.source_times, original.source_times)
    with pytest.raises(ValueError, match="strict-audited"):
        module.curate_row({**metadata, "kinematics_accepted": False}, tmp_path, "fixture-revision")
    assert module.corrected_labels({"family": "punch", "annotations": ["a boxing jab"]})[0] == "punch"
