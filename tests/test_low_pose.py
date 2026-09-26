from dataclasses import replace
import json

import mujoco
import numpy as np
import pytest

from k1_motion.adapters import BVH_BASIS, BVH_MAPS, Bvh, bvh_frames
from k1_motion.contracts import HumanFrame, MotionClip
from k1_motion.low_pose import (LOW_POSE_SOURCE, LOW_POSE_VERSION, LowPoseRetargeter,
                               correct_low_pose_ground, retarget_low_pose, skeletal_calibration,
                               source_pose_features)
from k1_motion.low_pose_contract import (low_pose_accepted, low_pose_reference_audit,
                                        minimum_tracking_height, minimum_tracking_upright)
from k1_motion.low_pose_validation import LOW_POSE_AUDIT_VERSION, LOW_POSE_GATES, audit_low_pose
from k1_motion.recovery_geometry import geometry_forward, has_self_penetration
from k1_motion.reference_admission import reference_rejections
from k1_motion.robot import K1Model


def human_frames(count=20):
    robot = K1Model()
    return [HumanFrame(i*.02, i*.02, i, robot.neutral_landmarks*2,
                       np.tile([1., 0, 0, 0], (17, 1)), "fixture") for i in range(count)]


def kneeling_frames(count=20):
    robot = K1Model()
    robot.data.qpos[2] = .32
    for offset, roll in ((10, .05), (16, -.05)):
        robot.data.qpos[7+offset:7+offset+6] = [-.06, roll, 0, 1.85, .344, 0]
    geometry_forward(robot.model, robot.data)
    # Human kneecap markers in this source convention lie below the robot's
    # 45mm collision cylinder center; skeleton segment lengths stay unchanged.
    p = (robot.landmarks()-[0, 0, .04])*2
    return [HumanFrame(i*.02, i*.02, i, p, np.tile([1., 0, 0, 0], (17, 1)), "kneel-fixture")
            for i in range(count)]


def test_absolute_soma_translation_does_not_double_initial_hips_height():
    channels = [f"{axis}{kind}" for kind in ("position", "rotation") for axis in "XYZ"]
    bvh = Bvh(["Root", "Hips"], [-1, 0], np.array([[0., 0, 0], [0, 50, 0]]),
              [channels, channels], np.tile([0]*6+[0, 50, 0, 0, 0, 0], (2, 1)), .01)
    absolute, _ = bvh.fk("replace")
    old, _ = bvh.fk("add")
    assert absolute[0, 1, 1] == 50
    assert old[0, 1, 1] == 100


def test_absolute_profile_retains_end_timestamp_without_early_source_access(monkeypatch):
    p = BVH_BASIS.inv().apply(human_frames(1)[0].positions)*100
    names = ["Root", *BVH_MAPS["bones_seed_v2"]]
    translation = [f"{axis}{kind}" for kind in ("position", "rotation") for axis in "XYZ"]
    channels = [translation, translation] + [[axis+"rotation" for axis in "XYZ"]]*16
    offsets = np.vstack([np.zeros(3), p[0], p[1:]-p[0]])
    values = np.zeros((10, sum(map(len, channels))))
    values[:, 6:9] = p[0]
    rig = Bvh(names, [-1, 0]+[1]*16, offsets, channels, values, 1/120)
    monkeypatch.setattr(Bvh, "load", classmethod(lambda cls, path: rig))
    frames = list(bvh_frames("fixture", "bones_seed_v2", target_hz=50))
    assert frames[-1].source_time == pytest.approx(9/120)
    clip, _ = retarget_low_pose(K1Model(), frames, {})
    np.testing.assert_allclose(clip.times, [0, .02, .04, .06])
    assert np.all(clip.source_times <= clip.times)
    assert frames[-1].source_time not in clip.source_times
    np.testing.assert_allclose(frames[0].positions, human_frames(1)[0].positions)
    values[1, 0] = 1
    with pytest.raises(ValueError, match="stationary SOMA"):
        list(bvh_frames("fixture", "bones_seed_v2", target_hz=50))


def test_bone_length_calibration_is_independent_of_initial_pelvis_height():
    frame = human_frames(1)[0]
    low = replace(frame, positions=frame.positions-[0, 0, .5])
    robot = K1Model()
    a, b = skeletal_calibration(frame, robot), skeletal_calibration(low, robot)
    assert a.scale == pytest.approx(.5)
    assert b.scale == pytest.approx(a.scale)
    assert a.floor == b.floor == 0
    assert b.apply(low).positions[0, 2] == pytest.approx(low.positions[0, 2]*.5)
    with pytest.raises(ValueError, match="skeleton"):
        skeletal_calibration(replace(frame, positions=frame.positions*100), robot)


def test_kneeling_requires_low_and_folded_knees_not_an_annotation():
    frame = human_frames(1)[0]
    assert not source_pose_features(replace(frame, source_id="kneeling_axe"), .5)["kneeling"].any()
    p = frame.positions.copy()
    p[9] = [0, .15, .4]
    p[10] = [0, .15, .02]
    p[11] = [-.4, .15, .08]
    features = source_pose_features(replace(frame, positions=p), .5)
    assert features["kneeling"].tolist() == [True, False]


@pytest.mark.parametrize("factory", [human_frames, kneeling_frames])
def test_low_pose_prefix_causality_reset_and_velocity_limits(factory):
    frames = factory(8)
    robot = K1Model()
    full, _ = retarget_low_pose(robot, frames, {"source_motion_id": "fixture"})
    prefix, _ = retarget_low_pose(robot, frames[:4], {"source_motion_id": "fixture"})
    again, _ = retarget_low_pose(robot, frames, {"source_motion_id": "fixture"})
    for key in full.values:
        np.testing.assert_array_equal(full.values[key][:len(prefix.times)], prefix.values[key])
        np.testing.assert_array_equal(full.values[key], again.values[key])
    assert abs(np.diff(full.values["joint_position"], axis=0)/.02).max() <= 6+1e-6
    assert np.all(full.received_times <= full.times)
    assert not robot.model.geom_margin.any()
    for settings in ({"root_correction_limit_m": float("nan")}, {"iterations": 1.5}, {"unknown": 1}):
        with pytest.raises(ValueError, match="settings"):
            LowPoseRetargeter(robot, settings)


def test_margin_proximity_is_not_a_real_collision_and_does_not_block_a_clear_path():
    robot = K1Model()
    solver = LowPoseRetargeter(robot)
    solver.data.qpos[:] = robot.neutral_qpos
    solver.data.qpos[2] = 1.
    solver.data.qpos[7:] = [0.05170926422732919, 0.1672669029452999, -0.06924753330226105,
        -1.4307382206251047, 0.701797007693309, -1.3611799077065592, -0.1762604288780005,
        1.5160407532589373, 0.1878865868553333, 0.35990414841539037, -1.2300825723689948,
        -0.040188754899915524, 0.2144722214782696, 0.8862723133666903, -0.023881132523547628,
        0.20818629543989478, -0.2419701633319694, -0.3589611722544242, -0.7628112472475055,
        0.8346671286319666, -0.648448069398489, 0.34498999999999996]
    solver.robot.model.geom_margin[:] = .02
    geometry_forward(solver.robot.model, solver.data)
    phantom = [c for c in solver.data.contact[:solver.data.ncon] if c.dist < -.001
               and 0 not in solver.robot.model.geom_bodyid[[c.geom1, c.geom2]]]
    assert phantom
    assert all(mujoco.mj_geomDistance(solver.robot.model, solver.data, c.geom1, c.geom2, .1, None) > 0
               for c in phantom)
    solver.robot.model.geom_margin[:] = 0
    geometry_forward(solver.robot.model, solver.data)
    assert not has_self_penetration(solver.robot.model, solver.data, .00001)
    solver.robot.model.geom_margin[:] = .02
    old = solver.data.qpos.copy()
    solver.control_origin = old.copy()
    solver.data.qpos[0] += .005
    solver._safe_step(old)
    assert solver.data.qpos[0] == pytest.approx(old[0]+.005)
    assert np.all(solver.robot.model.geom_margin == .02)
    assert not robot.model.geom_margin.any()


def test_independent_audit_rejects_standing_as_kneeling_and_changed_clocks():
    frames = human_frames()
    clip, _ = retarget_low_pose(K1Model(), frames, {"family": "kneel"})
    clip, _ = correct_low_pose_ground(clip)
    result = audit_low_pose(clip, frames, "kneel")
    assert not result["accepted"]
    assert "no_genuine_kneeling_phase" in result["rejection_reasons"]
    assert result["source_kneeling_seconds"] == result["genuine_kneeling_seconds"] == 0
    truncated = MotionClip(clip.times[:-1], {k: v[:-1] for k, v in clip.values.items()}, clip.metadata,
                           clip.source_times[:-1], clip.received_times[:-1])
    with pytest.raises(ValueError, match="duration"):
        audit_low_pose(truncated, frames, "kneel")


def passing_receipt():
    geometry = {"version": LOW_POSE_AUDIT_VERSION, "gates": LOW_POSE_GATES, "family": "kneel",
                "accepted": True, "rejection_reasons": [], "geometry_audited": True, "audit_hz": 500,
                "full_source_duration_preserved": True, "max_self_penetration_m": 0,
                "max_ground_penetration_m": 0, "unsupported_body_penetration_m": 0,
                "saved_landmark_error_m": 0, "joint_speed_max_rad_s": 2, "stance_slip_p95_m_s": .01,
                "mean_directional_landmark_rms_m": .01, "max_directional_landmark_rms_m": .02,
                "feasible_knee_flexion_error_p95_rad": .1, "source_hand_support_seconds": 0,
                "knee_flexion_target": "source_clamped_to_unchanged_robot_joint_limits",
                "self_collision_samples": 0, "ground_penetration_samples": 0,
                "source_kneeling_seconds": 2, "genuine_kneeling_seconds": 2,
                "knee_phase_recall": 1, "knee_label_precision": 1}
    return {"family": "kneel", "rejected_ticks": 0, "retarget_version": LOW_POSE_VERSION,
            "source_adapter": LOW_POSE_SOURCE, "kinematics_accepted": True,
            "recovery_audit": geometry, "low_pose_reference_audit": low_pose_reference_audit(geometry),
            "frames": 100, "reference_path": "fixture.npz", "split": "train",
            "capture_group": "fixture/kneel"}


def test_knee_admission_fails_closed_without_pose_proof_or_with_tampered_metrics():
    row = passing_receipt()
    assert low_pose_accepted(row) and not reference_rejections(row)
    assert reference_rejections(row, {"fixture/kneel"}) == ["related_held_out_take"]
    assert "low_support_task_not_audited" in reference_rejections({**row, "low_pose_reference_audit": {}})
    for key, value in (("max_ground_penetration_m", .006), ("knee_phase_recall", .89),
                       ("genuine_kneeling_seconds", 0), ("joint_speed_max_rad_s", float("nan"))):
        changed = {**row, "recovery_audit": {**row["recovery_audit"], key: value}}
        changed["low_pose_reference_audit"] = low_pose_reference_audit(changed["recovery_audit"])
        assert not low_pose_accepted(changed)
    assert not low_pose_accepted({**row, "family": "climb"})


def test_low_support_fall_floor_is_explicit_and_does_not_change_standing_tasks():
    heights = np.array([.55, .3, .21, .16])
    np.testing.assert_array_equal(minimum_tracking_height(heights), [.22]*4)
    np.testing.assert_allclose(minimum_tracking_height(heights, low_support=True), [.22, .22, .14, .10])
    with pytest.raises(ValueError, match="Nonfinite"):
        minimum_tracking_height(float("nan"), low_support=True)
    np.testing.assert_array_equal(minimum_tracking_upright([1, .2, 0, -.2]), [.2]*4)
    np.testing.assert_allclose(minimum_tracking_upright([1, .2, 0, -.2], low_support=True),
                               [.2, -.15, -.25, -.25])


def test_library_rejects_a_kneeling_family_rename_without_audit(tmp_path):
    from k1_motion.learning import MotionLibrary
    robot = K1Model()
    row = {"capture_group": "fixture/kneeling", "family": "kneel", "model_signature": robot.signature,
           "kinematics_accepted": True, "split": "train", "reference_path": "clip.npz"}
    MotionClip.from_references([robot.neutral_reference(i*.02) for i in range(4)], row).save(tmp_path/"clip.npz")
    (tmp_path/"index.jsonl").write_text(json.dumps(row)+"\n")
    with pytest.raises(ValueError, match="Low support"):
        MotionLibrary(tmp_path, robot, "cpu")


def test_genuine_knee_support_round_trips_through_library_and_physics(tmp_path):
    import torch
    from k1_motion.reference_admission import admit_reference
    from k1_motion.tracking_env import LOW_SUPPORT_TASK_VERSION, TrackerEnv
    frames = kneeling_frames()
    robot = K1Model()
    clip, _ = retarget_low_pose(robot, frames, {"family": "kneel", "capture_group": "fixture/kneel"})
    clip, _ = correct_low_pose_ground(clip)
    audit = audit_low_pose(clip, frames, "kneel")
    assert audit["accepted"] and audit["knee_phase_recall"] == 1
    row = admit_reference({**clip.metadata, "recovery_audit": audit,
        "low_pose_reference_audit": low_pose_reference_audit(audit), "kinematics_accepted": True,
        "rejected_ticks": 0, "frames": len(clip.times), "split": "train", "reference_path": "clip.npz"})
    clip.metadata.update(row)
    clip.save(tmp_path/"clip.npz")
    (tmp_path/"index.jsonl").write_text(json.dumps(row)+"\n")
    env = TrackerEnv(tmp_path, num_envs=1, device="cpu", reference_storage="packed")
    zeros = torch.zeros(1, dtype=torch.long)
    env.reset(clips=zeros, frames=zeros)
    ref = env.library.frames(zeros, zeros)
    assert env.task_version == LOW_SUPPORT_TASK_VERSION
    assert ref["knee_contacts"].eq(1).all() and ref["contacts"].eq(0).all()
    for _ in range(5):
        _, _, reward, _, _ = env.step(torch.zeros(1, 22), auto_reset=False)
        assert torch.isfinite(reward).all()
        assert not env.last_step["fallen"].any()
    actual = env.physics.robots[0]
    knees = {actual.model.body(f"{side}_knee_pitch_link").id for side in ("left", "right")}
    assert any(0 in actual.model.geom_bodyid[[c.geom1, c.geom2]]
               and any(b in knees for b in actual.model.geom_bodyid[[c.geom1, c.geom2]])
               for c in actual.data.contact[:actual.data.ncon])
    env.close()
