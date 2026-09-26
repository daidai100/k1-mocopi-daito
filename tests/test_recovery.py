import importlib.util
import io
import json
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import pytest

from k1_motion.contracts import HumanFrame, MotionClip
from k1_motion.recovery_validation import audit_recovery, retarget_recovery
from k1_motion.retarget_recovery import RecoveryRetargeter
from k1_motion.robot import K1Model


def human_fixture():
    robot = K1Model()
    times = np.array([0, .016666, .033332, .058331, .074997, .099996, .116662])
    positions = np.tile(robot.neutral_landmarks * 1.6, (len(times), 1, 1))
    positions[:, 5, 0] += np.array([0, .15, -.15, .15, -.15, .15, -.15])
    return {"times": times, "positions": positions,
            "orientations": np.tile([1., 0., 0., 0.], (len(times), 17, 1))}


def test_recovery_uses_control_clock_and_preserves_source_clocks():
    robot = K1Model()
    original_margins = robot.model.geom_margin.copy()
    human = human_fixture()
    clip, _ = retarget_recovery(robot, human, {"source_motion_id": "fixture"})
    speed = np.abs(np.diff(clip.values["joint_position"], axis=0) / np.diff(clip.times)[:, None])
    assert speed.max() <= 6.0 + 1e-6
    assert speed.max() > 5.9  # Actually exercises a saturated command.
    assert np.all(clip.received_times <= clip.times)
    assert set(clip.source_times) <= set(human["times"])
    assert np.array_equal(robot.model.geom_margin, original_margins)


def test_recovery_is_causal_and_resets_between_motions():
    human = human_fixture()
    full, _ = retarget_recovery(K1Model(), human, {"source_motion_id": "fixture"})
    prefix = {key: value[:4] for key, value in human.items()}
    short, _ = retarget_recovery(K1Model(), prefix, {"source_motion_id": "fixture"})
    assert np.array_equal(full.values["joint_position"][:len(short.times)], short.values["joint_position"])
    again, _ = retarget_recovery(K1Model(), human, {"source_motion_id": "fixture"})
    assert np.array_equal(full.values["joint_position"], again.values["joint_position"])


def test_recovery_rejects_bad_control_clock():
    human = human_fixture()
    retargeter = RecoveryRetargeter(K1Model())
    frame = HumanFrame(0., 0., 0, human["positions"][0], human["orientations"][0], "fixture")
    retargeter.calibrate(frame)
    retargeter.process(frame, control_time=0.)
    with pytest.raises(ValueError, match="Nonmonotonic recovery control clock"):
        retargeter.process(replace(frame, source_time=.02, received_time=.02), control_time=0.)


def test_audit_detects_collision_between_clear_endpoints():
    fixture = json.loads((Path(__file__).parent / "fixtures/k1_between_tick_collision.json").read_text())
    robot = K1Model()
    references = []
    for index, pose in enumerate(np.array(fixture["qpos"])):
        robot.data.qpos[:] = pose
        mujoco.mj_forward(robot.model, robot.data)
        assert not any(c.dist < -.0001 and 0 not in robot.model.geom_bodyid[[c.geom1, c.geom2]]
                       for c in robot.data.contact[:robot.data.ncon])
        references.append(replace(robot.neutral_reference(index * .02), root_position=pose[:3],
                                  root_orientation=pose[3:7], joint_position=pose[7:],
                                  landmarks=robot.landmarks()))
    clip = MotionClip.from_references(references, {"model_signature": robot.signature})
    audit = audit_recovery(clip, clip, [{"rms_landmark_error_m": 0.}], {"rms_landmark_error_m": 0.})
    assert not audit["accepted"]
    assert "self_collision_on_command_path" in audit["rejection_reasons"]
    assert audit["self_collision_samples"] > 0
    assert not audit["physics_qualified"]
    assert not audit["training_eligible"]


def test_following_ledger_waits_for_complete_final_line():
    script = Path(__file__).parents[1] / "scripts/recover_bones_seed.py"
    spec = importlib.util.spec_from_file_location("recover_bones_seed", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    stream = io.StringIO('{"id": "first"}\n{"id": "sec')
    assert list(module.read_complete_lines(stream)) == [{"id": "first"}]
    assert stream.tell() == len('{"id": "first"}\n')

