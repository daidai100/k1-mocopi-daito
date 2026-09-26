from dataclasses import replace

import numpy as np
import pytest
import torch

from k1_motion.contracts import CONVENTIONS, JOINT_NAMES, MotionClip, HumanFrame
from k1_motion.corpus import recording_split
from k1_motion.observations import ObservationBuilder
from k1_motion.retarget import Retargeter
from k1_motion.robot import K1Model
from k1_motion.runtime import CommandOwner, Controller, Mode
from test_input import decoder, skeleton
from k1_motion.mocopi import synthetic_packet


@pytest.fixture(scope="module")
def robot():
    torch.set_num_threads(1)
    return K1Model()


def armed(robot):
    robot.reset()
    c = Controller(robot)
    c.calibrate(robot.neutral_reference(), robot.state(0), True, 0)
    c.arm(robot.state(0), 0)
    return c


def test_loss_is_latched_and_airborne_loss_damps(robot):
    c = armed(robot)
    for now in np.arange(0, 0.21, 0.02):
        command = c.tick(robot.state(now), now, supported=True)
    assert command.mode == Mode.PAUSED
    c.set_reference(robot.neutral_reference(0.22))
    assert c.tick(robot.state(0.22), 0.22, supported=True).mode == Mode.PAUSED
    with pytest.raises(RuntimeError):
        c.arm(robot.state(0.22), 0.22)
    c = armed(robot)
    for now in np.arange(0, 0.21, 0.02):
        command = c.tick(robot.state(now), now, supported=False)
    assert command.mode == Mode.FAULT and command.damping_only


def test_stale_state_invalid_policy_and_stop(robot):
    c = armed(robot)
    assert c.tick(robot.state(-1), 0, True).mode == Mode.FAULT
    c = armed(robot)
    c.policy = lambda _: np.full(22, np.nan)
    assert c.tick(robot.state(0), 0, True).damping_only
    c = armed(robot)
    c.stop(0)
    assert c.tick(robot.state(0), 0, True).mode == Mode.STOPPED


def test_recalibration_resets_history_without_target_jump(robot):
    c = armed(robot)
    c.tick(robot.state(0), 0, True)
    c.pause(0.01)
    for now in np.arange(0.02, 0.5, 0.02):
        c.tick(robot.state(now), now, True)
    old = c.previous_target.copy()
    c.calibrate(replace(robot.neutral_reference(0.5), session=1), robot.state(0.5), True, 0.5)
    assert torch.count_nonzero(c.builder.history.frames) == 0
    np.testing.assert_array_equal(old, c.previous_target)
    c.arm(robot.state(0.5), 0.5)
    command = c.tick(robot.state(0.5), 0.5, True)
    assert np.max(abs(command.targets - old)) <= 0.12 + 1e-7


def test_command_lock_excludes_another_owner(tmp_path):
    with CommandOwner(tmp_path / "owner"):
        with pytest.raises(RuntimeError):
            CommandOwner(tmp_path / "owner")
    with CommandOwner(tmp_path / "owner"):
        pass


def test_causality_same_prefix_different_future(robot):
    torch.manual_seed(3)
    policy = torch.nn.Sequential(torch.nn.Linear(544, 22), torch.nn.Tanh())

    def run(future):
        c = armed(robot)
        c.policy = lambda o: policy(torch.tensor(o)).detach().numpy()
        outputs = []
        for i in range(10):
            t = i * 0.02
            ref = robot.neutral_reference(t)
            if i >= 7:
                ref = replace(ref, joint_position=ref.joint_position + future)
            if i:
                c.set_reference(ref)
            outputs.append(c.tick(robot.state(t), t, True).targets)
        return np.array(outputs)

    a, b = run(0.1), run(-0.1)
    np.testing.assert_array_equal(a[:7], b[:7])
    assert not np.allclose(a[7:], b[7:])


def test_retarget_prefix_causality_and_reset_derivatives(robot):
    d = decoder()
    human = [d.ingest(synthetic_packet(skeleton(), i, i * 20), i * 0.02) for i in range(8)]
    refs = []
    for future in (0.1, -0.1):
        r = Retargeter(robot)
        r.calibrate(human[0])
        row = []
        for i, frame in enumerate(human):
            if i >= 6:
                p = frame.positions.copy()
                p[5, 0] += future
                frame = replace(frame, positions=p)
            row.append(r.process(frame).joint_position)
        refs.append(np.array(row))
        moved = replace(human[0], positions=human[0].positions + [5, 0, 0], session=2)
        r.calibrate(moved)
        fresh = r.process(moved)
        np.testing.assert_array_equal(fresh.joint_velocity, 0)
        np.testing.assert_array_equal(fresh.root_velocity, 0)
    np.testing.assert_array_equal(refs[0][:6], refs[1][:6])


def test_stance_toe_marker_offset_does_not_tilt_robot_soles(robot):
    from scipy.spatial.transform import Rotation

    # A human marker well below the ankle must not become a robot tiptoe pose.
    positions = robot.neutral_landmarks * 1.8
    positions[[12, 16], 2] -= 0.08
    frame = HumanFrame(0, 0, 0, positions, np.tile([1.0, 0, 0, 0], (17, 1)), "synthetic/toe-markers")
    retarget = Retargeter(robot)
    retarget.calibrate(frame)
    ref = retarget.process(frame)
    ids = [robot.model.body(side + "_ankle_roll_link").id for side in ("left", "right")]
    angles = Rotation.from_matrix(retarget.data.xmat[ids].reshape(-1, 3, 3)).as_euler("xyz")
    assert ref.valid and np.all(ref.contacts == 1)
    assert np.max(np.abs(angles[:, :2])) < 0.03


def test_contact_projection_preserves_planted_feet_under_pelvis_motion(robot):
    positions = robot.neutral_landmarks * 1.8
    orientation = np.tile([1.0, 0, 0, 0], (17, 1))
    retarget = Retargeter(robot)
    first = HumanFrame(0, 0, 0, positions, orientation, "synthetic/planted-shift")
    retarget.calibrate(first)
    references = []
    for i in range(30):
        moved = positions.copy()
        moved[:11, 0] += i * 0.002
        moved[13:15, 0] += i * 0.002
        references.append(
            retarget.process(HumanFrame(i * 0.02, i * 0.02, i, moved, orientation, "synthetic/planted-shift"))
        )
    feet = np.stack([r.landmarks[[11, 15], :2] for r in references])
    speed = np.linalg.norm(np.diff(feet, axis=0), axis=-1) / 0.02
    assert all(r.valid and np.all(r.contacts == 1) for r in references)
    assert np.percentile(speed, 95) < 0.02
    q = np.stack([r.joint_position for r in references])
    assert np.max(abs(np.diff(q, axis=0))) <= robot.config["command_velocity_limit"] * 0.02 + 1e-8


def test_motion_roundtrip_masks_order_and_parent_split(robot, tmp_path):
    refs = [robot.neutral_reference(0), replace(robot.neutral_reference(0.02), valid=False)]
    clip = MotionClip.from_references(refs, {"capture_group": "source/parent"})
    path = tmp_path / "clip.npz"
    clip.save(path)
    loaded = MotionClip.load(path)
    assert not loaded.frame(1).valid
    assert loaded.causal_index(0.019) == 0
    assert recording_split("source/parent") == recording_split(loaded.metadata["capture_group"])
    with pytest.raises(ValueError, match="joint order"):
        MotionClip(
            clip.times, clip.values, {"joint_names": list(reversed(JOINT_NAMES)), "conventions": CONVENTIONS}
        )


def test_balanced_standing_and_effort_bounds(robot):
    c = armed(robot)
    for i in range(250):
        t = i * 0.02
        if i:
            c.set_reference(robot.neutral_reference(t))
        cmd = c.tick(robot.state(t), t, True)
        robot.step(cmd.targets)
        assert robot.data.qpos[2] > 0.45
        assert np.all(np.abs(robot.data.ctrl) <= robot.effort)


def test_observation_excludes_future_and_simulator_truth(robot):
    builder = ObservationBuilder(robot.neutral)
    observation = builder.build(robot.state(0), robot.neutral_reference(), np.zeros(22), 0)
    assert observation.shape == (544,)
    assert np.count_nonzero(observation[:408]) == 0
    assert observation[-1] == 1
