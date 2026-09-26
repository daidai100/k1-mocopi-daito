"""Test-first failure list: future leakage, clock conflation, stale robot feedback,
unbounded end drain, session/history leakage, mask/shape drift, and dishonest
weight-transfer or export metadata. Tests use the real K1 controller/model.
"""
from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from k1_motion.export import export_checkpoint
from k1_motion.learning import ActorCritic, Policy
from k1_motion.model_transfer import initialize_model, input_mapping
from k1_motion.observations import ObservationBuilder, observation_contract, reference_tensor
from k1_motion.robot import K1Model
from k1_motion.runtime import Controller, Mode


@pytest.fixture
def robot():
    torch.set_num_threads(1)
    return K1Model()


def reference(robot, index, *, origin=0., arrival_origin=0.):
    ref = robot.neutral_reference(origin+index*.02)
    return replace(ref, received_time=arrival_origin+index*.02,
                   joint_position=ref.joint_position+index*.001,
                   joint_velocity=np.ones(22)*.05)


def test_buffer_separates_clocks_and_masks_unarrived_and_out_of_range_future(robot):
    from k1_motion.preview import PreviewBuffer
    buffer = PreviewBuffer()
    refs = [reference(robot, i, origin=10., arrival_origin=2.) for i in range(17)]
    for ref in refs:
        buffer.push(ref)
    assert buffer.sample(2.299) is None
    sample = buffer.sample(2.3)
    assert sample.current is refs[0]
    assert sample.current.received_time == 2.0
    assert sample.latest_received_time == 2.3
    assert sample.playback_source_time == pytest.approx(10.)
    assert sample.current_sample_age == pytest.approx(0.)
    assert [r.source_time for r in sample.future] == pytest.approx([10.1, 10.2, 10.3])
    assert sample.mask == (True, True, True)
    assert buffer.sample(2.32).future[-1] is refs[16]
    later = buffer.sample(2.34)
    assert later.current is refs[2]
    assert later.mask == (True, True, False)
    assert later.future[-1] is None
    # Late old source packets cannot roll back playback or prolong input freshness.
    assert not buffer.push(replace(refs[1], received_time=2.4))
    assert buffer.sample(2.4).latest_received_time == pytest.approx(2.32)


def test_buffer_zero_preview_same_playback_and_bounded_explicit_finish(robot):
    from k1_motion.preview import PreviewBuffer
    masked, preview = PreviewBuffer(horizon_s=0.), PreviewBuffer(horizon_s=.3)
    for i in range(21):
        for buffer in (masked, preview):
            buffer.push(reference(robot, i))
    for buffer in (masked, preview):
        buffer.finish()
    assert masked.sample(.4).current.source_time == preview.sample(.4).current.source_time
    assert masked.sample(.4).mask == (False, False, False)
    assert preview.sample(.4).mask == (True, True, True)
    assert preview.sample(.68).draining
    assert preview.sample(.68).mask == (False, False, False)
    assert preview.sample(.7).completed
    assert not preview.sample(.72).draining
    with pytest.raises(ValueError, match='finished'):
        preview.push(reference(robot, 22))
    preview.reset()
    preview.push(replace(reference(robot, 30), session='new'))
    assert preview.sample(.89) is None
    assert preview.sample(.9).current.session == 'new'
    with pytest.raises(ValueError, match='session'):
        preview.push(replace(reference(robot, 31), session='other'))


def test_buffer_preserves_capture_clocks_with_repeated_source_and_playback_ticks(robot):
    from k1_motion.preview import PreviewBuffer
    buffer = PreviewBuffer()
    scheduled = []
    for index in range(17):
        tick = index*.02
        #30Hz capture resampled to50Hz; sourceclock starts10 seconds ahead.
        capture = np.floor((tick+1e-10)*30)/30
        ref = replace(reference(robot, index), source_time=10+capture, received_time=2+capture)
        scheduled.append(ref)
        buffer.push(ref, playback_time=tick)
    at_first = buffer.sample(2.3)
    assert at_first.current is scheduled[0]
    assert at_first.future[0] is scheduled[5]
    at_next = buffer.sample(2.32)
    assert at_next.current is scheduled[1]
    assert at_next.current.source_time == 10
    assert at_next.current.received_time == 2
    assert at_next.playback_time == pytest.approx(.02)
    assert at_next.current_sample_age == pytest.approx(.02)
    assert at_next.latest_received_time == pytest.approx(2.3)


@pytest.mark.parametrize('hz', [30, 60])
def test_live_cadence_preview_is_bounded_causal_hold_with_arrival_jitter(robot, hz):
    from k1_motion.preview import PreviewBuffer
    buffer = PreviewBuffer()
    frames = []
    for index in range(hz):
        source = index/hz
        arrival = source+.01+(index % 2)*.005
        ref = replace(reference(robot, index), source_time=source, received_time=arrival)
        frames.append(ref)
        buffer.push(ref)
    now = .35
    window = buffer.sample(now)
    assert window.mask[:2] == (True, True)
    for offset, present, ref in zip((.1, .2, .3), window.mask, window.future):
        if present:
            requested = window.playback_time+offset
            assert ref.received_time <= now
            assert 0 <= requested-ref.source_time <= .04+1e-8
            assert any(r.source_time >= requested and r.received_time <= now for r in frames)
    # The full300ms target is beyond the newest captured+arrived sample here.
    assert window.mask[-1] is False


def test_preview_observation_scalar_batch_parity_and_mask_nonleakage(robot):
    from k1_motion.observations import append_preview_tensor
    from k1_motion.preview import PreviewBuffer
    buffer = PreviewBuffer()
    for i in range(16):
        buffer.push(reference(robot, i))
    window = buffer.sample(.3)
    builder = ObservationBuilder(robot.neutral, history=3, profile='preview')
    state = robot.state(.3)
    observed = builder.build(state, window.current, np.zeros(22), .3, preview=window)
    contract = observation_contract(3, 'preview')
    assert contract['frame_size'] == 155
    assert contract['size'] == 3*156+120
    assert contract['future_frames'] == 3
    assert observed.shape == (contract['size'],)
    planar = ObservationBuilder(robot.neutral, history=3, profile='planar').build(
        state, window.current, np.zeros(22), .3, frame_age=window.current_sample_age)
    batch = append_preview_tensor(torch.tensor(planar)[None],
        torch.tensor(np.array(state.joint_position), dtype=torch.float32)[None],
        torch.tensor(np.array(state.orientation), dtype=torch.float32)[None],
        torch.tensor(np.array(state.root_position), dtype=torch.float32)[None],
        [reference_tensor(r) for r in window.future], torch.tensor([window.mask]))
    np.testing.assert_allclose(observed, batch[0], atol=1e-7)
    masks = replace(window, future=(None, None, None), mask=(False, False, False))
    builder.reset()
    zeros = builder.build(state, window.current, np.zeros(22), .3, preview=masks)
    assert np.count_nonzero(zeros[-120:]) == 0
    assert zeros[2*156+129] == 0  # current phase sample age, not300ms arrival age
    with pytest.raises(ValueError, match='preview'):
        builder.build(state, window.current, np.zeros(22), .3)
    with pytest.raises(ValueError, match='odometry'):
        builder.build(replace(state, root_position=None), window.current, np.zeros(22), .3, preview=window)


class CapturePolicy:
    def __init__(self, horizon=.3):
        self.metadata = {'observation': observation_contract(3, 'preview', preview_horizon_s=horizon),
                         'action_settings': {'target_velocity_scale': .25}}
        self.seen = []

    def __call__(self, observed):
        self.seen.append(observed.copy())
        return np.zeros(22)


def armed_preview(robot, horizon=.3):
    policy = CapturePolicy(horizon)
    controller = Controller(robot, policy)
    controller.calibrate(reference(robot, 0), robot.state(0), True, 0)
    with pytest.raises(RuntimeError, match='buffer'):
        controller.arm(robot.state(.28), .28)
    for i in range(1, 16):
        controller.set_reference(reference(robot, i))
    controller.arm(robot.state(.3), .3)
    return controller, policy


def test_runtime_keeps_current_feedback_and_feedforward_but_watchdog_is_real(robot):
    controller, policy = armed_preview(robot)
    state = replace(robot.state(.3), joint_position=robot.neutral+.05)
    command = controller.tick(state, .3, supported=True)
    assert command.mode == Mode.ACTIVE
    np.testing.assert_allclose(command.velocities, .0125)
    np.testing.assert_allclose(policy.seen[-1][2*156:2*156+22], .05, atol=1e-7)
    assert policy.seen[-1][-1] == 1
    # Fresh proprioception cannot hide actual input loss; existing deadline remains.
    for now in [.34, .38, .42, .46]:
        command = controller.tick(robot.state(now), now, supported=False)
    assert command.mode == Mode.FAULT and command.damping_only
    controller, _ = armed_preview(robot)
    assert controller.tick(robot.state(.1), .3, supported=True).mode == Mode.FAULT


def test_runtime_calibration_anchor_preserves_nonzero_initial_packet_age(robot):
    controller = Controller(robot, CapturePolicy())
    first = replace(reference(robot, 0), received_time=-.02)
    controller.calibrate(first, robot.state(0), True, 0, playback_time=0.)
    for index in range(1, 16):
        controller.set_reference(reference(robot, index), playback_time=index*.02)
    controller.arm(robot.state(.3), .3)
    window = controller.preview_buffer.sample(.3)
    assert window.current is first
    assert window.playback_time == pytest.approx(0.)
    assert window.current_sample_age == pytest.approx(.02)


def test_preview_receive_clock_roundoff_does_not_trigger_input_loss(robot):
    controller, _ = armed_preview(robot)
    controller.tick(robot.state(.3), .3)
    for index in (16, 17):
        controller.set_reference(reference(robot, index))
    now = .3+2*.02  #0.33999999999999997 versus receive17*.02 ==0.34
    assert now < reference(robot, 17).received_time
    assert controller.tick(robot.state(now), now).mode == Mode.ACTIVE


def test_runtime_tail_drains_only_when_finished_then_stops_and_session_resets(robot):
    controller, _ = armed_preview(robot)
    controller.finish_reference()
    for now in np.arange(.3, .6, .02):
        assert controller.tick(robot.state(now), float(now), supported=True).mode == Mode.ACTIVE
    command = controller.tick(robot.state(.6), .6, supported=True)
    assert command.mode == Mode.STOPPED and command.damping_only
    controller, _ = armed_preview(robot)
    controller.tick(robot.state(.3), .3, supported=True)
    controller.set_reference(replace(reference(robot, 16), session='changed'))
    assert controller.mode == Mode.PAUSED
    assert torch.count_nonzero(controller.builder.history.frames) == 0
    assert controller.preview_buffer.sample(.32) is None


@pytest.mark.parametrize('old_profile', ['causal', 'planar'])
def test_preview_expansion_export_reload_and_contract_tamper_rejection(tmp_path, robot, old_profile):
    old_contract = observation_contract(2, old_profile)
    new_contract = observation_contract(3, 'preview')
    old = ActorCritic(old_contract['size'], old_contract['size']+17, (32, 16))
    old.actor.normalizer.mean.normal_()
    old.actor.normalizer.variance.uniform_(.5, 2.)
    checkpoint = {'actor_size': old_contract['size'], 'critic_size': old_contract['size']+17,
                  'observation': old_contract, 'model': old.state_dict(), 'hidden_sizes': [32, 16]}
    new = ActorCritic(new_contract['size'], new_contract['size']+17, (32, 16))
    initialize_model(new, checkpoint, new_contract)
    mapping = input_mapping(old_contract['size'], new_contract['size'], old_contract, new_contract, 'cpu')
    sample = torch.randn(16, new_contract['size'])
    torch.testing.assert_close(new.actor(sample), old.actor(sample[:, mapping]), atol=2e-5, rtol=1e-6)
    assert torch.count_nonzero(new.actor.network[0].weight[:, -120:]) == 0
    saved = {**checkpoint, 'actor_size': new_contract['size'], 'critic_size': new_contract['size']+17,
             'model': new.state_dict(), 'observation': new_contract, 'stage': 'student',
             'model_signature': robot.signature, 'train_parents': [], 'iteration': 0}
    path = tmp_path/'checkpoint.pt'
    torch.save(saved, path)
    receipt = export_checkpoint(path, tmp_path/'actor.pt')
    assert receipt['finite'] and receipt['export_reload_max_error'] == 0
    deployed = Policy(tmp_path/'actor.pt', robot.signature)
    np.testing.assert_allclose(deployed(sample[0].numpy()), new.actor(sample[:1]).tanh()[0].detach(), atol=1e-7)
    metadata_path = tmp_path/'actor.json'
    metadata = json.loads(metadata_path.read_text())
    metadata['observation']['preview_offsets'] = [1, 2, 3]
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='compatible'):
        Policy(tmp_path/'actor.pt', robot.signature)
    malformed = {**saved, 'observation': {**new_contract, 'size': new_contract['size']-1}}
    torch.save(malformed, path)
    with pytest.raises(ValueError, match='observation'):
        export_checkpoint(path, tmp_path/'bad.pt')
