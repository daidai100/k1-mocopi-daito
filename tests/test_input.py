import socket
import time

import numpy as np
import pytest

from k1_motion.adapters import Bvh
from k1_motion.mocopi import StreamDecoder, UdpReceiver, decode_packet, synthetic_packet
from k1_motion.recording import Recorder, replay


def skeleton():
    t = np.zeros((27, 7), dtype=float)
    t[:, 3] = 1
    t[0, 5] = 1.0
    t[1:11, 5] = 0.05
    t[[11, 12, 13, 14], 4] = [0.1, 0.05, 0.25, 0.25]
    t[[15, 16, 17, 18], 4] = [-0.1, -0.05, -0.25, -0.25]
    t[19, 4], t[23, 4] = 0.1, -0.1
    t[[20, 21, 24, 25], 5] = -0.45
    t[[22, 26], 6] = 0.12
    return t


def decoder():
    d = StreamDecoder()
    d.ingest(synthetic_packet(skeleton(), skeleton=True), 0.0)
    return d


def test_truncation_duplicate_missing_nonfinite():
    packet = synthetic_packet(skeleton(), 7, 100)
    for end in range(0, len(packet), 7):
        with pytest.raises(ValueError):
            decode_packet(packet[:end])
    bad = skeleton()
    bad[5, 4] = np.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        decode_packet(synthetic_packet(bad))
    with pytest.raises(ValueError, match="27"):
        decode_packet(synthetic_packet(skeleton()[:-1]))


def test_order_wrap_and_explicit_restart():
    d = decoder()
    first = d.ingest(synthetic_packet(skeleton(), 0xFFFFFFFE, 0xFFFFFFF0), 10.0)
    next_frame = d.ingest(synthetic_packet(skeleton(), 1, 20), 10.05)
    assert next_frame.source_time - first.source_time == pytest.approx(0.036)
    assert d.counts["missing_source_frames"] == 2
    with pytest.raises(ValueError, match="out-of-order"):
        d.ingest(synthetic_packet(skeleton(), 1, 20), 10.06)
    with pytest.raises(ValueError, match="restart"):
        d.ingest(synthetic_packet(skeleton(), 0, 0), 12.0)
    d.new_session()
    with pytest.raises(ValueError, match="before skeleton"):
        d.ingest(synthetic_packet(skeleton()), 12.0)
    d.ingest(synthetic_packet(skeleton(), skeleton=True), 12.0)
    assert d.ingest(synthetic_packet(skeleton()), 12.0).session == 1


def test_immutable_complete_pose_and_both_clocks():
    f = decoder().ingest(synthetic_packet(skeleton(), 4, 1500), 77.0)
    assert (f.source_time, f.received_time) == (1.5, 77.0)
    assert f.positions[3, 1] > f.positions[6, 1]  # left remains left
    assert f.positions[2, 2] > f.positions[0, 2]  # head remains up
    with pytest.raises(ValueError):
        f.positions[0, 0] = 2


def test_record_replay_uses_same_decoder_and_preserves_rejects(tmp_path):
    path = tmp_path / "capture.jsonl"
    d = StreamDecoder()
    expected = []
    packets = [
        synthetic_packet(skeleton(), skeleton=True),
        synthetic_packet(skeleton(), 1, 10),
        b"broken",
        synthetic_packet(skeleton(), 2, 30),
        synthetic_packet(skeleton(), 2, 30),
    ]
    with Recorder(path, {"calibration": "none"}, "synthetic_fixture") as recorder:
        for i, raw in enumerate(packets):
            try:
                frame = d.ingest(raw, i * 0.02)
                if frame:
                    expected.append(frame)
            except ValueError:
                frame = None
            recorder.packet(raw, i * 0.02, ("127.0.0.1", 1000), frame)
    actual = list(replay(path))
    assert len(actual) == len(expected) == 2
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a.positions, b.positions)
        assert (a.source_time, a.received_time) == (b.source_time, b.received_time)
    assert len(path.read_text().splitlines()) == 6


def test_socket_exclusive_latest_and_sender_pinned():
    receiver = UdpReceiver("127.0.0.1", 0)
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        with pytest.raises(OSError):
            UdpReceiver("127.0.0.1", receiver.port)
        sender.sendto(synthetic_packet(skeleton(), skeleton=True), ("127.0.0.1", receiver.port))
        for i in range(20):
            sender.sendto(synthetic_packet(skeleton(), i, i * 20), ("127.0.0.1", receiver.port))
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline and (
            receiver.latest() is None or receiver.latest().frame_number != 19
        ):
            time.sleep(0.005)
        assert receiver.latest().frame_number == 19
        assert receiver.latest().received_time <= time.monotonic()
    finally:
        sender.close()
        receiver.close()


def test_bvh_nonroot_translations_and_rotation_order(tmp_path):
    path = tmp_path / "translation.bvh"
    path.write_text("""HIERARCHY
ROOT Hips { OFFSET 100 100 100 CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
JOINT Child { OFFSET 2 0 0 CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
End Site { OFFSET 0 0 1 } } }
MOTION
Frames: 2
Frame Time: 0.02
1 2 3 90 0 0 2 0 0 0 0 0
1 2 3 90 0 0 4 0 0 0 0 0
""")
    bvh = Bvh.load(path)
    p, _ = bvh.fk("replace")
    np.testing.assert_allclose(p[:, 0], [[1, 2, 3], [1, 2, 3]])
    np.testing.assert_allclose(p[:, 1], [[1, 4, 3], [1, 6, 3]], atol=1e-10)
