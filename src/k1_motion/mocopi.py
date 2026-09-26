"""Strict mocopi TLV decoder and a single-owner, latest-frame UDP receiver.

Format and axis mapping reviewed against the recovered mocopi adapter, originally
based on seagetch/mcp-receiver (MIT, copyright 2022 seagetch). This decoder is a
new bounded implementation; no ROS or robot SDK is imported.
"""

from collections import Counter
from dataclasses import dataclass
import socket
import struct
import threading
import time

import numpy as np

from .contracts import HumanFrame, LANDMARKS
from .math3d import anatomical_rotation, quaternion, rotation, unit_quat

NAMES = (
    "root",
    *(f"torso_{i}" for i in range(1, 8)),
    "neck_1",
    "neck_2",
    "head",
    "l_shoulder",
    "l_up_arm",
    "l_low_arm",
    "l_hand",
    "r_shoulder",
    "r_up_arm",
    "r_low_arm",
    "r_hand",
    "l_up_leg",
    "l_low_leg",
    "l_foot",
    "l_toes",
    "r_up_leg",
    "r_low_leg",
    "r_foot",
    "r_toes",
)
PARENTS = (-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 7, 11, 12, 13, 7, 15, 16, 17, 0, 19, 20, 21, 0, 23, 24, 25)
MAP = (0, 7, 10, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26)


def fields(data):
    """Size counts payload bytes, not the 8-byte little-endian TLV header."""
    result = []
    offset = 0
    while offset < len(data):
        if len(data) - offset < 8:
            raise ValueError("Truncated TLV header")
        size, tag = struct.unpack_from("<I4s", data, offset)
        offset += 8
        if not tag.isalpha() or size > len(data) - offset:
            raise ValueError("Invalid TLV tag or payload length")
        result.append((tag, data[offset : offset + size]))
        offset += size
    return result


def unique(data):
    result = {}
    for key, value in fields(data):
        if key in result:
            raise ValueError(f"Duplicate field {key!r}")
        result[key] = value
    return result


def unpack(fmt, data):
    if len(data) != struct.calcsize(fmt):
        raise ValueError("Incorrect scalar payload size")
    return struct.unpack(fmt, data)


@dataclass(frozen=True)
class Packet:
    kind: str
    transforms: np.ndarray
    parents: tuple | None = None
    frame_number: int | None = None
    timestamp_ms: int | None = None


def decode_packet(raw):
    if len(raw) > 65507 or len(raw) < 16:
        raise ValueError("Invalid UDP packet length")
    try:
        top = unique(raw)
        header = unique(top[b"head"])
        if not header[b"ftyp"] or unpack("<B", header[b"vrsn"])[0] != 1:
            raise ValueError("Unsupported mocopi header/version")
        if (b"skdf" in top) == (b"fram" in top):
            raise ValueError("Expected exactly one skeleton or frame")
        skeleton = b"skdf" in top
        body = unique(top[b"skdf" if skeleton else b"fram"])
        items = fields(body[b"bons" if skeleton else b"btrs"])
        if len(items) != 27:
            raise ValueError("Expected all 27 mocopi bones")
        transforms = np.zeros((27, 7))
        parents = [-2] * 27
        seen = set()
        for _, payload in items:
            bone = unique(payload)
            bone_id = unpack("<H", bone[b"bnid"])[0]
            if bone_id not in range(27) or bone_id in seen:
                raise ValueError("Unknown or duplicate bone ID")
            seen.add(bone_id)
            values = np.array(unpack("<7f", bone[b"tran"]))
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite bone transform")
            values[:4] = np.roll(unit_quat(np.roll(values[:4], 1)), -1)
            transforms[bone_id] = values
            if skeleton:
                parent = unpack("<H", bone[b"pbid"])[0]
                parents[bone_id] = -1 if parent == 65535 else parent
        if skeleton:
            if tuple(parents) != PARENTS:
                raise ValueError("Unsupported mocopi skeleton hierarchy")
            return Packet("skeleton", transforms, tuple(parents))
        return Packet(
            "frame",
            transforms,
            frame_number=unpack("<I", body[b"fnum"])[0],
            timestamp_ms=unpack("<I", body[b"time"])[0],
        )
    except KeyError as exc:
        raise ValueError(f"Missing mocopi field {exc.args[0]!r}") from exc


def to_human(packet, source_seconds, received_time, session):
    # Sony/source y-up -> canonical z-up, cyclic permutation with determinant +1.
    local_p = packet.transforms[:, [6, 4, 5]]
    local_r = rotation(packet.transforms[:, [3, 2, 0, 1]])
    global_p = np.zeros((27, 3))
    global_r = []
    for i, parent in enumerate(PARENTS):
        if parent < 0:
            global_p[i] = local_p[i]
            global_r.append(local_r[i])
        else:
            global_p[i] = global_p[parent] + global_r[parent].apply(local_p[i])
            global_r.append(global_r[parent] * local_r[i])
    positions = global_p[list(MAP)]
    orientations = np.stack([quaternion(global_r[i]) for i in MAP])
    orientations[0] = quaternion(anatomical_rotation(positions, LANDMARKS))
    return HumanFrame(
        source_seconds, received_time, packet.frame_number, positions, orientations, "mocopi_udp", session
    )


class StreamDecoder:
    def __init__(self):
        self.session = -1
        self.skeleton = None
        self.counts = Counter()
        self.new_session()

    def new_session(self):
        """Explicit app restart. Caller must pause/recalibrate before resuming."""
        self.session += 1
        self.last_number = self.last_ms = self.last_received = None
        self.elapsed_ms = 0
        self.skeleton = None

    def ingest(self, raw, received_time):
        if not np.isfinite(received_time):
            raise ValueError("Invalid receive clock")
        packet = decode_packet(raw)
        if packet.kind == "skeleton":
            if self.skeleton is not None and not np.allclose(self.skeleton.transforms, packet.transforms):
                self.new_session()
            self.skeleton = packet
            self.counts["skeleton"] += 1
            return None
        if self.skeleton is None:
            raise ValueError("Frame arrived before skeleton; send skeleton or begin a new capture")
        if self.last_number is not None:
            number_delta = (packet.frame_number - self.last_number) & 0xFFFFFFFF
            time_delta = (packet.timestamp_ms - self.last_ms) & 0xFFFFFFFF
            if not 0 < number_delta < 0x80000000 or not 0 < time_delta < 0x80000000:
                raise ValueError(
                    "Duplicate/out-of-order frame or app restart; explicit session reset required"
                )
            if received_time < self.last_received:
                raise ValueError("Receive clock moved backwards")
        else:
            number_delta, time_delta = 1, packet.timestamp_ms
        elapsed = self.elapsed_ms + time_delta
        frame = to_human(packet, elapsed * 0.001, received_time, self.session)
        self.elapsed_ms = elapsed
        self.last_number, self.last_ms, self.last_received = (
            packet.frame_number,
            packet.timestamp_ms,
            received_time,
        )
        self.counts["accepted"] += 1
        self.counts["missing_source_frames"] += number_delta - 1
        return frame


class UdpReceiver:
    """One socket, no SO_REUSEPORT, pinned sender, one replaceable complete frame."""

    def __init__(self, host="0.0.0.0", port=12351, recorder=None):
        self.decoder = StreamDecoder()
        self.recorder = recorder
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.socket.bind((host, port))
        except OSError:
            self.socket.close()
            raise
        self.socket.settimeout(0.05)
        self.port = self.socket.getsockname()[1]
        self.peer = None
        self._latest = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name="mocopi-receiver", daemon=True)
        self.thread.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                raw, peer = self.socket.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                if self._stop.is_set():
                    break
                self.decoder.counts["socket_error"] += 1
                self._stop.set()
                break
            now = time.monotonic()
            status, frame = "accepted", None
            try:
                if self.peer is not None and peer != self.peer:
                    raise ValueError("Unexpected UDP sender")
                with self._lock:
                    frame = self.decoder.ingest(raw, now)
                    self.peer = peer
                    if frame is not None:
                        self._latest = frame
            except ValueError as exc:
                status = str(exc)
                self.decoder.counts["rejected"] += 1
            if self.recorder:
                self.recorder.packet(raw, now, peer, frame, status)

    def latest(self):
        with self._lock:
            return self._latest

    def reset_session(self):
        with self._lock:
            now = time.monotonic()
            self.decoder.new_session()
            self.peer = self._latest = None
            if self.recorder:
                self.recorder.event("session_reset", now)

    def close(self):
        self._stop.set()
        self.thread.join(timeout=1.0)
        self.socket.close()
        if self.thread.is_alive():
            raise RuntimeError("Receiver thread did not terminate")


def synthetic_packet(transforms, frame_number=0, timestamp_ms=0, skeleton=False):
    """Fixture encoder only; these bytes are not evidence of a live Sony stream."""

    def tlv(tag, value):
        return struct.pack("<I4s", len(value), tag) + value

    head = tlv(b"head", tlv(b"ftyp", b"mocopi") + tlv(b"vrsn", b"\x01"))
    bones = b""
    for i, transform in enumerate(transforms):
        item = tlv(b"bnid", struct.pack("<H", i))
        if skeleton:
            item += tlv(b"pbid", struct.pack("<H", PARENTS[i] & 65535))
        item += tlv(b"tran", struct.pack("<7f", *transform))
        bones += tlv(b"bone" if skeleton else b"btrn", item)
    if skeleton:
        return head + tlv(b"skdf", tlv(b"bons", bones))
    return head + tlv(
        b"fram",
        tlv(b"fnum", struct.pack("<I", frame_number & 0xFFFFFFFF))
        + tlv(b"time", struct.pack("<I", timestamp_ms & 0xFFFFFFFF))
        + tlv(b"btrs", bones),
    )
