"""Lossless raw UDP journals, with replay through the live StreamDecoder."""

import base64
import json
from pathlib import Path
import threading
import time

from .contracts import CONVENTIONS
from .mocopi import StreamDecoder


class Recorder:
    def __init__(self, path, metadata=None, source_kind="live_mocopi_udp"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("x", buffering=1)
        self.lock = threading.Lock()
        self._write(
            {
                "type": "metadata",
                "format": "k1-udp-v1",
                "source_kind": source_kind,
                "conventions": CONVENTIONS,
                "session": metadata or {},
            }
        )

    def _write(self, value):
        with self.lock:
            self.stream.write(json.dumps(value, allow_nan=False) + "\n")

    def packet(self, raw, received_time, peer, frame=None, status="accepted"):
        self._write(
            {
                "type": "packet",
                "received_time": received_time,
                "peer": list(peer),
                "raw_base64": base64.b64encode(raw).decode("ascii"),
                "status": status,
                "source_time": frame.source_time if frame else None,
                "frame_number": frame.frame_number if frame else None,
                "session": frame.session if frame else None,
            }
        )

    def event(self, name, received_time, metadata=None):
        self._write(
            {"type": "event", "name": name, "received_time": received_time, "metadata": metadata or {}}
        )

    def close(self):
        with self.lock:
            self.stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def replay(path, realtime=False, speed=1.0, decoder=None):
    if speed <= 0:
        raise ValueError("Replay speed must be positive")
    decoder = decoder or StreamDecoder()
    start_rx, start_wall = None, time.monotonic()
    peer = None
    with Path(path).open() as stream:
        header = json.loads(next(stream))
        if header.get("format") != "k1-udp-v1":
            raise ValueError("Not a raw UDP recording")
        for line in stream:
            row = json.loads(line)
            if row["type"] == "event":
                if row["name"] == "session_reset":
                    decoder.new_session()
                    peer = None
                continue
            if row["type"] != "packet":
                continue
            rx = float(row["received_time"])
            if start_rx is None:
                start_rx = rx
            if realtime:
                delay = start_wall + (rx - start_rx) / speed - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
            try:
                incoming_peer = tuple(row["peer"])
                if peer is not None and incoming_peer != peer:
                    raise ValueError("Unexpected UDP sender")
                frame = decoder.ingest(base64.b64decode(row["raw_base64"], validate=True), rx)
                peer = incoming_peer
            except ValueError:
                decoder.counts["rejected"] += 1
                continue
            if frame is not None:
                yield frame
