"""Causal reference playback with intentional buffering and original packet clocks.

Robot feedback and actuator commands never enter this buffer. A fixed source to
wall-clock mapping prevents arrival jitter from changing playback speed.
"""
from collections import deque
from dataclasses import dataclass

import numpy as np

from .contracts import Reference

PREVIEW_OFFSETS = (5, 10, 15)
PREVIEW_DT = .02
PLAYBACK_DELAY_S = .3
MAX_PREVIEW_HOLD_S = .04
_EPS = 1e-8


@dataclass(frozen=True)
class BufferedReference:
    current: Reference
    future: tuple[Reference | None, Reference | None, Reference | None]
    mask: tuple[bool, bool, bool]
    playback_time: float
    latest_received_time: float
    current_sample_age: float
    draining: bool
    completed: bool

    @property
    def playback_source_time(self):
        """Compatibility name; phase is in the declared scheduling clock."""
        return self.playback_time


class PreviewBuffer:
    """Delay playback300ms from first arrival; expose only already-arrived frames.

    ``finish`` is an explicit finite-recording end, never inferred from silence.
    It permits only the bounded remaining playback tail. A live stream that
    stops arriving still uses the controller's unchanged input-loss watchdog.
    """
    def __init__(self, delay_s=PLAYBACK_DELAY_S, horizon_s=.3, capacity=256):
        if delay_s != PLAYBACK_DELAY_S or horizon_s not in (0., .3):
            raise ValueError('Preview requires300ms playback delay and0 or300ms horizon')
        if not isinstance(capacity, int) or capacity < 17:
            raise ValueError('Preview buffer capacity must retain at least17 frames')
        self.delay_s, self.horizon_s, self.capacity = float(delay_s), float(horizon_s), capacity
        self.reset()

    def reset(self, anchor_time=None):
        if anchor_time is not None and not np.isfinite(anchor_time):
            raise ValueError('Preview playback anchor must be finite')
        self.frames = deque(maxlen=self.capacity)
        self.playback_origin = self.session = None
        self.arrival_origin = anchor_time
        self.finished_at = None

    def push(self, reference, playback_time=None):
        explicit_schedule = playback_time is not None
        playback_time = reference.source_time if playback_time is None else float(playback_time)
        if not np.isfinite(playback_time):
            raise ValueError('Reference playback time must be finite')
        if self.finished_at is not None:
            raise ValueError('Reference stream already finished; reset before another recording')
        if self.session is not None and reference.session != self.session:
            raise ValueError('Reference session changed; reset and recalibrate preview buffer')
        if self.frames:
            previous_time, previous = self.frames[-1]
            if reference.source_time < previous.source_time or (not explicit_schedule
                                                               and reference.source_time == previous.source_time):
                return False
            if playback_time <= previous_time:
                raise ValueError('Reference playback schedule must advance')
            if reference.received_time < previous.received_time:
                raise ValueError('Reference arrival clock must be monotonic')
        else:
            self.playback_origin = playback_time
            if self.arrival_origin is None:
                self.arrival_origin = reference.received_time
            self.session = reference.session
        self.frames.append((playback_time, reference))
        return True

    def finish(self):
        if not self.frames:
            raise ValueError('Cannot finish an empty reference stream')
        self.finished_at = self.frames[-1][1].received_time

    def sample(self, now):
        if not np.isfinite(now):
            raise ValueError('Preview clock must be finite')
        if not self.frames or now < self.arrival_origin+self.delay_s-_EPS:
            return None
        phase = self.playback_origin+now-self.arrival_origin-self.delay_s
        available = [(tick, ref) for tick, ref in self.frames if ref.received_time <= now+_EPS]
        eligible = [(tick, ref) for tick, ref in available if tick <= phase+_EPS]
        if not eligible:
            # Fail closed when a too-small buffer has discarded the playback phase.
            return None
        _, current = eligible[-1]
        future = []
        for offset in PREVIEW_OFFSETS:
            requested = phase+offset*PREVIEW_DT
            candidates = [(tick, ref) for tick, ref in available if tick <= requested+_EPS]
            selected = candidates[-1] if candidates else None
            # Resample30/60Hz capture causally. An older sample may represent a
            # requested slot only after the arrived stream reaches that phase;
            # never extrapolate beyond the newest arrived source/scheduled tick.
            present = (offset*PREVIEW_DT <= self.horizon_s+_EPS and selected is not None
                       and requested <= available[-1][0]+_EPS
                       and requested-selected[0] <= MAX_PREVIEW_HOLD_S+_EPS and selected[1].valid)
            future.append(selected[1] if present else None)
        end_received = self.finished_at is not None and now >= self.finished_at-_EPS
        completed = bool(end_received and phase >= self.frames[-1][0]-_EPS)
        return BufferedReference(current, tuple(future), tuple(frame is not None for frame in future),
            float(phase), float(available[-1][1].received_time), max(0., float(now-self.delay_s-current.received_time)),
            bool(end_received and not completed), completed)
