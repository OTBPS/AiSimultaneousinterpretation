"""Shared data types and UI events.

The backend (pipeline) only ever emits the events defined here; the frontend
(CLI or Qt overlay) only ever consumes them. Neither imports the other. That
boundary is what lets the same pipeline drive a terminal and a GUI unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class LineState(str, Enum):
    PARTIAL = "partial"          # English hypothesis, still being spoken
    TRANSLATING = "translating"  # Chinese tokens streaming in
    DONE = "done"
    SKIPPED = "skipped"          # dropped by backpressure


@dataclass(slots=True)
class Segment:
    """A committed span of speech, ready for ASR."""
    seq: int
    audio: np.ndarray            # float32 mono @ 16 kHz
    duration_s: float
    mean_speech_prob: float
    is_continuation: bool = False
    t_speech_end: float = 0.0    # perf_counter() when the endpoint fired


@dataclass(slots=True)
class Line:
    """One unit of translation shown as a subtitle line."""
    line_id: int
    source: str = ""
    target: str = ""
    state: LineState = LineState.PARTIAL
    is_continuation: bool = False
    t_speech_end: float = 0.0
    t_first_token: float = 0.0
    t_done: float = 0.0


# ---------------------------------------------------------------- UI events

@dataclass(slots=True)
class StatusEvent:
    """Backend lifecycle / health, rendered as a status line."""
    state: str                   # loading | ready | paused | error | stopped
    detail: str = ""


@dataclass(slots=True)
class PartialEvent:
    """In-progress English hypothesis. Never translated, feedback only."""
    text: str


@dataclass(slots=True)
class LineStartEvent:
    line_id: int
    source: str
    is_continuation: bool = False


@dataclass(slots=True)
class TokenEvent:
    """One chunk of streamed Chinese output."""
    line_id: int
    text: str


@dataclass(slots=True)
class LineDoneEvent:
    line_id: int
    target: str
    ttft_ms: float = 0.0
    total_ms: float = 0.0


@dataclass(slots=True)
class LineSkippedEvent:
    line_id: int
    source: str
    reason: str = "backlog"


@dataclass(slots=True)
class MetricsEvent:
    """Rolling pipeline health, emitted every few seconds."""
    backlog: int = 0
    ttft_p95_ms: float = 0.0
    line_p95_ms: float = 0.0
    dropped: int = 0
    using_fast_model: bool = False


UiEvent = (StatusEvent | PartialEvent | LineStartEvent | TokenEvent
           | LineDoneEvent | LineSkippedEvent | MetricsEvent)
