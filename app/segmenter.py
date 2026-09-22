"""Turn a stream of VAD frames into committed speech segments.

This is where perceived latency is actually decided. The endpoint hangover
(`endpoint_silence_ms`) dominates the end-to-end budget -- ASR is ~320 ms and
MT first-token is ~173 ms, so a 550 ms hangover is the single largest term.
It is the primary tuning knob and is exposed in config.yaml.

Deliberately a pure state machine: it takes frames and returns segments, with
no threads, no audio device and no model. That makes the latency behaviour
unit-testable (tests/test_segmenter.py) instead of something you can only
observe by talking at the laptop.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from .types import Segment
from .vad import FRAME_SAMPLES, SAMPLE_RATE

FRAME_MS = FRAME_SAMPLES / SAMPLE_RATE * 1000.0      # 32 ms


@dataclass(slots=True)
class SegmenterParams:
    threshold: float = 0.5
    endpoint_silence_ms: int = 550
    eager_punct_ms: int = 250
    min_speech_ms: int = 300
    max_segment_s: float = 8.0
    preroll_ms: int = 300
    onset_frames: int = 3            # consecutive voiced frames to start


class Segmenter:
    """Feed 512-sample frames + their VAD probability; get Segments back."""

    def __init__(self, params: SegmenterParams | None = None) -> None:
        self.p = params or SegmenterParams()
        self._preroll: deque[np.ndarray] = deque(
            maxlen=max(1, int(self.p.preroll_ms / FRAME_MS)))
        self._hangover_frames = max(1, int(self.p.endpoint_silence_ms / FRAME_MS))
        self._eager_frames = max(1, int(self.p.eager_punct_ms / FRAME_MS))
        self._max_frames = int(self.p.max_segment_s * 1000 / FRAME_MS)
        self._min_frames = max(1, int(self.p.min_speech_ms / FRAME_MS))
        self.reset()

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._speaking = False
        self._onset_run = 0
        self._silence_run = 0
        self._frames: list[np.ndarray] = []
        self._probs: list[float] = []
        self._preroll.clear()
        self._partial_text = ""
        self._next_seq = getattr(self, "_next_seq", 0)
        self._pending_continuation = False
        self._last_voiced_at = 0.0

    def set_partial_text(self, text: str) -> None:
        """Latest English hypothesis; enables the eager punctuation endpoint."""
        self._partial_text = text or ""

    @property
    def speaking(self) -> bool:
        return self._speaking

    @property
    def current_ms(self) -> float:
        return len(self._frames) * FRAME_MS

    def current_audio(self) -> np.ndarray:
        """Audio accumulated so far (for partial ASR while still speaking)."""
        if not self._frames:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self._frames)

    # ------------------------------------------------------------------
    def push(self, frame: np.ndarray, prob: float) -> Segment | None:
        voiced = prob >= self.p.threshold

        if not self._speaking:
            self._preroll.append(frame)
            if voiced:
                self._onset_run += 1
                if self._onset_run >= self.p.onset_frames:
                    self._start()
            else:
                self._onset_run = 0
            return None

        self._frames.append(frame)
        self._probs.append(prob)

        if voiced:
            self._silence_run = 0
            # The moment the speaker actually stopped -- NOT the moment we
            # decided they had. Latency must be measured from here, otherwise
            # the endpoint hangover (the largest term in the budget) hides
            # itself from its own metric.
            self._last_voiced_at = time.perf_counter()
        else:
            self._silence_run += 1

        # Hard cap: a speaker who never pauses must still produce output.
        if len(self._frames) >= self._max_frames:
            return self._commit_hard_cut()

        if self._silence_run == 0:
            return None

        if self._silence_run >= self._hangover_frames:
            return self._commit()

        # Eager endpoint: ASR already produced a terminated sentence, so the
        # remaining hangover buys nothing but latency.
        if (self._silence_run >= self._eager_frames
                and self._partial_text.rstrip().endswith((".", "?", "!"))):
            return self._commit()

        return None

    # ------------------------------------------------------------------
    def _start(self) -> None:
        self._speaking = True
        self._silence_run = 0
        self._onset_run = 0
        self._frames = list(self._preroll)        # pre-roll keeps the first phoneme
        self._probs = []
        self._preroll.clear()

    def _commit_hard_cut(self) -> Segment | None:
        """Split a speaker who never pauses, at the quietest recent moment.

        Cutting bluntly at the cap slices mid-word and produces fragments like
        "hardware." on their own, which the translator then renders as a
        standalone sentence. Cutting at the local VAD minimum inside the last
        ~500 ms lands on a breath or a word gap far more often.
        """
        look = min(len(self._probs), max(2, int(500 / FRAME_MS)))
        tail_probs = self._probs[-look:]
        offset = len(self._probs) - look
        cut = offset + int(min(range(look), key=lambda i: tail_probs[i]))
        cut = max(1, min(cut + 1, len(self._frames) - 1))

        carry_frames = self._frames[cut:]
        carry_probs = self._probs[cut:]
        self._frames = self._frames[:cut]
        self._probs = self._probs[:cut]

        seg = self._commit(is_continuation=True)

        # Resume immediately: the speaker never actually stopped.
        self._speaking = True
        self._frames = carry_frames
        self._probs = carry_probs
        self._silence_run = 0
        return seg

    def _commit(self, is_continuation: bool = False) -> Segment | None:
        frames, probs = self._frames, self._probs
        was_continuation = self._pending_continuation

        self._speaking = False
        self._frames, self._probs = [], []
        self._silence_run = 0
        self._onset_run = 0
        self._partial_text = ""
        self._pending_continuation = is_continuation

        voiced = [p for p in probs if p >= self.p.threshold]
        if len(voiced) < self._min_frames:
            return None                            # a cough, a click, a keystroke

        audio = np.concatenate(frames) if frames else np.zeros(0, dtype=np.float32)
        seg = Segment(
            seq=self._next_seq,
            audio=audio,
            duration_s=len(audio) / SAMPLE_RATE,
            mean_speech_prob=float(np.mean(probs)) if probs else 0.0,
            is_continuation=was_continuation,
            t_speech_end=self._last_voiced_at or time.perf_counter(),
        )
        self._next_seq += 1
        return seg

    def flush(self) -> Segment | None:
        """Commit whatever is buffered (stream ended / pausing)."""
        return self._commit() if self._speaking else None
