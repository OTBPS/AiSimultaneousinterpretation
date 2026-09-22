"""Segmenter behaviour, driven by synthetic VAD probabilities.

No audio device and no model: we feed the state machine a probability sequence
and assert on the segments it commits. That makes the latency-critical rules
(hangover, eager endpoint, hard cap, minimum speech) testable in milliseconds.
"""
import numpy as np
import pytest

from app.segmenter import FRAME_MS, Segmenter, SegmenterParams
from app.vad import FRAME_SAMPLES


def frame(level=0.1):
    return np.full(FRAME_SAMPLES, level, dtype=np.float32)


def run(seg, pattern, partial=None):
    """pattern: iterable of bools (voiced?). Returns committed segments."""
    out = []
    for voiced in pattern:
        if partial is not None:
            seg.set_partial_text(partial)
        s = seg.push(frame(), 0.9 if voiced else 0.01)
        if s is not None:
            out.append(s)
    return out


def ms(n):
    return n * FRAME_MS


def test_commits_after_hangover():
    p = SegmenterParams(endpoint_silence_ms=550, min_speech_ms=300)
    seg = Segmenter(p)
    hang = int(550 / FRAME_MS)
    # 40 voiced frames (~1.28 s) then silence
    out = run(seg, [True] * 40 + [False] * (hang + 1))
    assert len(out) == 1
    assert out[0].duration_s > 1.0
    assert out[0].seq == 0


def test_does_not_commit_before_hangover():
    seg = Segmenter(SegmenterParams(endpoint_silence_ms=550))
    hang = int(550 / FRAME_MS)
    out = run(seg, [True] * 40 + [False] * (hang - 2))
    assert out == []
    assert seg.speaking


def test_eager_endpoint_fires_earlier_than_hangover():
    p = SegmenterParams(endpoint_silence_ms=550, eager_punct_ms=250)
    eager_frames = int(250 / FRAME_MS)

    slow = Segmenter(p)
    fast = Segmenter(p)
    tail = [False] * (eager_frames + 1)

    assert run(slow, [True] * 40 + tail, partial="no terminator here") == []
    got = run(fast, [True] * 40 + tail, partial="This is a full sentence.")
    assert len(got) == 1, "eager endpoint should commit on a terminated partial"


def test_short_blip_is_rejected():
    p = SegmenterParams(min_speech_ms=300, endpoint_silence_ms=200)
    seg = Segmenter(p)
    hang = int(200 / FRAME_MS)
    # 4 voiced frames = 128 ms, below the 300 ms floor
    out = run(seg, [True] * 4 + [False] * (hang + 1))
    assert out == []


def test_hard_cap_splits_and_marks_continuation():
    p = SegmenterParams(max_segment_s=1.0, endpoint_silence_ms=550)
    seg = Segmenter(p)
    n = int(1.0 * 1000 / FRAME_MS)
    out = run(seg, [True] * (n * 2 + 5))
    assert len(out) >= 2
    # the first cut is not a continuation; the one after it is
    assert out[0].is_continuation is False
    assert out[1].is_continuation is True


def test_preroll_keeps_audio_before_onset():
    p = SegmenterParams(preroll_ms=320, onset_frames=3, endpoint_silence_ms=200)
    seg = Segmenter(p)
    hang = int(200 / FRAME_MS)
    out = run(seg, [False] * 20 + [True] * 30 + [False] * (hang + 1))
    assert len(out) == 1
    # 30 voiced frames alone would be ~0.96 s; pre-roll must add to that
    assert out[0].duration_s > 30 * FRAME_MS / 1000


def test_sequence_numbers_increment():
    p = SegmenterParams(endpoint_silence_ms=200, min_speech_ms=100)
    seg = Segmenter(p)
    hang = int(200 / FRAME_MS)
    burst = [True] * 20 + [False] * (hang + 1)
    out = run(seg, burst * 3)
    assert [s.seq for s in out] == [0, 1, 2]


def test_flush_commits_in_flight_speech():
    seg = Segmenter(SegmenterParams(endpoint_silence_ms=550, min_speech_ms=100))
    run(seg, [True] * 40)
    assert seg.speaking
    s = seg.flush()
    assert s is not None and not seg.speaking
    assert seg.flush() is None


def test_t_speech_end_is_last_voiced_not_commit_time():
    """The hangover must not hide inside its own latency metric."""
    import time
    p = SegmenterParams(endpoint_silence_ms=550, min_speech_ms=100)
    seg = Segmenter(p)
    hang = int(550 / FRAME_MS)

    run(seg, [True] * 20)
    t_after_speech = time.perf_counter()
    time.sleep(0.05)                       # stand in for the silent hangover
    out = run(seg, [False] * (hang + 1))

    assert len(out) == 1
    committed_at = time.perf_counter()
    # t_speech_end belongs to the voiced part, well before the commit
    assert out[0].t_speech_end <= t_after_speech + 1e-3
    assert committed_at - out[0].t_speech_end >= 0.05


def test_hard_cut_resumes_speaking_and_carries_audio():
    p = SegmenterParams(max_segment_s=1.0, endpoint_silence_ms=550,
                        min_speech_ms=100)
    seg = Segmenter(p)
    n = int(1.0 * 1000 / FRAME_MS)
    out = run(seg, [True] * (n + 2))
    assert len(out) == 1
    assert seg.speaking, "speaker never paused, so we must keep accumulating"
    assert seg.current_ms > 0, "carried-over audio must not be discarded"


def test_hard_cut_prefers_quiet_frame():
    """The cut should land on the low-probability frame, not the hard cap."""
    p = SegmenterParams(max_segment_s=1.0, endpoint_silence_ms=550,
                        min_speech_ms=100, threshold=0.5)
    seg = Segmenter(p)
    n = int(1.0 * 1000 / FRAME_MS)
    out = None
    for i in range(n + 2):
        # one clear dip 5 frames before the cap
        prob = 0.55 if i == n - 5 else 0.95
        s = seg.push(frame(), prob)
        if s is not None:
            out = s
    assert out is not None
    # the dip is inside the last 500ms window, so the cut lands at/near it
    cut_frames = round(out.duration_s * 1000 / FRAME_MS)
    assert cut_frames <= n - 3
