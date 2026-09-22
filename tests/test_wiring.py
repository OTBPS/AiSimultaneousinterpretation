"""Wiring and wakeup behaviour.

Two classes of regression are pinned here.

1. The eager endpoint was dead for the entire life of the project: the
   segmenter implemented it, the config exposed it, the README described it,
   and nothing ever called set_partial_text(). It cannot be allowed to rot
   again, so the connection itself is a test.

2. Both worker loops used to poll (250 Hz and 200 Hz) and were rewritten to
   block on an event. A silent regression to polling would not fail any
   functional test, only the battery -- so the wakeup contract is tested
   directly.

These need the real config and the Silero model on disk, same as
tests/test_vad.py.
"""
import threading
import time

import numpy as np
import pytest

from app import config
from app.bus import EventBus
from app.gpu_worker import GpuWorker
from app.metrics import Metrics
from app.types import Segment
from app.vad import FRAME_SAMPLES


@pytest.fixture
def cfg():
    return config.load(root=".")


def a_segment(seq=0):
    return Segment(seq=seq, audio=np.zeros(16000, dtype=np.float32),
                   duration_s=1.0, mean_speech_prob=0.9,
                   t_speech_end=time.perf_counter())


# ------------------------------------------------------- worker wakeups
def test_submit_segment_wakes_the_worker(cfg):
    w = GpuWorker(cfg, EventBus(), Metrics())
    assert not w._wake.is_set(), "should start idle"
    w.submit_segment(a_segment())
    assert w._wake.is_set(), "a queued segment must wake the loop, not be polled for"


def test_stop_wakes_a_blocked_worker(cfg):
    w = GpuWorker(cfg, EventBus(), Metrics())
    w._wake.clear()
    w._stop.set()
    w._wake.set()                     # what stop() does
    assert w._wake.is_set()


def test_worker_loop_blocks_when_idle(cfg):
    """An idle loop must not spin: 0.3 s of nothing should cost ~1 wakeup."""
    w = GpuWorker(cfg, EventBus(), Metrics())
    waits = []
    real_wait = w._wake.wait

    def counting_wait(timeout=None):
        waits.append(timeout)
        return real_wait(timeout)

    w._wake.wait = counting_wait
    t = threading.Thread(target=w._run, daemon=True)
    t.start()
    time.sleep(0.3)
    w._stop.set()
    w._wake.set()
    t.join(timeout=2)

    assert waits, "idle loop never blocked -- it is polling again"
    # 0.3 s at the 0.25 s ceiling is at most a couple of waits; a poller
    # would have looped ~60 times.
    assert len(waits) <= 5, f"too many wakeups while idle: {len(waits)}"
    assert all(x == 0.25 for x in waits)


# --------------------------------------------------- eager endpoint wiring
def test_pipeline_feeds_partial_text_to_segmenter(cfg):
    from app.pipeline import Pipeline

    p = Pipeline(cfg)
    assert p.worker.on_partial is not None, "partial callback not wired"

    p.on_partial_text("This is a complete sentence.")
    assert p.segmenter._partial_text == "This is a complete sentence."


def test_worker_partial_callback_reaches_the_segmenter(cfg):
    """End to end through the worker's own dispatch, without a real model."""
    from app.pipeline import Pipeline
    from app.gpu_worker import PartialJob

    p = Pipeline(cfg)

    class FakeAsr:
        def transcribe(self, audio, duration_s=0.0, mean_prob=1.0):
            return "Ready to ship."

    p.worker.asr = FakeAsr()
    p.worker._do_asr(PartialJob(np.zeros(16000, dtype=np.float32)))
    assert p.segmenter._partial_text == "Ready to ship."


def test_eager_endpoint_actually_fires_once_wired(cfg):
    """The whole point: a terminated partial must shorten the hangover."""
    from app.segmenter import FRAME_MS, Segmenter, SegmenterParams

    params = SegmenterParams(endpoint_silence_ms=550, eager_punct_ms=250,
                             min_speech_ms=100)
    eager_frames = int(250 / FRAME_MS)
    frame = np.full(FRAME_SAMPLES, 0.1, dtype=np.float32)

    seg = Segmenter(params)
    for _ in range(30):
        seg.push(frame, 0.9)
    seg.set_partial_text("Ready to ship.")     # what the wiring now delivers
    committed = None
    for _ in range(eager_frames + 1):
        s = seg.push(frame, 0.01)
        if s is not None:
            committed = s
    assert committed is not None, "eager endpoint did not fire with a wired partial"


# ------------------------------------------------------ pipeline wakeups
def test_vad_frame_read_does_not_spin(cfg):
    from app.pipeline import Pipeline

    p = Pipeline(cfg)
    t0 = time.perf_counter()
    assert p._next_frame() is None, "no audio written yet"
    elapsed = time.perf_counter() - t0
    # A poller returns in ~4 ms; the blocking version waits out its timeout.
    assert elapsed >= 0.09, f"returned in {elapsed*1000:.0f} ms -- polling again?"


def test_audio_callback_wakes_the_vad_thread(cfg):
    from app.pipeline import Pipeline

    p = Pipeline(cfg)
    p._data_ready.clear()
    p._on_audio(np.zeros(FRAME_SAMPLES, dtype=np.float32))
    assert p._data_ready.is_set()
    assert p._next_frame() is not None, "written audio should be readable"


def test_partial_asr_is_off_by_default(cfg):
    """It doubles iGPU power (2.46 W -> 5.02 W) for a preview line."""
    assert cfg.asr.partial_interval_ms == 0


# ------------------------------------------------- UI drain rate switching
#
# These call app.ui.overlay.is_activity directly. An earlier version of this
# file re-implemented the rule inside the test, which would have passed even
# with the real code broken -- exactly the bug being pinned here.
def test_metrics_tick_alone_does_not_count_as_activity():
    from app.types import MetricsEvent
    from app.ui.overlay import is_activity

    assert is_activity(MetricsEvent(backlog=0)) is False


def test_backlog_counts_as_activity():
    from app.types import MetricsEvent
    from app.ui.overlay import is_activity

    assert is_activity(MetricsEvent(backlog=3)) is True


def test_subtitle_events_count_as_activity():
    from app.types import LineDoneEvent, LineStartEvent, TokenEvent
    from app.ui.overlay import is_activity

    for ev in (TokenEvent(1, "你好"),
               LineStartEvent(1, "hello"),
               LineDoneEvent(1, "你好")):
        assert is_activity(ev) is True


def test_metrics_tick_does_not_cancel_real_activity():
    """The bug: `saw = ev.backlog > 0` overwrote an earlier True, dropping
    the UI to 5 Hz while subtitles were still streaming."""
    from app.types import MetricsEvent, TokenEvent
    from app.ui.overlay import is_activity

    saw = False
    for ev in (TokenEvent(1, "你好"), MetricsEvent(backlog=0)):
        saw |= is_activity(ev)
    assert saw is True
