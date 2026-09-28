"""The capture watchdog must tell "nothing is playing" from "the device died".

A previous version timed the callbacks and reopened the device after 6 s of
silence. It passed a hand test -- run while a meeting was playing -- and then
reopened the device every 6 seconds forever on a quiet machine, because
WASAPI loopback delivers exactly ZERO callbacks while the render endpoint is
idle. Measured: 0 callbacks in 12 s with nothing playing, while is_active()
kept reporting True.

So: silence is not a fault, and these tests encode that.
"""
import time

import numpy as np
import pytest

from app import config
from app.audio.capture import AudioSource, DeviceInfo
from app.pipeline import AUDIO_CHECK_S, Pipeline


class FakeSource(AudioSource):
    """A capture stream whose liveness we control."""

    def __init__(self, alive: bool = True) -> None:
        self._alive = alive
        self._info = DeviceInfo(0, "fake loopback", 2, 48000, True)
        self.starts = 0
        self.stops = 0

    def start(self, callback) -> None:
        self.starts += 1

    def stop(self) -> None:
        self.stops += 1

    @property
    def info(self) -> DeviceInfo:
        return self._info

    def is_alive(self) -> bool:
        return self._alive

    def die(self) -> None:
        self._alive = False


@pytest.fixture
def pipe():
    p = Pipeline(config.load(root="."))
    p._ready = True
    return p


def _tick(p: Pipeline) -> None:
    """Force the rate-limited check to run."""
    p._last_check_at = 0.0
    p._check_audio_alive()


def test_silent_but_live_stream_is_not_reopened(pipe):
    """The bug: a quiet machine must not churn the audio device."""
    src = FakeSource(alive=True)
    pipe._source = src
    pipe._last_audio_at = time.monotonic() - 3600     # an hour of silence

    for _ in range(10):
        _tick(pipe)

    assert src.stops == 0, "a live stream was torn down just for being quiet"


def test_dead_stream_is_reopened(pipe, monkeypatch):
    reopened = []
    monkeypatch.setattr(pipe, "switch_source",
                        lambda *a, **k: reopened.append(True))
    src = FakeSource(alive=True)
    pipe._source = src

    _tick(pipe)
    assert not reopened, "live stream should be left alone"

    src.die()
    _tick(pipe)
    assert reopened == [True], "a stopped stream must be reopened"


def test_reopen_is_attempted_once_per_death(pipe, monkeypatch):
    """No 6-second reopen loop, even if the reopen does not help."""
    reopened = []
    monkeypatch.setattr(pipe, "switch_source",
                        lambda *a, **k: reopened.append(True))
    src = FakeSource(alive=False)
    pipe._source = src

    for _ in range(20):
        _tick(pipe)

    assert len(reopened) == 1, f"reopened {len(reopened)} times -- looping again"


def test_recovery_rearms_the_watchdog(pipe, monkeypatch):
    reopened = []
    monkeypatch.setattr(pipe, "switch_source",
                        lambda *a, **k: reopened.append(True))
    src = FakeSource(alive=False)
    pipe._source = src

    _tick(pipe)
    assert len(reopened) == 1

    src._alive = True            # the reopen worked
    _tick(pipe)
    src.die()                    # ...and it dies again later
    _tick(pipe)
    assert len(reopened) == 2, "a second, separate failure must be handled"


def test_paused_pipeline_is_left_alone(pipe, monkeypatch):
    reopened = []
    monkeypatch.setattr(pipe, "switch_source",
                        lambda *a, **k: reopened.append(True))
    pipe._source = FakeSource(alive=False)
    pipe._paused = True

    _tick(pipe)
    assert not reopened, "pausing stops the stream on purpose"


def test_check_is_rate_limited(pipe, monkeypatch):
    calls = []
    src = FakeSource(alive=True)
    monkeypatch.setattr(src, "is_alive", lambda: calls.append(1) or True)
    pipe._source = src

    pipe._last_check_at = 0.0
    pipe._check_audio_alive()          # runs
    pipe._check_audio_alive()          # too soon
    pipe._check_audio_alive()          # too soon
    assert len(calls) == 1, "watchdog should not poll the backend every frame"
    assert AUDIO_CHECK_S >= 0.5


def test_every_backend_implements_is_alive():
    """The contract is abstract, so a new backend cannot forget it."""
    import inspect

    from app.audio import soundcard_src, wasapi

    for cls in (wasapi.WasapiSource, soundcard_src.SoundCardSource):
        assert "is_alive" in cls.__dict__, f"{cls.__name__} must implement it"
        assert not inspect.isabstract(cls)


def test_source_is_published_only_after_it_starts(monkeypatch, pipe):
    """The watchdog must never see a source that has not been started yet.

    Assigning self._source before calling start() left a window where
    is_alive() was False for a perfectly good stream; the watchdog raced into
    it during a source switch and logged a spurious "stream stopped".
    """
    seen_during_start = []
    src = FakeSource(alive=False)        # not "alive" until start() runs

    def fake_start(callback):
        # While start() is in flight, the pipeline must not have published it.
        seen_during_start.append(pipe._source)
        src._alive = True
        src.starts += 1

    monkeypatch.setattr(src, "start", fake_start)
    monkeypatch.setattr("app.pipeline.open_source", lambda *a, **k: src)

    pipe._source = None
    pipe._open_audio()

    assert seen_during_start == [None], \
        "pipeline published the source before it was started"
    assert pipe._source is src
    assert src.is_alive()
