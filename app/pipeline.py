"""Orchestrator: owns the threads, wires audio -> VAD -> GPU -> bus.

Thread map (see gpu_worker for why the GPU side is single-threaded):

    T1  PortAudio callback      -> RingBuffer            (real-time context)
    T2  vad-loop                -> Segmenter -> GpuWorker queues
    T3  gpu-worker              -> EventBus
    T4  whoever calls bus.drain (CLI or Qt)              frontend

The frontend is not referenced anywhere here. That is the point: `run.py --cli`
and `run.py --gui` drive the exact same pipeline.
"""
from __future__ import annotations

import logging
import threading
import time

import numpy as np

from .audio.capture import AudioSource, CaptureError, open_source
from .audio.ring import RingBuffer
from .bus import EventBus
from .config import Config
from .gpu_worker import GpuWorker
from .metrics import Metrics
from .segmenter import Segmenter, SegmenterParams
from .types import StatusEvent
from .vad import FRAME_SAMPLES, SAMPLE_RATE, SileroVad

log = logging.getLogger("pipeline")

RING_SECONDS = 30


class Pipeline:
    def __init__(self, cfg: Config, bus: EventBus | None = None) -> None:
        self.cfg = cfg
        self.bus = bus or EventBus()
        self.metrics = Metrics(endpoint_ms=cfg.vad.endpoint_silence_ms)

        self.ring = RingBuffer(SAMPLE_RATE * RING_SECONDS)
        self.vad = SileroVad(cfg.vad.model)
        self.segmenter = Segmenter(SegmenterParams(
            threshold=cfg.vad.threshold,
            endpoint_silence_ms=cfg.vad.endpoint_silence_ms,
            eager_punct_ms=cfg.vad.eager_punct_ms,
            min_speech_ms=cfg.vad.min_speech_ms,
            max_segment_s=cfg.vad.max_segment_s,
            preroll_ms=cfg.vad.preroll_ms,
        ))
        self.worker = GpuWorker(cfg, self.bus, self.metrics)
        self.worker.on_partial = self.on_partial_text

        self._source: AudioSource | None = None
        self._vad_thread: threading.Thread | None = None
        self._stop = threading.Event()
        # Signalled by the audio callback so the VAD thread can block instead
        # of polling. Polling every 4 ms is 250 wakeups/second that mostly
        # find nothing, and sustained sub-millisecond timer activity keeps the
        # CPU out of its deep C-states -- a cost paid by the whole system, not
        # just this process.
        self._data_ready = threading.Event()
        self._paused = False
        self._last_partial_at = 0.0
        self._ready = False

    # ------------------------------------------------------------ start
    def start(self) -> None:
        """Load models, then open audio. Blocks for the model load."""
        self.worker.load()
        self._ready = True
        self.worker.start()
        self._open_audio()

        self._vad_thread = threading.Thread(target=self._vad_loop, daemon=True,
                                            name="vad-loop")
        self._vad_thread.start()

        src = self._source.info if self._source else None
        self.bus.emit(StatusEvent(
            "ready", f"listening: {src.name}" if src else "listening"))

    def _open_audio(self) -> None:
        try:
            self._source = open_source(self.cfg.audio.backend,
                                       self.cfg.audio.source,
                                       self.cfg.audio.device)
        except CaptureError as e:
            self.bus.emit(StatusEvent("error", str(e)))
            raise
        log.info("capture: %s", self._source.info)
        self._source.start(self._on_audio)

    # T1 -- real-time audio thread. Must not block, allocate heavily, or log.
    def _on_audio(self, frames: np.ndarray) -> None:
        self.ring.write(frames)
        self._data_ready.set()

    # --------------------------------------------------------- switching
    def switch_source(self, source: str | None = None,
                      device: str | None = None) -> None:
        """Change loopback/mic or device without restarting models."""
        if source is not None:
            self.cfg.audio.source = source
        if device is not None or source is not None:
            self.cfg.audio.device = device

        if self._source is not None:
            self._source.stop()
            self._source = None
        self.ring.clear()
        self.vad.reset()
        self.segmenter.reset()
        self._open_audio()
        info = self._source.info if self._source else None
        self.bus.emit(StatusEvent("ready",
                                  f"listening: {info.name}" if info else "listening"))

    def set_paused(self, paused: bool) -> None:
        self._paused = paused
        self.worker.set_paused(paused)
        if paused:
            self.segmenter.reset()
            self.vad.reset()
            self.ring.clear()
        self.bus.emit(StatusEvent("paused" if paused else "ready", ""))

    # -------------------------------------------------------------- T2
    def _next_frame(self) -> np.ndarray | None:
        """One VAD frame, blocking on the audio callback rather than polling.

        The clear-then-recheck dance closes the lost-wakeup race: if the
        callback fires between the failed read and the wait, the event is
        already set and wait() returns immediately.
        """
        frame = self.ring.read(FRAME_SAMPLES)
        if frame is not None:
            return frame
        self._data_ready.clear()
        frame = self.ring.read(FRAME_SAMPLES)
        if frame is not None:
            return frame
        self._data_ready.wait(0.1)
        return None

    def _vad_loop(self) -> None:
        partial_interval = self.cfg.asr.partial_interval_ms / 1000.0
        while not self._stop.is_set():
            frame = self._next_frame()
            if frame is None:
                continue
            if self._paused:
                continue

            try:
                prob = self.vad(frame)
            except Exception as e:                        # noqa: BLE001
                log.error("VAD failure: %s", e)
                self.vad.reset()
                continue

            seg = self.segmenter.push(frame, prob)
            if seg is not None:
                self.worker.submit_segment(seg)
                self._last_partial_at = 0.0
                continue

            if partial_interval > 0 and self.segmenter.speaking:
                now = time.monotonic()
                if (now - self._last_partial_at >= partial_interval
                        and self.segmenter.current_ms >= 700):
                    self._last_partial_at = now
                    self.worker.submit_partial(self.segmenter.current_audio())

        # Stream ending: do not silently swallow a half-spoken sentence.
        tail = self.segmenter.flush()
        if tail is not None:
            self.worker.submit_segment(tail)

    def on_partial_text(self, text: str) -> None:
        """Feed the latest English hypothesis back to the segmenter.

        This is what arms the eager endpoint (commit at 250 ms of silence when
        the transcript already ends in .?!) instead of waiting the full 550 ms
        hangover. It was dead code until now -- nothing called it, so the
        eager path had never once fired in production.

        It only works while partial ASR is enabled, and partials cost real
        power: measured 2.46 W -> 5.02 W of iGPU. Off by default; turn
        asr.partial_interval_ms back on if you want the lower latency and are
        willing to pay for it.
        """
        self.segmenter.set_partial_text(text)

    # --------------------------------------------------------------- io
    @property
    def ready(self) -> bool:
        return self._ready

    def underruns(self) -> int:
        return self.ring.overruns

    def stop(self) -> None:
        self._stop.set()
        if self._source is not None:
            self._source.stop()
            self._source = None
        if self._vad_thread is not None:
            self._vad_thread.join(timeout=3)
        self.worker.stop()
        self.metrics.underruns = self.ring.overruns
        self.worker.close()
        self.bus.emit(StatusEvent("stopped", ""))
