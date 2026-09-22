"""The single owner of the iGPU.

E1 (tools/exp_contention.py) measured what happens when Whisper and Qwen3-4B
run on the Arc 140T at the same time:

    ASR p95      424 ms -> 1337 ms   (3.15x worse)
    MT decode   15.2 t/s ->  9.9 t/s (1.54x worse)
    sum 4.69 against a 2.00 "concurrency pays" threshold

There is no overlap to win -- OpenVINO funnels both into one GPU command queue
and they share a 51 GB/s memory bus. So exactly one thread touches the GPU,
and it interleaves ASR and MT explicitly.

ASR is always served before MT: a stale transcript stalls the whole pipeline,
whereas a late translation only delays one subtitle line.

Backpressure, in escalating order, because a subtitle 30 s behind the speaker
is worse than no subtitle:
    backlog > 2   drop glossary/context injection (saves prefill)
    backlog >= N  switch to the 2.2x faster fallback model
    backlog > 8   drop the oldest untranslated lines
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .asr import AsrEngine
from .bus import EventBus
from .config import Config
from .context import Context
from .glossary import Glossary
from .metrics import Metrics
from .mt import MtEngine
from .sentence_split import split_sentences
from .types import (LineDoneEvent, LineSkippedEvent, LineStartEvent,
                    MetricsEvent, PartialEvent, Segment, StatusEvent,
                    TokenEvent)

log = logging.getLogger("gpu")

MAX_BACKLOG = 8
CONTEXT_OFF_BACKLOG = 2


@dataclass(slots=True)
class MtJob:
    line_id: int
    source: str
    is_continuation: bool
    t_speech_end: float


@dataclass(slots=True)
class AsrJob:
    segment: Segment


@dataclass(slots=True)
class PartialJob:
    audio: np.ndarray


class GpuWorker:
    def __init__(self, cfg: Config, bus: EventBus, metrics: Metrics) -> None:
        self.cfg = cfg
        self.bus = bus
        self.metrics = metrics

        self._asr_q: queue.Queue = queue.Queue(maxsize=8)
        self._mt_q: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._paused = threading.Event()
        # Set whenever work is queued, so the loop can block instead of
        # spinning at 200 Hz. Polling an empty queue burns wakeups and keeps
        # the CPU out of deep C-states for no benefit.
        self._wake = threading.Event()

        # Optional hook back to the pipeline for partial transcripts; this is
        # what arms the segmenter's eager endpoint. Kept as a callback rather
        # than routed through the bus, because the bus has exactly one
        # consumer (the frontend) and must not be split.
        self.on_partial: Callable[[str], None] | None = None

        self.asr: AsrEngine | None = None
        self.mt: MtEngine | None = None
        self._fast_mt: MtEngine | None = None
        self._using_fast = False

        self._context = Context(cfg.mt.context_pairs, cfg.mt.chat_reset_every)
        self._glossary = Glossary(cfg.glossary_path, cfg.glossary.enabled)
        self._next_line_id = 0
        self._last_metrics_at = 0.0

    # ------------------------------------------------------------- setup
    def load(self) -> None:
        """Blocking model load. Caller runs this off the UI thread."""
        props = self.cfg.ov_props()
        self.bus.emit(StatusEvent("loading", "loading speech recognition…"))
        self.asr = AsrEngine(self.cfg.asr.model, self.cfg.asr.device,
                             self.cfg.asr.language, props,
                             self.cfg.asr.drop_hallucinations)
        self.asr.warmup()

        self.bus.emit(StatusEvent("loading", "loading translation model…"))
        self.mt = MtEngine(self.cfg.mt.model, self.cfg.mt.device, props,
                           self.cfg.mt.max_new_tokens, self._context,
                           self._glossary)
        self.mt.warmup()
        self.bus.emit(StatusEvent("ready", f"{len(self._glossary)} glossary terms"))

    def _ensure_fast(self) -> MtEngine | None:
        if self._fast_mt is None and self.cfg.mt.fast_model:
            try:
                self._fast_mt = MtEngine(
                    self.cfg.mt.fast_model, self.cfg.mt.device,
                    self.cfg.ov_props(), self.cfg.mt.max_new_tokens,
                    self._context, self._glossary)
                self._fast_mt.warmup()
            except Exception as e:                        # noqa: BLE001
                log.warning("fallback model unavailable: %s", e)
                self.cfg.mt.fast_model = ""
        return self._fast_mt

    # -------------------------------------------------------- submission
    def submit_segment(self, seg: Segment) -> None:
        try:
            self._asr_q.put_nowait(AsrJob(seg))
            self._wake.set()
        except queue.Full:
            self.metrics.dropped += 1
            log.warning("ASR queue full, dropped segment %d", seg.seq)

    def submit_partial(self, audio: np.ndarray) -> None:
        if self.backlog > 0 or self._asr_q.qsize() > 0:
            return                       # partials are a luxury, never a cost
        try:
            self._asr_q.put_nowait(PartialJob(audio))
            self._wake.set()
        except queue.Full:
            pass

    @property
    def backlog(self) -> int:
        return self._mt_q.qsize()

    # ------------------------------------------------------------- loop
    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="gpu-worker")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()                 # unblock the loop immediately
        if self.mt is not None:
            self.mt.cancel()
        if self._thread is not None:
            self._thread.join(timeout=15)

    def set_paused(self, paused: bool) -> None:
        if paused:
            self._paused.set()
            if self.mt is not None:
                self.mt.cancel()
        else:
            self._paused.clear()

    def _run(self) -> None:
        while not self._stop.is_set():
            did_work = False
            try:
                job = self._asr_q.get_nowait()
                self._do_asr(job)
                did_work = True
            except queue.Empty:
                pass

            if not self._stop.is_set() and self._asr_q.empty():
                try:
                    self._do_mt(self._mt_q.get_nowait())
                    did_work = True
                except queue.Empty:
                    pass

            self._maybe_emit_metrics()
            if did_work:
                continue

            # Nothing to do: clear, re-check (closing the lost-wakeup race),
            # then block. The 0.25 s ceiling keeps metrics ticking and lets
            # stop() be noticed promptly.
            self._wake.clear()
            if self._asr_q.empty() and self._mt_q.empty():
                self._wake.wait(0.25)

    # -------------------------------------------------------------- ASR
    def _do_asr(self, job) -> None:
        if self.asr is None or self._paused.is_set():
            return
        if isinstance(job, PartialJob):
            t0 = time.perf_counter()
            try:
                text = self.asr.transcribe(job.audio)
            except Exception as e:                        # noqa: BLE001
                log.warning("partial ASR failed: %s", e)
                return
            self.metrics.record_asr((time.perf_counter() - t0) * 1000)
            if text:
                self.bus.emit(PartialEvent(text))
                if self.on_partial is not None:
                    self.on_partial(text)
            return

        seg = job.segment
        t0 = time.perf_counter()
        try:
            text = self.asr.transcribe(seg.audio, seg.duration_s,
                                       seg.mean_speech_prob)
        except Exception as e:                            # noqa: BLE001
            log.error("ASR failed on segment %d: %s", seg.seq, e)
            return
        self.metrics.record_asr((time.perf_counter() - t0) * 1000)
        if not text:
            return

        units = split_sentences(text)
        for i, unit in enumerate(units):
            line_id = self._next_line_id
            self._next_line_id += 1
            cont = seg.is_continuation and i == 0
            self.bus.emit(LineStartEvent(line_id, unit, cont))
            self._mt_q.put(MtJob(line_id, unit, cont, seg.t_speech_end))
            self._wake.set()

        self._shed_backlog()

    def _shed_backlog(self) -> None:
        while self._mt_q.qsize() > MAX_BACKLOG:
            try:
                old: MtJob = self._mt_q.get_nowait()
            except queue.Empty:
                return
            self.metrics.dropped += 1
            self.bus.emit(LineSkippedEvent(old.line_id, old.source))

    # --------------------------------------------------------------- MT
    def _do_mt(self, job: MtJob) -> None:
        if self._paused.is_set():
            self.bus.emit(LineSkippedEvent(job.line_id, job.source, "paused"))
            return

        engine = self._select_engine()
        if engine is None:
            return
        engine.recycle_if_needed()

        use_context = self.backlog <= CONTEXT_OFF_BACKLOG
        line_id = job.line_id
        first_token_at: float | None = None

        def on_token(text: str) -> None:
            nonlocal first_token_at
            if first_token_at is None:
                first_token_at = time.perf_counter()
            self.bus.emit(TokenEvent(line_id, text))

        try:
            text, _ = engine.translate(job.source, on_token, use_context,
                                       job.is_continuation)
        except Exception as e:                            # noqa: BLE001
            log.error("MT failed on line %d: %s", line_id, e)
            self.bus.emit(LineSkippedEvent(line_id, job.source, "error"))
            return

        # Latency is measured from when the speaker stopped, not from when we
        # happened to dequeue the job -- queueing delay is delay the user feels.
        now = time.perf_counter()
        total_ms = (now - job.t_speech_end) * 1000
        ttft_ms = ((first_token_at or now) - job.t_speech_end) * 1000
        self.metrics.record_line(ttft_ms, total_ms)
        self.bus.emit(LineDoneEvent(line_id, text, ttft_ms, total_ms))

    def _select_engine(self) -> MtEngine | None:
        threshold = self.cfg.mt.fallback_on_backlog
        if threshold and self.backlog >= threshold:
            fast = self._ensure_fast()
            if fast is not None:
                if not self._using_fast:
                    log.info("backlog %d -> switching to fallback model",
                             self.backlog)
                    self._using_fast = True
                return fast
        if self._using_fast and self.backlog == 0:
            log.info("backlog cleared -> back to primary model")
            self._using_fast = False
        return self.mt

    # ---------------------------------------------------------- metrics
    def _maybe_emit_metrics(self) -> None:
        if not self.cfg.runtime.metrics:
            return
        now = time.monotonic()
        if now - self._last_metrics_at < 2.0:
            return
        self._last_metrics_at = now
        s = self.metrics.snapshot()
        self.bus.emit(MetricsEvent(self.backlog, s["ttft_p95"], s["total_p95"],
                                   int(s["dropped"]), self._using_fast))

    def close(self) -> None:
        for e in (self.mt, self._fast_mt):
            if e is not None:
                e.close()
