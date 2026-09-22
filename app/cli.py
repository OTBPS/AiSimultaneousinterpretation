"""Terminal frontend.

Phase-1 deliverable: proves the whole pipeline and the latency budget before
any GUI exists. It consumes exactly the same EventBus the Qt overlay does, so
if this works the overlay is a rendering exercise, not a pipeline change.
"""
from __future__ import annotations

import sys
import time

from .bus import EventBus
from .metrics import Metrics
from .types import (LineDoneEvent, LineSkippedEvent, LineStartEvent,
                    MetricsEvent, PartialEvent, StatusEvent, TokenEvent)

DIM = "\033[2m"
GREY = "\033[90m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
RED = "\033[31m"
RESET = "\033[0m"
CLEAR_LINE = "\r\033[2K"


class CliRenderer:
    def __init__(self, bus: EventBus, metrics: Metrics,
                 show_source: bool = True, show_metrics: bool = True) -> None:
        self.bus = bus
        self.metrics = metrics
        self.show_source = show_source
        self.show_metrics = show_metrics
        self._transient = False          # a partial/status line is on screen
        self._open_line: int | None = None
        self._last_status = ""

    # ------------------------------------------------------------------
    def _clear_transient(self) -> None:
        if self._transient:
            sys.stdout.write(CLEAR_LINE)
            self._transient = False

    def _w(self, text: str) -> None:
        sys.stdout.write(text)
        sys.stdout.flush()

    def _close_open_line(self) -> None:
        if self._open_line is not None:
            self._w("\n")
            self._open_line = None

    # ------------------------------------------------------------------
    def handle(self, ev) -> None:
        if isinstance(ev, StatusEvent):
            self._clear_transient()
            self._close_open_line()
            colour = RED if ev.state == "error" else GREY
            msg = f"[{ev.state}] {ev.detail}".rstrip()
            if msg != self._last_status:
                self._w(f"{colour}{msg}{RESET}\n")
                self._last_status = msg

        elif isinstance(ev, PartialEvent):
            if self._open_line is None and self.show_source:
                self._clear_transient()
                self._w(f"{DIM}… {ev.text}{RESET}")
                self._transient = True

        elif isinstance(ev, LineStartEvent):
            self._clear_transient()
            self._close_open_line()
            if self.show_source:
                mark = "↳" if ev.is_continuation else "»"
                self._w(f"{GREY}{mark} {ev.source}{RESET}\n")
            self._open_line = ev.line_id
            self._w(f"{CYAN}")

        elif isinstance(ev, TokenEvent):
            if self._open_line != ev.line_id:
                self._clear_transient()
                self._close_open_line()
                self._open_line = ev.line_id
                self._w(f"{CYAN}")
            self._w(ev.text)

        elif isinstance(ev, LineDoneEvent):
            if self._open_line == ev.line_id:
                self._w(RESET)
                if self.show_metrics:
                    self._w(f"{GREY}   [{ev.ttft_ms:.0f}/{ev.total_ms:.0f} ms]{RESET}")
                self._w("\n")
                self._open_line = None

        elif isinstance(ev, LineSkippedEvent):
            self._clear_transient()
            self._close_open_line()
            self._w(f"{YELLOW}(… skipped: {ev.source[:60]}){RESET}\n")

        elif isinstance(ev, MetricsEvent):
            if self.show_metrics and ev.backlog > 1:
                self._clear_transient()
                flag = " fast-model" if ev.using_fast_model else ""
                self._w(f"{YELLOW}[backlog {ev.backlog}{flag}]{RESET}")
                self._transient = True

    # ------------------------------------------------------------------
    def run(self, stop_check, poll_s: float = 0.03) -> None:
        """Drain the bus until `stop_check()` returns True."""
        while not stop_check():
            drained = False
            for ev in self.bus.drain():
                self.handle(ev)
                drained = True
            if not drained:
                time.sleep(poll_s)
        for ev in self.bus.drain():
            self.handle(ev)
        self._clear_transient()
        self._close_open_line()
        self._w(RESET)
