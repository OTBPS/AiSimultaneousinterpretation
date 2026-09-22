"""The backend/frontend boundary: a bounded queue of UI events.

The pipeline calls `emit()`; a frontend calls `drain()` on its own clock.
Nothing else crosses. A slow or dead frontend must never stall the pipeline,
so `emit()` drops the oldest event rather than blocking.
"""
from __future__ import annotations

import queue
import threading
from collections.abc import Iterator

from .types import UiEvent


class EventBus:
    def __init__(self, maxsize: int = 2048) -> None:
        self._q: queue.Queue = queue.Queue(maxsize=maxsize)
        self._dropped = 0
        self._lock = threading.Lock()

    def emit(self, event: UiEvent) -> None:
        """Non-blocking. Drops the oldest event if the consumer fell behind."""
        try:
            self._q.put_nowait(event)
        except queue.Full:
            try:
                self._q.get_nowait()
                self._q.put_nowait(event)
            except queue.Empty:                       # pragma: no cover
                pass
            with self._lock:
                self._dropped += 1

    def drain(self, max_items: int = 512) -> Iterator[UiEvent]:
        """Yield everything currently queued, without waiting."""
        for _ in range(max_items):
            try:
                yield self._q.get_nowait()
            except queue.Empty:
                return

    def get(self, timeout: float | None = None) -> UiEvent | None:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    @property
    def dropped(self) -> int:
        with self._lock:
            return self._dropped
