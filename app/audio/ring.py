"""Lock-light ring buffer for float32 mono audio.

Written from the PortAudio callback thread, read by the VAD thread. The
callback must never block, so writes overwrite the oldest samples and simply
count the overrun instead of waiting for the reader.
"""
from __future__ import annotations

import threading

import numpy as np


class RingBuffer:
    def __init__(self, capacity: int) -> None:
        self._buf = np.zeros(capacity, dtype=np.float32)
        self._cap = capacity
        self._w = 0              # total samples ever written
        self._r = 0              # total samples ever read
        self._lock = threading.Lock()
        self.overruns = 0

    def write(self, data: np.ndarray) -> None:
        n = len(data)
        if n == 0:
            return
        if n >= self._cap:                    # pathological block, keep the tail
            data = data[-self._cap:]
            n = self._cap
        with self._lock:
            start = self._w % self._cap
            end = start + n
            if end <= self._cap:
                self._buf[start:end] = data
            else:
                split = self._cap - start
                self._buf[start:] = data[:split]
                self._buf[:end - self._cap] = data[split:]
            self._w += n
            behind = self._w - self._r
            if behind > self._cap:            # reader lost data
                self.overruns += 1
                self._r = self._w - self._cap

    @property
    def available(self) -> int:
        with self._lock:
            return self._w - self._r

    def read(self, n: int) -> np.ndarray | None:
        """Consume exactly n samples, or return None if not enough yet."""
        with self._lock:
            if self._w - self._r < n:
                return None
            start = self._r % self._cap
            end = start + n
            if end <= self._cap:
                out = self._buf[start:end].copy()
            else:
                out = np.concatenate((self._buf[start:],
                                      self._buf[:end - self._cap]))
            self._r += n
            return out

    def peek_back(self, n: int) -> np.ndarray:
        """Look at the n most recently *read* samples (for VAD pre-roll)."""
        with self._lock:
            n = min(n, self._cap, self._r)
            if n == 0:
                return np.zeros(0, dtype=np.float32)
            start = (self._r - n) % self._cap
            end = start + n
            if end <= self._cap:
                return self._buf[start:end].copy()
            return np.concatenate((self._buf[start:], self._buf[:end - self._cap]))

    def clear(self) -> None:
        with self._lock:
            self._r = self._w
