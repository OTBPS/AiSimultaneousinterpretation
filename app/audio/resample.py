"""Native device format -> float32 mono @ 16 kHz, cheaply.

Runs inside the audio callback, so it must be allocation-light and fast.
soxr's 'QQ' (quick) preset costs microseconds per 10 ms block and is far more
than good enough for a 16 kHz ASR front-end.
"""
from __future__ import annotations

import numpy as np
import soxr

TARGET_SR = 16000


def to_mono(data: np.ndarray, channels: int) -> np.ndarray:
    if channels <= 1:
        return data.reshape(-1)
    return data.reshape(-1, channels).mean(axis=1)


def to_16k_mono(data: np.ndarray, src_sr: int, channels: int) -> np.ndarray:
    """`data` is interleaved float32. Returns contiguous float32 mono @ 16 kHz."""
    mono = to_mono(np.asarray(data, dtype=np.float32), channels)
    if src_sr == TARGET_SR:
        return np.ascontiguousarray(mono)
    return np.ascontiguousarray(
        soxr.resample(mono, src_sr, TARGET_SR, quality="QQ").astype(np.float32)
    )
