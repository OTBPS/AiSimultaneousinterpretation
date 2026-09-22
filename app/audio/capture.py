"""Audio source abstraction.

Two concrete backends exist (pyaudiowpatch, soundcard). Everything upstream
talks to `AudioSource`, so switching backend or device is a config change and
a restart of this object only -- the VAD, GPU worker and UI never notice.
"""
from __future__ import annotations

import abc
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True, frozen=True)
class DeviceInfo:
    index: int
    name: str
    channels: int
    sample_rate: int
    is_loopback: bool

    def __str__(self) -> str:
        kind = "loopback" if self.is_loopback else "input   "
        return (f"[{self.index:3d}] {kind}  {self.sample_rate:6d} Hz  "
                f"{self.channels}ch  {self.name}")


# Called from the audio thread with float32 mono @ 16 kHz. Must not block.
FrameCallback = Callable[[np.ndarray], None]


class CaptureError(RuntimeError):
    pass


class AudioSource(abc.ABC):
    """One open capture stream."""

    @abc.abstractmethod
    def start(self, callback: FrameCallback) -> None: ...

    @abc.abstractmethod
    def stop(self) -> None: ...

    @property
    @abc.abstractmethod
    def info(self) -> DeviceInfo: ...

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.stop()
        return False


def open_source(backend: str, source: str, device: str | None) -> AudioSource:
    """Factory. `source` is loopback or mic; `device` is a name substring."""
    if backend == "pyaudiowpatch":
        from .wasapi import WasapiSource
        return WasapiSource(source, device)
    if backend == "soundcard":
        from .soundcard_src import SoundCardSource
        return SoundCardSource(source, device)
    raise CaptureError(f"unknown audio backend: {backend!r}")


def list_devices(backend: str = "pyaudiowpatch") -> list[DeviceInfo]:
    if backend == "pyaudiowpatch":
        from .wasapi import enumerate_devices
        return enumerate_devices()
    from .soundcard_src import enumerate_devices as sc_enumerate
    return sc_enumerate()
