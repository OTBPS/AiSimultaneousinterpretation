"""Fallback capture backend using `soundcard`.

Kept because PyAudioWPatch wraps an unmaintained upstream (PyAudio) and WASAPI
device quirks vary by machine. `soundcard` is pull-based, so unlike the
PortAudio backend it needs its own polling thread to feed the same callback.
"""
from __future__ import annotations

import threading

import numpy as np
import soundcard as sc

from .capture import AudioSource, CaptureError, DeviceInfo, FrameCallback
from .resample import to_16k_mono

BLOCK_MS = 20
NATIVE_SR = 48000


def _matches(name: str, want: str | None) -> bool:
    return want is None or want.lower() in name.lower()


def enumerate_devices() -> list[DeviceInfo]:
    out: list[DeviceInfo] = []
    for i, spk in enumerate(sc.all_speakers()):
        out.append(DeviceInfo(i, f"{spk.name} [Loopback]", 2, NATIVE_SR, True))
    for i, mic in enumerate(sc.all_microphones(include_loopback=False)):
        out.append(DeviceInfo(1000 + i, mic.name, mic.channels, NATIVE_SR, False))
    return out


class SoundCardSource(AudioSource):
    def __init__(self, source: str, device: str | None = None) -> None:
        if source == "loopback":
            spks = [s for s in sc.all_speakers() if _matches(s.name, device)]
            if device is None:
                spks = [sc.default_speaker()] + spks
            if not spks:
                raise CaptureError(f"no speaker matching {device!r}")
            self._dev = sc.get_microphone(spks[0].name, include_loopback=True)
            name, loop = f"{spks[0].name} [Loopback]", True
        else:
            mics = [m for m in sc.all_microphones() if _matches(m.name, device)]
            if device is None:
                mics = [sc.default_microphone()] + mics
            if not mics:
                raise CaptureError(f"no microphone matching {device!r}")
            self._dev = mics[0]
            name, loop = mics[0].name, False

        self._channels = max(1, int(getattr(self._dev, "channels", 2) or 2))
        self._info = DeviceInfo(-1, name, self._channels, NATIVE_SR, loop)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._cb: FrameCallback | None = None
        self.callback_errors = 0

    @property
    def info(self) -> DeviceInfo:
        return self._info

    def _loop(self) -> None:
        frames = int(NATIVE_SR * BLOCK_MS / 1000)
        try:
            with self._dev.recorder(samplerate=NATIVE_SR,
                                    channels=self._channels,
                                    blocksize=frames) as rec:
                while not self._stop.is_set():
                    data = rec.record(numframes=frames)
                    cb = self._cb
                    if cb is not None:
                        cb(to_16k_mono(np.asarray(data, dtype=np.float32),
                                       NATIVE_SR, self._channels))
        except Exception:                                  # noqa: BLE001
            self.callback_errors += 1

    def start(self, callback: FrameCallback) -> None:
        if self._thread is not None:
            raise CaptureError("source already started")
        self._cb = callback
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="soundcard-capture")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._cb = None
