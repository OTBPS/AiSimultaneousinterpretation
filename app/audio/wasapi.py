"""WASAPI capture via PyAudioWPatch -- system loopback and microphone.

Loopback is why this backend is primary: plain sounddevice cannot capture what
Windows is playing. PyAudioWPatch exposes every render device as a matching
loopback input and delivers audio on a PortAudio callback thread, which is
exactly the push model the ring buffer wants.

The callback runs in a real-time context: no logging, no locks beyond the ring
buffer's, and no exception may ever escape it.
"""
from __future__ import annotations

import numpy as np
import pyaudiowpatch as pyaudio

from .capture import AudioSource, CaptureError, DeviceInfo, FrameCallback
from .resample import to_16k_mono

BLOCK_MS = 20


def _matches(name: str, want: str | None) -> bool:
    return want is None or want.lower() in name.lower()


def enumerate_devices() -> list[DeviceInfo]:
    out: list[DeviceInfo] = []
    with pyaudio.PyAudio() as pa:
        for d in pa.get_device_info_generator():
            if d.get("maxInputChannels", 0) < 1:
                continue
            out.append(DeviceInfo(
                index=int(d["index"]),
                name=str(d["name"]),
                channels=int(d["maxInputChannels"]),
                sample_rate=int(d["defaultSampleRate"]),
                is_loopback=bool(d.get("isLoopbackDevice", False)),
            ))
    return out


def _pick_loopback(pa, device: str | None) -> dict:
    cands = [d for d in pa.get_loopback_device_info_generator()
             if _matches(str(d["name"]), device)]
    if not cands:
        raise CaptureError(
            f"no WASAPI loopback device matching {device!r}. "
            f"Run: python tools/list_devices.py")
    if device is None:
        # Prefer the loopback belonging to the current default speaker.
        try:
            api = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
            spk = pa.get_device_info_by_index(api["defaultOutputDevice"])
            for d in cands:
                if str(spk["name"]) in str(d["name"]):
                    return d
        except Exception:                                  # noqa: BLE001
            pass
    return cands[0]


def _pick_mic(pa, device: str | None) -> dict:
    if device is None:
        try:
            return pa.get_default_input_device_info()
        except Exception:                                  # noqa: BLE001
            pass
    cands = [d for d in pa.get_device_info_generator()
             if d.get("maxInputChannels", 0) > 0
             and not d.get("isLoopbackDevice", False)
             and _matches(str(d["name"]), device)]
    if not cands:
        raise CaptureError(f"no input device matching {device!r}")
    return cands[0]


class WasapiSource(AudioSource):
    def __init__(self, source: str, device: str | None = None) -> None:
        self._pa = pyaudio.PyAudio()
        try:
            d = (_pick_loopback(self._pa, device) if source == "loopback"
                 else _pick_mic(self._pa, device))
        except Exception:
            self._pa.terminate()
            self._pa = None
            raise
        self._rate = int(d["defaultSampleRate"])
        self._channels = int(d["maxInputChannels"])
        self._info = DeviceInfo(
            index=int(d["index"]), name=str(d["name"]),
            channels=self._channels, sample_rate=self._rate,
            is_loopback=bool(d.get("isLoopbackDevice", False)),
        )
        self._stream = None
        self._cb: FrameCallback | None = None
        self.callback_errors = 0

    @property
    def info(self) -> DeviceInfo:
        return self._info

    def _on_audio(self, in_data, frame_count, time_info, status):
        try:
            raw = np.frombuffer(in_data, dtype=np.float32)
            cb = self._cb
            if cb is not None:
                cb(to_16k_mono(raw, self._rate, self._channels))
        except Exception:                                  # noqa: BLE001
            self.callback_errors += 1                      # never raise here
        return (None, pyaudio.paContinue)

    def start(self, callback: FrameCallback) -> None:
        if self._stream is not None:
            raise CaptureError("source already started")
        self._cb = callback
        frames = max(160, int(self._rate * BLOCK_MS / 1000))
        try:
            self._stream = self._pa.open(
                format=pyaudio.paFloat32,
                channels=self._channels,
                rate=self._rate,
                input=True,
                input_device_index=self._info.index,
                frames_per_buffer=frames,
                stream_callback=self._on_audio,
            )
        except Exception as e:                             # noqa: BLE001
            raise CaptureError(f"cannot open {self._info.name!r}: {e}") from e
        self._stream.start_stream()

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop_stream()
                self._stream.close()
            finally:
                self._stream = None
        self._cb = None
        if self._pa is not None:
            self._pa.terminate()
            self._pa = None
