"""Launch the overlay, run it for N seconds, then quit and report.

The GUI is the one path that unit tests cannot cover: it needs a real Qt event
loop, a real window handle (for the click-through ctypes call) and a real
audio device. This drives all three and fails loudly instead of looking fine
in a screenshot.

    python tools/smoke_gui.py --seconds 35
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QTimer                         # noqa: E402
from PySide6.QtWidgets import QApplication                # noqa: E402

from app import config, storage                           # noqa: E402
from app.console import setup_console                     # noqa: E402
from app.ui.overlay import Overlay                        # noqa: E402


def _play_async(wav: str) -> None:
    """Play a wav to the default output device on a background thread."""
    import threading

    def run():
        import numpy as np
        import pyaudiowpatch as pyaudio
        import soundfile as sf
        import soxr

        data, sr = sf.read(wav, dtype="float32")
        if data.ndim == 1:
            data = np.stack([data, data], axis=1)
        pa = pyaudio.PyAudio()
        try:
            out_sr = 48000
            if sr != out_sr:
                data = soxr.resample(data, sr, out_sr).astype(np.float32)
            stream = pa.open(format=pyaudio.paFloat32, channels=2,
                             rate=out_sr, output=True)
            block = 1024
            for i in range(0, len(data), block):
                stream.write(data[i:i + block].tobytes())
            stream.stop_stream()
            stream.close()
        finally:
            pa.terminate()

    threading.Thread(target=run, daemon=True, name="wav-player").start()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=35.0)
    ap.add_argument("--source", default=None, choices=("loopback", "mic"))
    ap.add_argument("--play", default=None,
                    help="play this wav out the speakers so loopback capture "
                         "has something real to transcribe")
    args = ap.parse_args()

    setup_console()
    cfg = config.load(root=ROOT)
    if args.source:
        cfg.audio.source = args.source
    storage.prepare(cfg.cache_path, cfg.runtime.cache_budget_gb)

    errors: list[str] = []

    def excepthook(exc_type, exc, tb):
        errors.append("".join(traceback.format_exception(exc_type, exc, tb)))
    sys.excepthook = excepthook

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    overlay = Overlay(cfg)
    overlay.show()

    device_name = {"name": "(not opened)"}

    def remember_device():
        src = overlay.pipeline._source
        if src is not None:
            device_name["name"] = src.info.name

    QTimer.singleShot(8000, remember_device)

    if args.play:
        # Give the models time to load, then play into the speakers; the
        # loopback device we are capturing from will hear it.
        QTimer.singleShot(9000, lambda: _play_async(args.play))

    state = {"frames": 0}

    def tick():
        state["frames"] += 1

    heartbeat = QTimer()
    heartbeat.timeout.connect(tick)
    heartbeat.start(100)

    QTimer.singleShot(int(args.seconds * 1000), overlay._quit)
    app.exec()

    pipe = overlay.pipeline
    print("\n" + "=" * 60)
    print(f"ran for              {args.seconds:.0f}s")
    print(f"event loop ticks     {state['frames']} "
          f"(expect ~{int(args.seconds * 10)}; far fewer means the UI froze)")
    print(f"pipeline ready       {pipe.ready}")
    print(f"capture device       {device_name['name']}")
    print(f"ring overruns        {pipe.ring.overruns}")
    print(f"bus events dropped   {overlay.bus.dropped}")
    print(f"lines rendered       {len(overlay.view._lines)}")
    print(pipe.metrics.report())
    if errors:
        print("\nUNCAUGHT EXCEPTIONS:")
        for e in errors:
            print(e)
    print("=" * 60)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
