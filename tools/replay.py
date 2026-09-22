"""Feed a WAV file through the real pipeline -- no audio device needed.

This is how Phase 1 is verified and how any later change is regression-tested:
same input, same code path as live capture, deterministic enough to diff.

Audio is fed in real time by default, because that is the only way the latency
numbers mean anything -- pushing 95 s of audio in 2 s would just measure how
fast the backlog policy sheds lines.

    python tools/replay.py bench/speech.wav
    python tools/replay.py bench/speech.wav --fast     (throughput only)
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config                                    # noqa: E402
from app.cli import CliRenderer                           # noqa: E402
from app.console import setup_console                     # noqa: E402
from app.gpu_worker import GpuWorker                      # noqa: E402
from app.bus import EventBus                              # noqa: E402
from app.metrics import Metrics                           # noqa: E402
from app.segmenter import Segmenter, SegmenterParams      # noqa: E402
from app.types import StatusEvent                         # noqa: E402
from app.vad import FRAME_SAMPLES, SAMPLE_RATE, SileroVad  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("wav", nargs="?", default="bench/speech.wav")
    ap.add_argument("--fast", action="store_true",
                    help="skip real-time pacing (latency numbers become meaningless)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", default=None, help="override mt.model")
    ap.add_argument("--endpoint-ms", type=int, default=None,
                    help="override vad.endpoint_silence_ms")
    args = ap.parse_args()

    setup_console()
    cfg = config.load(args.config, root=ROOT)
    if args.model:
        cfg.mt.model = args.model
    if args.endpoint_ms:
        cfg.vad.endpoint_silence_ms = args.endpoint_ms

    audio, sr = sf.read(args.wav, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        import soxr
        audio = soxr.resample(audio, sr, SAMPLE_RATE).astype(np.float32)
    dur = len(audio) / SAMPLE_RATE

    bus = EventBus()
    metrics = Metrics(endpoint_ms=cfg.vad.endpoint_silence_ms)
    worker = GpuWorker(cfg, bus, metrics)
    vad = SileroVad(cfg.vad.model)
    seg = Segmenter(SegmenterParams(
        threshold=cfg.vad.threshold,
        endpoint_silence_ms=cfg.vad.endpoint_silence_ms,
        eager_punct_ms=cfg.vad.eager_punct_ms,
        min_speech_ms=cfg.vad.min_speech_ms,
        max_segment_s=cfg.vad.max_segment_s,
        preroll_ms=cfg.vad.preroll_ms,
    ))

    print(f"replay: {args.wav}  {dur:.1f}s  "
          f"endpoint={cfg.vad.endpoint_silence_ms}ms  "
          f"mt={Path(cfg.mt.model).name}")
    print("loading models …")
    worker.load()
    worker.start()

    done = threading.Event()
    renderer = CliRenderer(bus, metrics, show_source=True, show_metrics=True)
    ui = threading.Thread(target=renderer.run, args=(done.is_set,), daemon=True)
    ui.start()

    n_frames = len(audio) // FRAME_SAMPLES
    frame_s = FRAME_SAMPLES / SAMPLE_RATE
    t_start = time.perf_counter()
    for i in range(n_frames):
        frame = audio[i * FRAME_SAMPLES:(i + 1) * FRAME_SAMPLES]
        s = seg.push(frame, vad(frame))
        if s is not None:
            worker.submit_segment(s)
        if not args.fast:
            target = t_start + (i + 1) * frame_s
            slack = target - time.perf_counter()
            if slack > 0:
                time.sleep(slack)

    tail = seg.flush()
    if tail is not None:
        worker.submit_segment(tail)

    # Let the backlog finish rather than cutting it off mid-sentence.
    deadline = time.perf_counter() + 120
    while worker.backlog > 0 and time.perf_counter() < deadline:
        time.sleep(0.2)
    time.sleep(1.0)

    wall = time.perf_counter() - t_start
    worker.stop()
    done.set()
    ui.join(timeout=3)
    worker.close()

    print("\n" + "=" * 64)
    print(f"audio {dur:.1f}s processed in {wall:.1f}s "
          f"({dur / wall:.2f}x realtime)")
    print(metrics.report())
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
