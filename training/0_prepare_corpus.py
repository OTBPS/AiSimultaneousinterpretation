"""Step 0 (LAPTOP): build the English side of the corpus from real audio.

RUNS ON THE LAPTOP, using the iGPU and the exact ASR + VAD + sentence-splitting
code the live pipeline uses. That is the entire point.

Why not just scrape clean written English? Because the student never sees
clean written English. In production its input is whatever Whisper emitted,
which on this project's own test audio included:

    "...contention when multiple accelerations"      (misheard "accelerators")
    "The operators access shared system memories."   (sentence split wrongly)
    "copies between stages."                         (fragment from a hard cut)

Fine-tuning on tidy prose teaches the model to handle a distribution it will
never encounter. Transcribing real audio with the same models, the same VAD
thresholds and the same splitter reproduces the production input distribution
exactly, including its defects.

Whisper turbo runs at ~38x realtime here, so an hour of audio costs about a
minute and a half.

    python training/0_prepare_corpus.py --audio D:\\talks --out corpus.txt
    python training/0_prepare_corpus.py --audio talk.mp4 --out corpus.txt --keep-raw
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config                                    # noqa: E402
from app.asr import AsrEngine                             # noqa: E402
from app.console import setup_console                     # noqa: E402
from app.segmenter import Segmenter, SegmenterParams      # noqa: E402
from app.sentence_split import split_sentences            # noqa: E402
from app.vad import FRAME_SAMPLES, SAMPLE_RATE, SileroVad  # noqa: E402

AUDIO_EXT = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".mp4",
             ".mkv", ".webm", ".aac"}
_WORD = re.compile(r"[A-Za-z']+")


def load_audio(path: Path) -> np.ndarray | None:
    """Decode to float32 mono @16 kHz. Falls back to ffmpeg for video/compressed."""
    try:
        import soundfile as sf
        import soxr

        data, sr = sf.read(path, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != SAMPLE_RATE:
            data = soxr.resample(data, sr, SAMPLE_RATE).astype(np.float32)
        return np.ascontiguousarray(data)
    except Exception:                                      # noqa: BLE001
        pass

    import subprocess

    try:
        proc = subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(path),
             "-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-"],
            capture_output=True, check=True)
        return np.frombuffer(proc.stdout, dtype=np.float32)
    except FileNotFoundError:
        print(f"  {path.name}: needs ffmpeg on PATH for this format")
    except subprocess.CalledProcessError as e:
        print(f"  {path.name}: ffmpeg failed ({e.returncode})")
    return None


def transcribe(audio: np.ndarray, asr: AsrEngine, seg: Segmenter,
               vad: SileroVad) -> list[str]:
    """Same VAD -> segment -> ASR -> split path the live pipeline takes."""
    vad.reset()
    seg.reset()
    lines: list[str] = []

    def emit(segment):
        text = asr.transcribe(segment.audio, segment.duration_s,
                              segment.mean_speech_prob)
        if text:
            lines.extend(split_sentences(text))

    n = len(audio) // FRAME_SAMPLES
    for i in range(n):
        frame = audio[i * FRAME_SAMPLES:(i + 1) * FRAME_SAMPLES]
        s = seg.push(frame, vad(frame))
        if s is not None:
            emit(s)
    tail = seg.flush()
    if tail is not None:
        emit(tail)
    return lines


def usable(line: str, min_words: int, max_words: int) -> bool:
    words = _WORD.findall(line)
    if not (min_words <= len(words) <= max_words):
        return False
    # Mostly-non-alphabetic lines are ASR noise, not speech.
    letters = sum(c.isalpha() for c in line)
    return letters >= len(line) * 0.5


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True,
                    help="audio/video file, or a directory to walk")
    ap.add_argument("--out", default="corpus.txt")
    ap.add_argument("--min-words", type=int, default=5)
    ap.add_argument("--max-words", type=int, default=60)
    ap.add_argument("--keep-raw", action="store_true",
                    help="also write corpus.raw.txt before dedupe/filtering")
    ap.add_argument("--limit", type=int, default=0, help="stop after N files")
    args = ap.parse_args()

    setup_console()
    cfg = config.load(root=ROOT)

    src = Path(args.audio)
    files = ([src] if src.is_file()
             else sorted(p for p in src.rglob("*") if p.suffix.lower() in AUDIO_EXT))
    if args.limit:
        files = files[:args.limit]
    if not files:
        print(f"no audio found under {src}", file=sys.stderr)
        return 1
    print(f"{len(files)} file(s) to transcribe")

    asr = AsrEngine(cfg.asr.model, cfg.asr.device, cfg.asr.language,
                    cfg.ov_props(), cfg.asr.drop_hallucinations)
    asr.warmup()
    vad = SileroVad(cfg.vad.model)
    seg = Segmenter(SegmenterParams(
        threshold=cfg.vad.threshold,
        endpoint_silence_ms=cfg.vad.endpoint_silence_ms,
        eager_punct_ms=cfg.vad.eager_punct_ms,
        min_speech_ms=cfg.vad.min_speech_ms,
        max_segment_s=cfg.vad.max_segment_s,
        preroll_ms=cfg.vad.preroll_ms,
    ))

    raw: list[str] = []
    total_audio = 0.0
    t0 = time.perf_counter()
    for i, f in enumerate(files, 1):
        audio = load_audio(f)
        if audio is None or audio.size < SAMPLE_RATE:
            continue
        dur = len(audio) / SAMPLE_RATE
        total_audio += dur
        lines = transcribe(audio, asr, seg, vad)
        raw.extend(lines)
        print(f"  [{i}/{len(files)}] {f.name}  {dur/60:.1f} min  "
              f"-> {len(lines)} lines", flush=True)

    wall = time.perf_counter() - t0

    if args.keep_raw:
        Path("corpus.raw.txt").write_text("\n".join(raw), encoding="utf-8")

    seen: set[str] = set()
    kept: list[str] = []
    dropped = Counter()
    for line in raw:
        line = " ".join(line.split())
        if not usable(line, args.min_words, args.max_words):
            dropped["too short / too long / non-speech"] += 1
            continue
        key = line.lower()
        if key in seen:
            dropped["duplicate"] += 1
            continue
        seen.add(key)
        kept.append(line)

    Path(args.out).write_text("\n".join(kept), encoding="utf-8")

    words = sum(len(_WORD.findall(x)) for x in kept)
    print(f"\n{'-'*60}")
    print(f"audio        {total_audio/60:8.1f} min  "
          f"(transcribed in {wall/60:.1f} min, {total_audio/wall:.0f}x realtime)")
    print(f"raw lines    {len(raw):8d}")
    for why, n in dropped.most_common():
        print(f"  dropped    {n:8d}  {why}")
    print(f"kept         {len(kept):8d} lines, {words} words -> {args.out}")
    if kept:
        print("\nsample (this is what the student will actually see at runtime):")
        for x in kept[:5]:
            print(f"  {x}")
    print(f"\nNext, on the desktop:")
    print(f"  python 1_generate.py --corpus {args.out} --out train.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
