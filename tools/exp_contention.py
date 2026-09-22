"""E1 — Measure iGPU contention when Whisper (ASR) and Qwen3-4B (MT) share the GPU.

Decides the threading model for the pipeline:
  serial   = one worker thread owns the GPU, ASR and MT never overlap
  parallel = two threads, ASR and MT run concurrently

Decision rule: adopt `parallel` only if asr_slowdown + mt_slowdown < 2.0
(i.e. concurrency actually buys overlap) and neither pipeline errors out.

Run from the project root:  python tools/exp_contention.py
"""
import io
import statistics
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import openvino_genai as ov_genai

ROOT = Path(__file__).resolve().parent.parent
MODELS = Path(r"D:\AI\Models")
ASR_MODEL = MODELS / "whisper-large-v3-turbo-int8-ov"
MT_MODEL = MODELS / "Qwen3-4B-int4-ov"
WAV = ROOT / "bench" / "speech.wav"

CHUNK_S = 5
CONCURRENT_S = 60
SYS = ("You are a professional simultaneous interpreter. Translate the user's "
       "English into natural, fluent Chinese. Output ONLY the translation.")
MT_PROMPT = ("Although the initial benchmarks suggested that the new allocator "
             "would reduce fragmentation, subsequent testing under sustained load "
             "revealed that it actually increased peak memory usage by roughly "
             "twelve percent. /no_think")

OUT = io.open(ROOT / "tools" / "e1_contention.txt", "w", encoding="utf-8")


def P(*a):
    print(*a, file=OUT, flush=True)
    print(*a, flush=True)


def pct(xs, q):
    return statistics.quantiles(xs, n=100)[q - 1] if len(xs) > 1 else xs[0]


def load_audio():
    audio, sr = sf.read(WAV, dtype="float32")
    assert sr == 16000, f"expected 16 kHz, got {sr}"
    n = CHUNK_S * sr
    return [audio[i:i + n] for i in range(0, len(audio) - n, n)]


def mt_generate(pipe):
    """One MT generation. Returns (ttft_s, decode_tok_s, n_tokens)."""
    st = {"first": None, "n": 0}

    def cb(_tok):
        st["n"] += 1
        if st["first"] is None:
            st["first"] = time.perf_counter()
        return ov_genai.StreamingStatus.RUNNING

    t0 = time.perf_counter()
    pipe.generate(MT_PROMPT, max_new_tokens=200, do_sample=False, streamer=cb)
    end = time.perf_counter()
    if st["n"] < 2:
        return end - t0, 0.0, st["n"]
    return st["first"] - t0, (st["n"] - 1) / (end - st["first"]), st["n"]


def main():
    chunks = load_audio()
    P(f"audio: {len(chunks)} x {CHUNK_S}s chunks from {WAV.name}")

    P("\nloading pipelines (both on GPU) ...")
    t0 = time.perf_counter()
    asr = ov_genai.WhisperPipeline(str(ASR_MODEL), "GPU")
    t_asr = time.perf_counter() - t0
    t0 = time.perf_counter()
    mt = ov_genai.LLMPipeline(str(MT_MODEL), "GPU")
    t_mt = time.perf_counter() - t0
    P(f"  ASR load {t_asr:.1f}s | MT load {t_mt:.1f}s")
    P("  -> two GPU contexts coexist OK")

    asr.generate(chunks[0])
    mt.start_chat(SYS)
    mt.generate("hi", max_new_tokens=4)
    mt.finish_chat()

    # ---------- solo baselines ----------
    P("\n" + "=" * 62)
    P("SOLO BASELINE")
    P("=" * 62)

    solo_asr = []
    for i in range(30):
        c = chunks[i % len(chunks)]
        t0 = time.perf_counter()
        asr.generate(c)
        solo_asr.append(time.perf_counter() - t0)
    P(f"ASR  n={len(solo_asr):3d}  p50 {statistics.median(solo_asr)*1000:6.1f} ms"
      f"  p95 {pct(solo_asr, 95)*1000:6.1f} ms")

    solo_ttft, solo_tps = [], []
    mt.start_chat(SYS)
    for _ in range(6):
        ttft, tps, _ = mt_generate(mt)
        solo_ttft.append(ttft)
        solo_tps.append(tps)
    mt.finish_chat()
    P(f"MT   n={len(solo_tps):3d}  TTFT p50 {statistics.median(solo_ttft)*1000:6.1f} ms"
      f"  decode {statistics.median(solo_tps):5.2f} tok/s")

    # ---------- concurrent ----------
    P("\n" + "=" * 62)
    P(f"CONCURRENT ({CONCURRENT_S}s, ASR thread + MT thread)")
    P("=" * 62)

    stop = threading.Event()
    co_asr, co_ttft, co_tps = [], [], []
    errors = []

    def asr_loop():
        i = 0
        try:
            while not stop.is_set():
                c = chunks[i % len(chunks)]
                i += 1
                t0 = time.perf_counter()
                asr.generate(c)
                co_asr.append(time.perf_counter() - t0)
        except Exception as e:                      # noqa: BLE001
            errors.append(f"ASR thread: {type(e).__name__}: {e}")

    def mt_loop():
        try:
            mt.start_chat(SYS)
            while not stop.is_set():
                ttft, tps, _ = mt_generate(mt)
                co_ttft.append(ttft)
                co_tps.append(tps)
            mt.finish_chat()
        except Exception as e:                      # noqa: BLE001
            errors.append(f"MT thread: {type(e).__name__}: {e}")

    ta = threading.Thread(target=asr_loop, daemon=True)
    tm = threading.Thread(target=mt_loop, daemon=True)
    wall0 = time.perf_counter()
    ta.start()
    tm.start()
    time.sleep(CONCURRENT_S)
    stop.set()
    ta.join(timeout=60)
    tm.join(timeout=60)
    wall = time.perf_counter() - wall0

    if errors:
        for e in errors:
            P(f"  !! {e}")
    if not co_asr or not co_tps:
        P("  concurrent run produced no samples -- aborting")
        OUT.close()
        return

    P(f"ASR  n={len(co_asr):3d}  p50 {statistics.median(co_asr)*1000:6.1f} ms"
      f"  p95 {pct(co_asr, 95)*1000:6.1f} ms")
    P(f"MT   n={len(co_tps):3d}  TTFT p50 {statistics.median(co_ttft)*1000:6.1f} ms"
      f"  decode {statistics.median(co_tps):5.2f} tok/s")

    # ---------- verdict ----------
    asr_slow = pct(co_asr, 95) / pct(solo_asr, 95)
    mt_slow = statistics.median(solo_tps) / statistics.median(co_tps)
    asr_done = len(co_asr) * CHUNK_S

    P("\n" + "=" * 62)
    P("VERDICT")
    P("=" * 62)
    P(f"asr_slowdown (p95)      {asr_slow:5.2f}x")
    P(f"mt_slowdown  (decode)   {mt_slow:5.2f}x")
    P(f"sum                     {asr_slow + mt_slow:5.2f}   (threshold 2.00)")
    P(f"audio transcribed       {asr_done:.0f}s of speech in {wall:.0f}s wall"
      f"  ({asr_done/wall:.1f}x realtime, while MT also running)")

    ok = (asr_slow + mt_slow) < 2.0 and not errors
    P(f"\n=> RECOMMENDED gpu_concurrency: {'parallel' if ok else 'serial'}")
    if not ok:
        P("   (concurrency buys no overlap on a shared iGPU command queue;")
        P("    one worker thread owning the GPU is simpler and no slower)")
    OUT.close()


if __name__ == "__main__":
    main()
