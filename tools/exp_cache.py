"""E2 — Measure OpenVINO CACHE_DIR benefit on model load time.

First load compiles the model for the GPU; with CACHE_DIR the compiled blob is
reused on subsequent loads. Tells us the real cold/warm startup cost to quote
in the README and to size the startup spinner.

Run from the project root:  python tools/exp_cache.py
"""
import io
import shutil
import time
from pathlib import Path

import numpy as np
import openvino_genai as ov_genai

ROOT = Path(__file__).resolve().parent.parent
MODELS = Path(r"D:\AI\Models")
ASR_MODEL = MODELS / "whisper-large-v3-turbo-int8-ov"
MT_MODEL = MODELS / "Qwen3-4B-int4-ov"
CACHE = ROOT / ".ovcache"

OUT = io.open(ROOT / "tools" / "e2_cache.txt", "w", encoding="utf-8")


def P(*a):
    print(*a, file=OUT, flush=True)
    print(*a, flush=True)


def time_load(kind, path, cfg):
    t0 = time.perf_counter()
    if kind == "asr":
        p = ov_genai.WhisperPipeline(str(path), "GPU", **cfg)
        p.generate(np.zeros(16000 * 3, dtype=np.float32))
    else:
        p = ov_genai.LLMPipeline(str(path), "GPU", **cfg)
        p.generate("hi", max_new_tokens=4)
    el = time.perf_counter() - t0
    del p
    return el


def main():
    if CACHE.exists():
        shutil.rmtree(CACHE)
    P(f"cache dir: {CACHE}  (cleared)\n")

    rows = []
    for kind, path, name in [("asr", ASR_MODEL, "whisper-large-v3-turbo"),
                             ("mt", MT_MODEL, "Qwen3-4B")]:
        no_cache = time_load(kind, path, {})
        cold = time_load(kind, path, {"CACHE_DIR": str(CACHE)})
        warm = time_load(kind, path, {"CACHE_DIR": str(CACHE)})
        rows.append((name, no_cache, cold, warm))
        P(f"{name:24s} no-cache {no_cache:5.2f}s | cold(+write) {cold:5.2f}s | "
          f"warm {warm:5.2f}s | saving {no_cache - warm:5.2f}s")

    size = sum(f.stat().st_size for f in CACHE.rglob("*") if f.is_file()) / 1e6
    total_cold = sum(r[1] for r in rows)
    total_warm = sum(r[3] for r in rows)
    P(f"\ncache size on disk: {size:.0f} MB")
    P(f"startup (both models): {total_cold:.1f}s cold -> {total_warm:.1f}s warm "
      f"({total_cold - total_warm:.1f}s saved)")
    OUT.close()


if __name__ == "__main__":
    main()
