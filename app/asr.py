"""Whisper ASR on the iGPU via OpenVINO GenAI.

Model: whisper-large-v3-turbo INT8. Measured on this machine -- 5 s chunks at
320 ms mean / 343 ms p95, RTF 0.026 on a 95 s clip, and 5.5x faster than CPU.
turbo also beat whisper-small despite being larger (4 decoder layers vs 12),
so there is no reason to use a smaller model here.

Note turbo does NOT support Whisper's built-in `translate` task -- it was
trained for transcription only. We transcribe here and translate in mt.py.

Not thread-safe by design: only GpuWorker may call it (E1 showed concurrent
GPU use is 4.69x worse than serial).
"""
from __future__ import annotations

import logging
import re
import time

import numpy as np
import openvino_genai as ov_genai

log = logging.getLogger("asr")

# Whisper emits these on silence or noise with high confidence. They are
# training-set artefacts (subtitle boilerplate), not transcription errors.
_HALLUCINATIONS = {
    "thank you.", "thanks for watching!", "thanks for watching.",
    "you", "bye.", "bye bye.", ".", "...", "thank you very much.",
    "please subscribe.", "subscribe to my channel.",
    "字幕由amara.org社区提供", "字幕志愿者", "please subscribe to my channel.",
}
_REPEAT_RE = re.compile(r"\b(\w+(?:\s+\w+){0,3})\b(?:\s+\1\b){3,}", re.IGNORECASE)


class AsrEngine:
    def __init__(self, model_path: str, device: str = "GPU",
                 language: str = "en", ov_props: dict | None = None,
                 drop_hallucinations: bool = True) -> None:
        self.model_path = model_path
        self.language = language
        self.drop_hallucinations = drop_hallucinations
        t0 = time.perf_counter()
        self._pipe = ov_genai.WhisperPipeline(model_path, device,
                                              **(ov_props or {}))
        self.load_s = time.perf_counter() - t0
        log.info("ASR loaded in %.1fs (%s on %s)", self.load_s,
                 model_path.rsplit("\\", 1)[-1], device)

    def warmup(self) -> None:
        self._pipe.generate(np.zeros(16000 * 3, dtype=np.float32))

    def transcribe(self, audio: np.ndarray, duration_s: float = 0.0,
                   mean_prob: float = 1.0) -> str:
        if audio.size == 0:
            return ""
        raw = self._pipe.generate(audio)
        text = str(raw).strip()
        if self.drop_hallucinations and self._is_hallucination(
                text, duration_s or len(audio) / 16000, mean_prob):
            log.debug("dropped likely hallucination: %r", text[:60])
            return ""
        return text

    @staticmethod
    def _is_hallucination(text: str, duration_s: float, mean_prob: float) -> bool:
        if not text:
            return True
        low = text.strip().lower()
        # Boilerplate is only suspicious on short or weakly-voiced audio;
        # someone may genuinely say "thank you" in a long, clearly voiced turn.
        if low in _HALLUCINATIONS and (duration_s < 1.2 or mean_prob < 0.6):
            return True
        if _REPEAT_RE.search(text):
            return True
        return False
