"""English -> Chinese translation on the iGPU via OpenVINO GenAI.

Model: Qwen3-4B INT4. Measured 15.5 tok/s steady decode, 173 ms TTFT, ~1.8 s
per sentence. Qwen2.5-1.5B is 2.2x faster but rendered "prefill" as 预读取
(prefetch), so it is only used as a backlog escape hatch.

Two Qwen3 specifics, both measured:
  * thinking mode must be off -- one sentence cost 43.65 s / 674 tokens with it
    on versus 1.61 s / 22 tokens with ` /no_think`;
  * even with /no_think the model still emits an empty <think></think> pair,
    which think_filter.py removes from the token stream.

Not thread-safe by design: only GpuWorker may call it.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable

import openvino_genai as ov_genai

from .context import Context
from .glossary import Glossary
from .think_filter import ThinkFilter

log = logging.getLogger("mt")

SYSTEM_PROMPT = (
    "You are a professional simultaneous interpreter. Translate the user's "
    "English into natural, fluent Chinese.\n"
    "Rules:\n"
    "- Output ONLY the Chinese translation. No explanation, no pinyin, no "
    "quotes, no English.\n"
    "- Translate meaning, not words. Render idioms as idiomatic Chinese.\n"
    "- Keep technical terms accurate and consistent.\n"
    "- If a [Terms] line is given, you MUST use those translations.\n"
    "Examples:\n"
    "Let's not boil the ocean here. -> 咱们别一上来就搞得太大太全。\n"
    "That's low-hanging fruit. -> 那是很容易拿下的部分。"
)

NO_THINK = " /no_think"

# Emitted by mt.translate for each visible chunk of Chinese.
TokenSink = Callable[[str], None]


class MtEngine:
    def __init__(self, model_path: str, device: str = "GPU",
                 ov_props: dict | None = None, max_new_tokens: int = 200,
                 context: Context | None = None,
                 glossary: Glossary | None = None) -> None:
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.context = context or Context()
        self.glossary = glossary
        self._cancel = False

        t0 = time.perf_counter()
        self._pipe = ov_genai.LLMPipeline(model_path, device, **(ov_props or {}))
        self.load_s = time.perf_counter() - t0
        self._chat_open = False
        self._open_chat()
        log.info("MT loaded in %.1fs (%s on %s)", self.load_s,
                 model_path.rsplit("\\", 1)[-1], device)

    # ------------------------------------------------------------------
    @property
    def wants_no_think(self) -> bool:
        """Only Qwen3+ has a thinking mode; 1.5B would take it as literal text."""
        name = self.model_path.lower()
        return "qwen3" in name

    def _open_chat(self) -> None:
        preamble = self.context.preamble()
        self._pipe.start_chat(SYSTEM_PROMPT + ("\n\n" + preamble if preamble else ""))
        self._chat_open = True
        self.context.mark_reset()

    def _close_chat(self) -> None:
        if self._chat_open:
            try:
                self._pipe.finish_chat()
            except Exception:                         # noqa: BLE001
                pass
            self._chat_open = False

    def recycle_if_needed(self) -> bool:
        """Bound KV growth: restart the chat every `reset_every` lines."""
        if not self.context.needs_reset():
            return False
        self._close_chat()
        self._open_chat()
        log.debug("chat session recycled")
        return True

    def warmup(self) -> None:
        self._pipe.generate("hi" + (NO_THINK if self.wants_no_think else ""),
                            max_new_tokens=4, do_sample=False)

    def cancel(self) -> None:
        """Ask the in-flight generation to stop at the next token."""
        self._cancel = True

    # ------------------------------------------------------------------
    def translate(self, source: str, on_token: TokenSink,
                  use_context: bool = True,
                  is_continuation: bool = False) -> tuple[str, float]:
        """Stream a translation. Returns (full_text, ttft_seconds)."""
        self._cancel = False
        filt = ThinkFilter()
        pieces: list[str] = []
        first_at: float | None = None

        prompt = self._build_prompt(source, use_context, is_continuation)
        t0 = time.perf_counter()

        def cb(chunk: str):
            nonlocal first_at
            visible = filt.feed(chunk)
            if visible:
                if first_at is None:
                    first_at = time.perf_counter()
                pieces.append(visible)
                on_token(visible)
            return (ov_genai.StreamingStatus.STOP if self._cancel
                    else ov_genai.StreamingStatus.RUNNING)

        self._pipe.generate(prompt, max_new_tokens=self.max_new_tokens,
                            do_sample=False, streamer=cb)

        tail = filt.flush()
        if tail:
            pieces.append(tail)
            on_token(tail)

        text = "".join(pieces).strip().rstrip('"”')
        ttft = (first_at - t0) if first_at else (time.perf_counter() - t0)
        if text:
            self.context.add(source, text)
        return text, ttft

    def _build_prompt(self, source: str, use_context: bool,
                      is_continuation: bool) -> str:
        parts: list[str] = []
        if self.glossary is not None and use_context:
            parts.append(self.glossary.prompt_line(source))
        if is_continuation:
            # "This continues the previous sentence" alone makes the model
            # restate the previous line -- observed in replay, where a
            # trailing fragment re-emitted the whole prior translation.
            # The instruction has to forbid that explicitly.
            parts.append("[Sentence fragment. Translate ONLY the text below. "
                         "Do NOT repeat or re-translate anything already "
                         "translated.]\n")
        parts.append(source)
        if self.wants_no_think:
            parts.append(NO_THINK)
        return "".join(parts)

    def close(self) -> None:
        self._close_chat()
