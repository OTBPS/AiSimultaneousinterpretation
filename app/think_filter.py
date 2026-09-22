"""Strip Qwen3 reasoning tags out of a *token stream*.

Measured: with thinking on, one sentence took 43.65 s / 674 tokens instead of
1.61 s / 22. We always append ` /no_think`, but the model still emits an empty
`<think></think>` pair, so it has to be removed before display.

A plain `str.replace` on the final string is not enough -- we stream to the UI,
and a tag can arrive split across token boundaries (`<thi` + `nk>`). This is a
small incremental state machine that holds back only the bytes that could still
turn out to be part of a tag.
"""
from __future__ import annotations

OPEN, CLOSE = "<think>", "</think>"
# Junk some models prepend before the actual translation.
_PREFIXES = ("翻译：", "翻译:", "译文：", "译文:", "Translation:", "中文：", "中文:")


def _could_start_tag(buf: str) -> bool:
    return OPEN.startswith(buf) or CLOSE.startswith(buf)


class ThinkFilter:
    """Feed chunks in, get displayable chunks out."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._buf = ""          # possible partial tag, held back
        self._in_think = False
        self._emitted = False   # have we released any visible text yet?

    def feed(self, chunk: str) -> str:
        out: list[str] = []
        for ch in chunk:
            self._buf += ch

            if self._buf == OPEN:
                self._in_think, self._buf = True, ""
                continue
            if self._buf == CLOSE:
                self._in_think, self._buf = False, ""
                continue
            if _could_start_tag(self._buf):
                continue                      # still ambiguous, keep holding

            # Not a tag prefix. Release the buffer one char at a time so a `<`
            # that begins later in the buffer still gets a chance to match.
            text, self._buf = self._buf, ""
            if len(text) > 1 and "<" in text[1:]:
                idx = text.index("<", 1)
                text, self._buf = text[:idx], text[idx:]
            if not self._in_think:
                out.append(text)

        return self._clean("".join(out))

    def flush(self) -> str:
        """Release anything still held back (end of generation)."""
        tail = "" if self._in_think else self._buf
        self._buf = ""
        return self._clean(tail)

    def _clean(self, text: str) -> str:
        if self._emitted or not text:
            return text
        stripped = text.lstrip()
        if not stripped:
            return ""                          # swallow leading blank lines
        for p in _PREFIXES:
            if stripped.startswith(p):
                stripped = stripped[len(p):].lstrip()
                break
        stripped = stripped.lstrip('"“')
        if stripped:
            self._emitted = True
        return stripped


def strip_think(text: str) -> str:
    """One-shot convenience for non-streaming callers."""
    f = ThinkFilter()
    return (f.feed(text) + f.flush()).strip().rstrip('"”')
