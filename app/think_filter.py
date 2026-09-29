"""Strip Qwen3 reasoning tags out of a *token stream*.

Measured: with thinking on, one sentence took 43.65 s / 674 tokens instead of
1.61 s / 22. We always append ` /no_think`, but the model still emits an empty
`<think></think>` pair, so it has to be removed before display.

A plain `str.replace` on the final string is not enough -- we stream to the UI,
and a tag can arrive split across token boundaries (`<thi` + `nk>`).

The harder case, and the one that silently ate most translations: the model
frequently emits the OPENING tag and then the answer, with no closing tag at
all --

    <think>\\n\\n请看下一张。

An earlier version treated `<think>` as "suppress until `</think>`", so when
the close never arrived it discarded the entire translation. The subtitle line
then rendered its English source with nothing under it, which is exactly the
"sometimes only English shows" report. 20 of 28 sample utterances were lost
this way.

The fix relies on `/no_think` semantics: a legitimate no-think block is
*empty*. So Han text appearing inside the block is not reasoning, it is the
answer, and the block is abandoned as soon as that is clear -- which also
keeps it streaming rather than arriving all at once at the end.
"""
from __future__ import annotations

import re

OPEN, CLOSE = "<think>", "</think>"
# Junk some models prepend before the actual translation.
_PREFIXES = ("翻译：", "翻译:", "译文：", "译文:", "Translation:", "中文：", "中文:")
_HAN = re.compile(r"[一-鿿]")


def _could_start_tag(buf: str) -> bool:
    return OPEN.startswith(buf) or CLOSE.startswith(buf)


class ThinkFilter:
    """Feed chunks in, get displayable chunks out."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._buf = ""          # possible partial tag, held back
        self._in_think = False
        self._think_buf = ""    # content seen inside an unclosed think block
        self._emitted = False   # have we released any visible text yet?

    # ------------------------------------------------------------------
    def feed(self, chunk: str) -> str:
        out: list[str] = []
        for ch in chunk:
            self._buf += ch

            if self._buf == OPEN:
                self._in_think, self._buf = True, ""
                self._think_buf = ""
                continue
            if self._buf == CLOSE:
                # Properly closed: whatever was inside really was reasoning.
                self._in_think, self._buf = False, ""
                self._think_buf = ""
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
                continue

            # Inside a think block. Under /no_think that block should be
            # empty, so Han text preceded by nothing but whitespace is the
            # answer with a missing close tag -- release it and keep
            # streaming. Real reasoning opens with English prose ("Okay, the
            # user wants..."), which fails this test and stays suppressed.
            self._think_buf += text
            if _HAN.search(self._think_buf) and self._looks_like_answer():
                self._in_think = False
                out.append(self._think_buf)
                self._think_buf = ""

        return self._clean("".join(out))

    def _looks_like_answer(self) -> bool:
        """True if the block holds only whitespace and then Han text.

        Cannot be perfect: at the moment the first Han character arrives,
        "<think>\n\n你好" (answer, close tag never comes) and
        "<think>你好吗</think>..." (Chinese reasoning) are indistinguishable --
        the close tag has not been seen yet either way. This rule favours the
        case that actually occurs: with /no_think the block is meant to be
        empty, and Qwen3 reasons in English for this prompt.
        """
        head = self._think_buf[:self._think_buf.find(
            _HAN.search(self._think_buf).group())]
        return head.strip() == ""

    def flush(self) -> str:
        """Release anything still held back (end of generation)."""
        tail = ""
        if self._in_think:
            # Never closed. Keep it only if it looks like the answer; English
            # here is genuine reasoning that got truncated, and showing that
            # as a subtitle would be worse than showing nothing.
            if _HAN.search(self._think_buf):
                tail = self._think_buf
            self._think_buf = ""
            self._in_think = False
        else:
            tail = self._buf
        self._buf = ""
        return self._clean(tail)

    # ------------------------------------------------------------------
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
