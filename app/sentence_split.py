"""Split an ASR transcript into translation units.

Whisper large-v3-turbo emits punctuation, so one committed segment often holds
several sentences. Translating them separately is strictly better here:
first-token latency is per-unit, and at 15.5 tok/s a long unit visibly lags.

Guards against the two degenerate cases: a wall of text with no punctuation
(hard-wrapped at `max_chars`) and a flood of two-word fragments (merged up to
`min_chars`).
"""
from __future__ import annotations

import re

_TERMINATORS = ".?!…"
# Split after a terminator + whitespace, but not on common abbreviations or
# decimals ("Dr. Smith", "12.5 percent", "e.g. this").
_ABBREV = {"mr", "mrs", "ms", "dr", "prof", "st", "vs", "etc", "e.g", "i.e",
           "fig", "no", "inc", "ltd", "jr", "sr", "approx"}
_SPLIT_RE = re.compile(r"(?<=[.?!…])\s+")


def _is_false_split(left: str) -> bool:
    if not left:
        return True
    if left[-1] != ".":
        return False
    tail = re.split(r"[\s(]", left)[-1].rstrip(".").lower()
    if tail in _ABBREV:
        return True
    return bool(re.search(r"\d\.$", left))        # "12." of "12.5"


def split_sentences(text: str, min_chars: int = 12,
                    max_chars: int = 220) -> list[str]:
    text = " ".join((text or "").split())
    if not text:
        return []

    parts: list[str] = []
    for piece in _SPLIT_RE.split(text):
        if parts and _is_false_split(parts[-1]):
            parts[-1] = f"{parts[-1]} {piece}"
        else:
            parts.append(piece)

    # Merge fragments that are too short to translate well on their own.
    merged: list[str] = []
    for p in parts:
        if merged and (len(merged[-1]) < min_chars
                       or merged[-1][-1] not in _TERMINATORS):
            merged[-1] = f"{merged[-1]} {p}"
        else:
            merged.append(p)

    # Hard-wrap anything still oversized (speaker never punctuated).
    out: list[str] = []
    for p in merged:
        while len(p) > max_chars:
            cut = p.rfind(" ", 0, max_chars)
            if cut <= 0:
                cut = max_chars
            out.append(p[:cut].strip())
            p = p[cut:].strip()
        if p:
            out.append(p)
    return out
