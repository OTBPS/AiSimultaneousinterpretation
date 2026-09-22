"""Rolling conversational context for the MT model.

Context materially improves pronoun and terminology consistency, and it is
cheap *in the right shape*: prefill runs at ~1360 tok/s on this iGPU while
decode runs at 15.5 tok/s, so re-reading history is ~90x cheaper than
regenerating it.

It is not free, though. Rather than re-sending the last N pairs as text on
every request (which re-prefills them every time), we keep a persistent
`start_chat()` session so the KV cache is reused and only the new sentence is
prefilled. The cost is unbounded KV growth, so the session is recycled every
`reset_every` lines -- one ~150 ms hiccup instead of a slow drift upward.
"""
from __future__ import annotations

from collections import deque


class Context:
    def __init__(self, pairs: int = 3, reset_every: int = 8) -> None:
        self.pairs = max(0, pairs)
        self.reset_every = max(1, reset_every)
        self._history: deque[tuple[str, str]] = deque(maxlen=max(1, self.pairs))
        self._since_reset = 0

    def add(self, source: str, target: str) -> None:
        if self.pairs and source and target:
            self._history.append((source, target))
        self._since_reset += 1

    def needs_reset(self) -> bool:
        return self._since_reset >= self.reset_every

    def mark_reset(self) -> None:
        self._since_reset = 0

    def clear(self) -> None:
        self._history.clear()
        self._since_reset = 0

    def recent(self) -> list[tuple[str, str]]:
        return list(self._history)

    def preamble(self) -> str:
        """Context block for a *fresh* chat session (re-seeding after a reset).

        Inside a live session the history is already in the KV cache, so this
        is only used to warm a new one.
        """
        if not self._history:
            return ""
        lines = [f"{en}\n-> {zh}" for en, zh in list(self._history)[-2:]]
        return ("[Recent context, for consistency only -- do not retranslate]\n"
                + "\n".join(lines) + "\n\n")
