"""User term map, injected selectively into the MT prompt.

Both Qwen3-4B and Qwen2.5-1.5B mistranslated the idiom "boil the ocean" in
testing, and the 1.5B model rendered "prefill" as 预读取 (prefetch) rather than
预填充. Neither is fixable by a bigger model at this size -- a glossary is.

Only terms that actually occur in the current sentence are injected, so the
file can grow to thousands of entries without costing prompt tokens. Hot
reload on mtime means the user can fix a term mid-meeting.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

log = logging.getLogger("glossary")


class Glossary:
    def __init__(self, path: str | Path, enabled: bool = True) -> None:
        self.path = Path(path)
        self.enabled = enabled
        self._terms: dict[str, str] = {}
        self._regex: re.Pattern | None = None
        self._mtime: float = -1.0
        self.reload()

    # ------------------------------------------------------------------
    def reload(self) -> bool:
        """Re-read if the file changed. Returns True if it was reloaded."""
        if not self.enabled or not self.path.exists():
            return False
        try:
            mtime = self.path.stat().st_mtime
        except OSError:
            return False
        if mtime == self._mtime:
            return False
        self._mtime = mtime

        terms: dict[str, str] = {}
        try:
            for lineno, raw in enumerate(
                    self.path.read_text(encoding="utf-8").splitlines(), 1):
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if "\t" in line:
                    en, zh = line.split("\t", 1)
                elif "=" in line:
                    en, zh = line.split("=", 1)
                else:
                    log.warning("%s:%d: no TAB separator, skipped: %r",
                                self.path.name, lineno, raw[:60])
                    continue
                en, zh = en.strip(), zh.strip()
                if en and zh:
                    terms[en.lower()] = zh
        except OSError as e:
            log.warning("cannot read glossary: %s", e)
            return False

        self._terms = terms
        self._regex = self._compile(terms)
        log.info("glossary loaded: %d terms", len(terms))
        return True

    @staticmethod
    def _compile(terms: dict[str, str]) -> re.Pattern | None:
        if not terms:
            return None
        # Longest first so "boil the ocean" wins over a hypothetical "ocean".
        alts = sorted(map(re.escape, terms), key=len, reverse=True)
        return re.compile(r"(?<!\w)(" + "|".join(alts) + r")(?!\w)",
                          re.IGNORECASE)

    # ------------------------------------------------------------------
    def match(self, text: str) -> dict[str, str]:
        """Terms occurring in `text`, in order of first appearance."""
        if not self.enabled or not self._regex or not text:
            return {}
        found: dict[str, str] = {}
        for m in self._regex.finditer(text):
            surface = m.group(0)
            zh = self._terms.get(surface.lower())
            if zh and surface not in found:
                found[surface] = zh
        return found

    def prompt_line(self, text: str) -> str:
        """The `[Terms] ...` line to prepend to the user message, or ''."""
        hits = self.match(text)
        if not hits:
            return ""
        pairs = "; ".join(f"{en}={zh}" for en, zh in hits.items())
        return f"[Terms] {pairs}\n"

    def __len__(self) -> int:
        return len(self._terms)
