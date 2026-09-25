"""Persisted subtitle window geometry and font size.

Deliberately NOT stored in config.yaml. That file is hand-edited and carries
explanatory comments; rewriting it from code every time the user nudges the
window would strip them. Geometry is machine state, not configuration, so it
gets its own file.

`clamp_to` exists because restoring a saved position blindly is a good way to
make the window invisible: unplug the external monitor it was dragged to, or
change display scaling, and the coordinates now point off-screen. The symptom
-- app runs, nothing visible -- is indistinguishable from a crash.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

log = logging.getLogger("ui_state")

MIN_WIDTH = 320
MIN_HEIGHT = 90
MIN_FONT = 12
MAX_FONT = 72


@dataclass(slots=True)
class UiState:
    x: int | None = None
    y: int | None = None
    width: int | None = None
    height: int | None = None
    font_size: int | None = None

    @property
    def has_geometry(self) -> bool:
        return None not in (self.x, self.y, self.width, self.height)

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path) -> "UiState":
        p = Path(path)
        if not p.exists():
            return cls()
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            log.warning("ignoring unreadable ui state (%s): %s", p.name, e)
            return cls()
        known = {f for f in cls.__slots__}
        return cls(**{k: v for k, v in raw.items() if k in known})

    def save(self, path: str | Path) -> None:
        try:
            Path(path).write_text(
                json.dumps(asdict(self), indent=2), encoding="utf-8")
        except OSError as e:
            log.warning("cannot save ui state: %s", e)

    # ------------------------------------------------------------------
    def clamp_to(self, sx: int, sy: int, sw: int, sh: int) -> "UiState":
        """Force the window back onto the given screen rectangle.

        Requires a visible sliver to remain grabbable, so a window saved on a
        monitor that no longer exists reappears somewhere reachable instead of
        silently off-screen.
        """
        if not self.has_geometry:
            return self

        w = max(MIN_WIDTH, min(int(self.width), sw))
        h = max(MIN_HEIGHT, min(int(self.height), sh))
        # Keep at least 80x40 of the window inside the screen.
        x = max(sx - w + 80, min(int(self.x), sx + sw - 80))
        y = max(sy, min(int(self.y), sy + sh - 40))
        return UiState(x, y, w, h, self.clamped_font())

    def clamped_font(self) -> int | None:
        if self.font_size is None:
            return None
        return max(MIN_FONT, min(MAX_FONT, int(self.font_size)))
