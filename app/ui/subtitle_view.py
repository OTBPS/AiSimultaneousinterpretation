"""The subtitle text widget.

Painted by hand rather than assembled from QLabels: the overlay sits on top of
arbitrary video, so every glyph needs an outline to stay readable against both
white slides and dark footage, and tokens arrive ~15 times a second, which is
too often to rebuild a layout tree for.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import (QColor, QFont, QFontMetrics, QPainter, QPainterPath,
                           QPen)
from PySide6.QtWidgets import QWidget

TEXT = QColor(255, 255, 255)
TEXT_PARTIAL = QColor(190, 200, 215)
TEXT_SOURCE = QColor(150, 158, 172)
OUTLINE = QColor(0, 0, 0, 235)
BACKDROP = QColor(16, 18, 22)
CARET = QColor(120, 200, 255)


@dataclass(slots=True)
class ViewLine:
    line_id: int
    source: str = ""
    target: str = ""
    done: bool = False
    skipped: bool = False


class SubtitleView(QWidget):
    def __init__(self, font_size: int = 24, max_lines: int = 3,
                 show_source: bool = True, opacity: float = 0.85,
                 parent=None) -> None:
        super().__init__(parent)
        self.max_lines = max(1, max_lines)
        self.show_source = show_source
        self.backdrop_alpha = opacity
        self._lines: list[ViewLine] = []
        self._partial = ""
        self._status = ""
        self._caret = False

        self._font = QFont("Microsoft YaHei UI", font_size, QFont.Weight.DemiBold)
        self._font_src = QFont("Segoe UI", max(10, int(font_size * 0.62)))
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setMinimumHeight(120)

    # ------------------------------------------------------------- model
    def set_status(self, text: str) -> None:
        self._status = text
        self.update()

    def set_partial(self, text: str) -> None:
        self._partial = text
        self.update()

    def start_line(self, line_id: int, source: str) -> None:
        self._partial = ""
        self._lines.append(ViewLine(line_id, source))
        del self._lines[:-self.max_lines]
        self.update()

    def append_token(self, line_id: int, text: str) -> None:
        line = self._find(line_id)
        if line is None:
            line = ViewLine(line_id)
            self._lines.append(line)
            del self._lines[:-self.max_lines]
        line.target += text
        self._caret = True
        self.update()

    def finish_line(self, line_id: int, target: str) -> None:
        line = self._find(line_id)
        if line is not None:
            if target:
                line.target = target
            line.done = True
        self._caret = any(not ln.done for ln in self._lines)
        self.update()

    def skip_line(self, line_id: int, source: str) -> None:
        line = self._find(line_id)
        if line is None:
            line = ViewLine(line_id, source)
            self._lines.append(line)
            del self._lines[:-self.max_lines]
        line.skipped = True
        line.done = True
        self.update()

    def clear(self) -> None:
        self._lines.clear()
        self._partial = ""
        self.update()

    def _find(self, line_id: int) -> ViewLine | None:
        for ln in reversed(self._lines):
            if ln.line_id == line_id:
                return ln
        return None

    # ------------------------------------------------------------- paint
    def paintEvent(self, _event) -> None:  # noqa: N802 (Qt naming)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing)

        rect = QRectF(self.rect())
        backdrop = QColor(BACKDROP)
        backdrop.setAlphaF(self.backdrop_alpha * 0.72)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(backdrop)
        p.drawRoundedRect(rect.adjusted(2, 2, -2, -2), 12, 12)

        pad = 16.0
        y = rect.height() - pad
        fm = QFontMetrics(self._font)
        fm_src = QFontMetrics(self._font_src)

        # Bottom-up: newest line sits at the bottom, older ones scroll away.
        for ln in reversed(self._lines):
            if ln.skipped:
                y = self._draw(p, self._font_src, fm_src,
                               f"(… skipped) {ln.source}", pad, y,
                               QColor(200, 170, 90))
                continue
            text = ln.target or ("…" if not ln.done else "")
            if text:
                colour = TEXT if ln.done else TEXT_PARTIAL
                if not ln.done and self._caret:
                    text += " ▍"
                y = self._draw(p, self._font, fm, text, pad, y, colour)
            if self.show_source and ln.source:
                y = self._draw(p, self._font_src, fm_src, ln.source, pad, y,
                               TEXT_SOURCE)
            if y < pad:
                break

        if self._partial and self.show_source:
            self._draw(p, self._font_src, fm_src, f"… {self._partial}", pad, y,
                       TEXT_SOURCE)
        elif self._status and not self._lines:
            self._draw(p, self._font_src, fm_src, self._status, pad,
                       rect.height() / 2 + fm_src.height() / 2, TEXT_SOURCE)
        p.end()

    def _draw(self, p: QPainter, font: QFont, fm: QFontMetrics, text: str,
              pad: float, y: float, colour: QColor) -> float:
        """Draw wrapped, outlined text upward from `y`. Returns the new y."""
        width = int(self.width() - pad * 2)
        if width <= 0 or not text:
            return y
        rows = self._wrap(text, fm, width)
        line_h = fm.height() + 2
        y -= line_h * len(rows)
        top = y
        for i, row in enumerate(rows):
            path = QPainterPath()
            path.addText(pad, top + line_h * (i + 1) - fm.descent(), font, row)
            p.setPen(QPen(OUTLINE, 3.0, Qt.PenStyle.SolidLine,
                          Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(path)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(colour)
            p.drawPath(path)
        return y - 4

    @staticmethod
    def _wrap(text: str, fm: QFontMetrics, width: int) -> list[str]:
        """Wrap for mixed CJK/Latin: break on spaces, fall back to characters."""
        rows: list[str] = []
        cur = ""
        for ch in text:
            trial = cur + ch
            if fm.horizontalAdvance(trial) <= width:
                cur = trial
                continue
            cut = cur.rfind(" ")
            # Only break at a space if it is reasonably near the end, else the
            # line would be left half empty (common with long CJK runs).
            if cut > len(cur) * 0.6:
                rows.append(cur[:cut])
                cur = cur[cut + 1:] + ch
            else:
                rows.append(cur)
                cur = ch
        if cur:
            rows.append(cur)
        return rows
