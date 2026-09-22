"""Generate assets/app.ico for the desktop shortcut.

Qt renders the glyph; the .ico container is assembled by hand because Qt's ICO
support is read-only on some builds. Modern Windows accepts PNG-compressed
icon entries, so each size is just a PNG blob with a directory entry.
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QBuffer, QByteArray, QRectF, Qt  # noqa: E402
from PySide6.QtGui import (QColor, QFont, QImage, QLinearGradient,  # noqa: E402
                           QPainter, QPainterPath)
from PySide6.QtWidgets import QApplication                  # noqa: E402

SIZES = (16, 24, 32, 48, 64, 128, 256)
OUT = ROOT / "assets" / "app.ico"


def render(size: int) -> QByteArray:
    img = QImage(size, size, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(Qt.GlobalColor.transparent)

    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setRenderHint(QPainter.RenderHint.TextAntialiasing)

    r = QRectF(0.5, 0.5, size - 1.0, size - 1.0)
    grad = QLinearGradient(r.topLeft(), r.bottomRight())
    grad.setColorAt(0.0, QColor(28, 34, 46))
    grad.setColorAt(1.0, QColor(14, 18, 26))
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(grad)
    p.drawRoundedRect(r, size * 0.22, size * 0.22)

    # Glyph: 译. Drawn as a path so it scales cleanly down to 16 px.
    font = QFont("Microsoft YaHei UI", int(size * 0.62), QFont.Weight.Bold)
    path = QPainterPath()
    path.addText(0, 0, font, "译")
    bounds = path.boundingRect()
    if bounds.width() > 0 and bounds.height() > 0:
        scale = min(r.width() * 0.62 / bounds.width(),
                    r.height() * 0.62 / bounds.height())
        p.save()
        p.translate(r.center())
        p.scale(scale, scale)
        p.translate(-bounds.center())
        p.setBrush(QColor(120, 205, 255))
        p.drawPath(path)
        p.restore()

    p.end()

    data = QByteArray()
    buf = QBuffer(data)
    buf.open(QBuffer.OpenModeFlag.WriteOnly)
    img.save(buf, "PNG")
    buf.close()
    return data


def main() -> int:
    app = QApplication([])                     # needed for font rendering
    pngs = [(s, bytes(render(s).data())) for s in SIZES]

    header = struct.pack("<HHH", 0, 1, len(pngs))
    entries, blobs = b"", b""
    offset = len(header) + 16 * len(pngs)
    for size, png in pngs:
        dim = 0 if size >= 256 else size
        entries += struct.pack("<BBBBHHII", dim, dim, 0, 0, 1, 32,
                               len(png), offset)
        blobs += png
        offset += len(png)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(header + entries + blobs)
    print(f"wrote {OUT}  ({OUT.stat().st_size / 1024:.1f} KB, "
          f"{len(pngs)} sizes)")
    del app
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
