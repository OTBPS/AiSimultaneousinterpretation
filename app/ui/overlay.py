"""Floating subtitle overlay (the frontend).

Consumes the same EventBus the CLI does -- the pipeline has no idea this
exists. Model loading happens on a worker thread so the window paints
immediately instead of freezing for the ~3.4 s warm start.
"""
from __future__ import annotations

import logging
import sys
import threading
import time
from pathlib import Path

from PySide6.QtCore import QPoint, QRect, Qt, QTimer
from PySide6.QtGui import QAction, QActionGroup, QCursor, QGuiApplication, QIcon
from PySide6.QtWidgets import (QApplication, QMenu, QSystemTrayIcon, QVBoxLayout,
                               QWidget)

from ..bus import EventBus
from ..config import Config
from ..pipeline import Pipeline
from ..types import (LineDoneEvent, LineSkippedEvent, LineStartEvent,
                     MetricsEvent, PartialEvent, StatusEvent, TokenEvent)
from ..ui_state import MAX_FONT, MIN_FONT, MIN_HEIGHT, MIN_WIDTH, UiState
from .subtitle_view import SubtitleView

log = logging.getLogger("ui")

# Drain fast only while subtitles are moving. A fixed 30 Hz timer wakes
# the UI 30 times a second through long silences for nothing; MT streams
# at ~15 tok/s so 20 Hz is already faster than the content arrives.
DRAIN_HZ_ACTIVE = 20
DRAIN_HZ_IDLE = 5
IDLE_AFTER_S = 2.0
ICON_PATH = Path(__file__).resolve().parent.parent.parent / "assets" / "app.ico"

# How close to an edge counts as "grab to resize" rather than "drag to move".
RESIZE_MARGIN = 10
SAVE_DEBOUNCE_MS = 800

_CURSORS = {
    "l": Qt.CursorShape.SizeHorCursor, "r": Qt.CursorShape.SizeHorCursor,
    "t": Qt.CursorShape.SizeVerCursor, "b": Qt.CursorShape.SizeVerCursor,
    "tl": Qt.CursorShape.SizeFDiagCursor, "br": Qt.CursorShape.SizeFDiagCursor,
    "tr": Qt.CursorShape.SizeBDiagCursor, "bl": Qt.CursorShape.SizeBDiagCursor,
}


def is_activity(ev) -> bool:
    """Should this event hold the UI at the fast drain rate?

    A MetricsEvent arrives every 2 s whether or not anything is happening, so
    on its own it is not activity -- otherwise the UI would never idle down.
    A backlog, however, is activity.

    Module-level and pure so the rule is tested directly rather than mirrored
    in a test (a test that re-implements the logic passes even when the real
    code is wrong).
    """
    if isinstance(ev, MetricsEvent):
        return ev.backlog > 0
    return True


def app_icon(widget: QWidget) -> QIcon:
    """The real .ico. QIcon.fromTheme returns null on Windows, which silently
    degrades the tray to a generic glyph the user cannot find."""
    if ICON_PATH.exists():
        icon = QIcon(str(ICON_PATH))
        if not icon.isNull():
            return icon
    return widget.style().standardIcon(
        widget.style().StandardPixmap.SP_MediaVolume)


def _set_click_through(widget: QWidget, enabled: bool) -> None:
    """Toggle WS_EX_TRANSPARENT in place.

    Flipping Qt's WindowTransparentForInput instead would destroy and recreate
    the native window, losing position, z-order and the tray association.
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes

        GWL_EXSTYLE = -20
        WS_EX_TRANSPARENT = 0x20
        WS_EX_LAYERED = 0x80000
        hwnd = int(widget.winId())
        user32 = ctypes.windll.user32
        get_ = getattr(user32, "GetWindowLongPtrW", user32.GetWindowLongW)
        set_ = getattr(user32, "SetWindowLongPtrW", user32.SetWindowLongW)
        style = get_(hwnd, GWL_EXSTYLE)
        style = (style | WS_EX_TRANSPARENT | WS_EX_LAYERED) if enabled \
            else (style & ~WS_EX_TRANSPARENT)
        set_(hwnd, GWL_EXSTYLE, style)
    except Exception as e:                                # noqa: BLE001
        log.warning("click-through toggle failed: %s", e)


class Overlay(QWidget):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.bus = EventBus()
        self.pipeline = Pipeline(cfg, self.bus)
        self._drag_from: QPoint | None = None
        self._resize_edge = ""
        self._resize_from: QPoint | None = None
        self._resize_geom: QRect | None = None
        self._click_through = False
        self._paused = False
        self._announced = False
        self._edit_mode = False

        self._state_path = cfg.root / "ui_state.json"
        self._state = UiState.load(self._state_path)

        self.setWindowTitle("同声传译")
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint
                            | Qt.WindowType.WindowStaysOnTopHint
                            | Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self.view = SubtitleView(self._state.clamped_font() or cfg.ui.font_size,
                                 cfg.ui.max_lines,
                                 cfg.ui.show_source, cfg.ui.opacity)
        self.setMinimumSize(MIN_WIDTH, MIN_HEIGHT)
        self.setMouseTracking(True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.view)

        self._place()
        self.view.set_status("loading models…")

        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.timeout.connect(self._save_state)

        self._tray = self._build_tray()
        self._last_event_at = 0.0
        self._drain_hz = DRAIN_HZ_ACTIVE
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._drain)
        self._timer.start(int(1000 / self._drain_hz))

        threading.Thread(target=self._start_pipeline, daemon=True,
                         name="pipeline-start").start()

    # ------------------------------------------------------------ layout
    def _place(self) -> None:
        screen = QGuiApplication.primaryScreen().availableGeometry()
        if self._state.has_geometry:
            # Clamped, because a position saved on a monitor that is no longer
            # attached would put the window off-screen -- which looks exactly
            # like the app failing to start.
            st = self._state.clamp_to(screen.x(), screen.y(),
                                      screen.width(), screen.height())
            self.resize(st.width, st.height)
            self.move(st.x, st.y)
            return
        w = int(screen.width() * 0.62)
        h = 190
        self.resize(w, h)
        self.move(screen.center().x() - w // 2, screen.bottom() - h - 70)

    def _save_state_soon(self) -> None:
        """Debounced: a drag fires hundreds of move events."""
        self._save_timer.start(SAVE_DEBOUNCE_MS)

    def _save_state(self) -> None:
        g = self.geometry()
        UiState(g.x(), g.y(), g.width(), g.height(),
                self.view.font_size).save(self._state_path)

    def reset_layout(self) -> None:
        self._state = UiState()
        self.view.set_font_size(self.cfg.ui.font_size)
        self._place()
        self._save_state()

    # ------------------------------------------------------------- tray
    def _build_tray(self) -> QSystemTrayIcon:
        tray = QSystemTrayIcon(self)
        tray.setIcon(app_icon(self))
        tray.setToolTip("同声传译 EN→ZH")

        menu = QMenu()

        self._act_pause = QAction("暂停", self, checkable=True)
        self._act_pause.triggered.connect(self._toggle_pause)
        menu.addAction(self._act_pause)
        menu.addSeparator()

        src_menu = menu.addMenu("音频来源")
        group = QActionGroup(self)
        group.setExclusive(True)
        for label, value in (("系统播放声音（回录）", "loopback"), ("麦克风", "mic")):
            act = QAction(label, self, checkable=True)
            act.setChecked(self.cfg.audio.source == value)
            act.triggered.connect(lambda _c, v=value: self._switch_source(v))
            group.addAction(act)
            src_menu.addAction(act)

        model_menu = menu.addMenu("翻译模型")
        for label, path in (("Qwen3-4B（质量优先）", self.cfg.mt.model),
                            ("Qwen2.5-1.5B（速度优先）", self.cfg.mt.fast_model)):
            if not path:
                continue
            act = QAction(label, self)
            act.setEnabled(False)   # switching reloads a model; Phase 4 work
            model_menu.addAction(act)

        menu.addSeparator()
        act_src = QAction("显示英文原文", self, checkable=True)
        act_src.setChecked(self.cfg.ui.show_source)
        act_src.triggered.connect(self._toggle_source_text)
        menu.addAction(act_src)

        act_edit = QAction("调整位置和大小", self, checkable=True)
        act_edit.setToolTip("拖动窗口移动，拖边缘缩放，滚轮调字号")
        act_edit.triggered.connect(self._toggle_edit_mode)
        menu.addAction(act_edit)
        self._act_edit = act_edit

        act_reset = QAction("恢复默认位置和字号", self)
        act_reset.triggered.connect(self.reset_layout)
        menu.addAction(act_reset)

        act_ct = QAction("鼠标点击穿透", self, checkable=True)
        act_ct.setChecked(self.cfg.ui.click_through)
        act_ct.triggered.connect(self._toggle_click_through)
        menu.addAction(act_ct)
        self._act_click_through = act_ct

        menu.addSeparator()
        act_clear = QAction("清屏", self)
        act_clear.triggered.connect(self.view.clear)
        menu.addAction(act_clear)

        act_quit = QAction("退出", self)
        act_quit.triggered.connect(self._quit)
        menu.addAction(act_quit)

        tray.setContextMenu(menu)
        tray.activated.connect(
            lambda reason: self._toggle_visible()
            if reason == QSystemTrayIcon.ActivationReason.Trigger else None)
        tray.show()
        return tray

    # ----------------------------------------------------------- actions
    def _toggle_visible(self) -> None:
        self.setVisible(not self.isVisible())

    def _toggle_pause(self, checked: bool) -> None:
        self._paused = checked
        self._act_pause.setText("继续" if checked else "暂停")
        if self.pipeline.ready:
            self.pipeline.set_paused(checked)

    def _toggle_source_text(self, checked: bool) -> None:
        self.cfg.ui.show_source = checked
        self.view.show_source = checked
        self.view.update()

    def _toggle_click_through(self, checked: bool) -> None:
        self._click_through = checked
        _set_click_through(self, checked)

    def _toggle_edit_mode(self, checked: bool) -> None:
        """Make the window grabbable and say so.

        Click-through is on by default, which means mouse events never reach
        the window at all -- drag and resize are dead until it is off. Rather
        than expect the user to work that out, this flips it and draws a frame
        so there is something visible to aim at.
        """
        self._edit_mode = checked
        self.view.edit_mode = checked
        self.view.update()
        if checked:
            self._toggle_click_through(False)
            self._act_click_through.setChecked(False)
            self._tray.showMessage(
                "调整模式",
                "拖动窗口移动位置\n拖动边缘或四角改变大小\n滚轮调整字号\n"
                "调好后再点一次「调整位置和大小」",
                app_icon(self), 6000)
        else:
            self._save_state()
            if self.cfg.ui.click_through:
                self._toggle_click_through(True)
                self._act_click_through.setChecked(True)

    def _switch_source(self, value: str) -> None:
        if not self.pipeline.ready:
            return
        try:
            self.pipeline.switch_source(value, None)
        except Exception as e:                            # noqa: BLE001
            self.view.set_status(f"切换失败: {e}")

    def _quit(self) -> None:
        self._timer.stop()
        try:
            self.pipeline.stop()
        finally:
            self._tray.hide()
            QApplication.quit()

    # ----------------------------------------------------------- startup
    def _start_pipeline(self) -> None:
        try:
            self.pipeline.start()
        except Exception as e:                            # noqa: BLE001
            log.exception("pipeline failed to start")
            self.bus.emit(StatusEvent("error", str(e)))

    # ------------------------------------------------------------- drain
    def _set_drain_rate(self, hz: int) -> None:
        if hz != self._drain_hz:
            self._drain_hz = hz
            self._timer.setInterval(int(1000 / hz))

    def _drain(self) -> None:
        saw_event = False
        for ev in self.bus.drain(256):
            saw_event |= is_activity(ev)
            if isinstance(ev, TokenEvent):
                self.view.append_token(ev.line_id, ev.text)
            elif isinstance(ev, LineStartEvent):
                self.view.start_line(ev.line_id, ev.source)
            elif isinstance(ev, LineDoneEvent):
                self.view.finish_line(ev.line_id, ev.target)
            elif isinstance(ev, PartialEvent):
                self.view.set_partial(ev.text)
            elif isinstance(ev, LineSkippedEvent):
                self.view.skip_line(ev.line_id, ev.source)
            elif isinstance(ev, StatusEvent):
                self.view.set_status(ev.detail or ev.state)
                if ev.state == "error":
                    self._tray.showMessage("同声传译 — 启动失败", ev.detail,
                                           QSystemTrayIcon.MessageIcon.Critical,
                                           10000)
                elif ev.state == "ready" and not self._announced:
                    # The overlay is translucent and has no taskbar button, so
                    # without this the app looks like it never started.
                    self._announced = True
                    self._tray.showMessage(
                        "同声传译已启动",
                        f"{ev.detail}\n字幕窗在屏幕底部；右键托盘图标可切换音频来源。",
                        app_icon(self), 6000)
                # Click-through is only armed once we are actually running, so
                # a startup failure still leaves a window the user can click.
                if (ev.state == "ready" and self.cfg.ui.click_through
                        and not self._click_through):
                    self._toggle_click_through(True)
            elif isinstance(ev, MetricsEvent):
                if ev.backlog > 2:
                    self._tray.setToolTip(
                        f"同声传译 — 积压 {ev.backlog}"
                        + ("（已切到快速模型）" if ev.using_fast_model else ""))
                else:
                    self._tray.setToolTip("同声传译 EN→ZH")

        now = time.monotonic()
        if saw_event:
            self._last_event_at = now
            self._set_drain_rate(DRAIN_HZ_ACTIVE)
        elif now - self._last_event_at > IDLE_AFTER_S:
            self._set_drain_rate(DRAIN_HZ_IDLE)

    # -------------------------------------------------------- drag/move
    def _edge_at(self, pos: QPoint) -> str:
        """Which edge the cursor is on, as a string like "br", or "" for none.

        The window is frameless, so Qt gives us no resize border; it has to be
        derived from the hit position.
        """
        m = RESIZE_MARGIN
        vertical = ""
        if pos.y() <= m:
            vertical = "t"
        elif pos.y() >= self.height() - m:
            vertical = "b"

        horizontal = ""
        if pos.x() <= m:
            horizontal = "l"
        elif pos.x() >= self.width() - m:
            horizontal = "r"

        return vertical + horizontal

    def mousePressEvent(self, event) -> None:               # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton:
            return
        edge = self._edge_at(event.position().toPoint())
        if edge:
            self._resize_edge = edge
            self._resize_from = event.globalPosition().toPoint()
            self._resize_geom = QRect(self.geometry())
        else:
            self._drag_from = event.globalPosition().toPoint() - self.pos()

    def mouseMoveEvent(self, event) -> None:                # noqa: N802
        if self._resize_edge and self._resize_from is not None:
            self._apply_resize(event.globalPosition().toPoint())
            return
        if self._drag_from is not None:
            self.move(event.globalPosition().toPoint() - self._drag_from)
            self._save_state_soon()
            return
        # Not dragging: just advertise what a click here would do.
        edge = self._edge_at(event.position().toPoint())
        self.setCursor(QCursor(_CURSORS.get(edge, Qt.CursorShape.SizeAllCursor)))

    def _apply_resize(self, gpos: QPoint) -> None:
        dx = gpos.x() - self._resize_from.x()
        dy = gpos.y() - self._resize_from.y()
        g = QRect(self._resize_geom)
        if "l" in self._resize_edge:
            g.setLeft(min(g.left() + dx, g.right() - MIN_WIDTH + 1))
        if "r" in self._resize_edge:
            g.setRight(max(g.right() + dx, g.left() + MIN_WIDTH - 1))
        if "t" in self._resize_edge:
            g.setTop(min(g.top() + dy, g.bottom() - MIN_HEIGHT + 1))
        if "b" in self._resize_edge:
            g.setBottom(max(g.bottom() + dy, g.top() + MIN_HEIGHT - 1))
        self.setGeometry(g)
        self._save_state_soon()

    def mouseReleaseEvent(self, _event) -> None:            # noqa: N802
        self._drag_from = None
        self._resize_edge = ""
        self._resize_from = None
        self._resize_geom = None
        self._save_state_soon()

    def wheelEvent(self, event) -> None:                    # noqa: N802
        """Wheel zooms the subtitle text.

        Font size, not window scale: the window is a viewport onto scrolling
        text, so making the glyphs bigger is what the user actually wants when
        the subtitles are hard to read from across the room.
        """
        step = 1 if event.angleDelta().y() > 0 else -1
        size = max(MIN_FONT, min(MAX_FONT, self.view.font_size + step))
        if size != self.view.font_size:
            self.view.set_font_size(size)
            self._tray.setToolTip(f"同声传译 EN→ZH — 字号 {size}")
            self._save_state_soon()
        event.accept()

    def leaveEvent(self, _event) -> None:                   # noqa: N802
        self.unsetCursor()

    def closeEvent(self, event) -> None:                    # noqa: N802
        self._quit()
        event.accept()


def run_gui(cfg: Config) -> int:
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)     # tray keeps it alive
    overlay = Overlay(cfg)
    overlay.show()
    return app.exec()
