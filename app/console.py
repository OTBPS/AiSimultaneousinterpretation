"""Console and logging setup.

Two Windows-specific problems are solved here.

1. Consoles default to a legacy ANSI code page (GBK on a Chinese install), so
   printing the Chinese translation -- the entire point of this app -- raises
   UnicodeEncodeError.

2. Launched from a desktop shortcut via `pythonw.exe` there is no console at
   all: `sys.stdout` and `sys.stderr` are None, and anything written to them
   is lost. A startup failure would then be completely silent, so we always
   also log to a size-capped file.

The file log is bounded (1 MB x 2) on purpose: the install footprint must not
grow on its own. See app/storage.py for the same policy applied to the model
cache.
"""
from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

_ANSI_ENABLED = False
LOG_MAX_BYTES = 1_000_000
LOG_BACKUPS = 1
# Relative to the project root. Defined here so run.py and the tray menu
# cannot drift apart about where the log lives.
LOG_RELATIVE = Path("logs") / "app.log"


def _enable_ansi() -> None:
    global _ANSI_ENABLED
    if _ANSI_ENABLED or sys.platform != "win32":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        for handle_id in (-11, -12):                 # stdout, stderr
            handle = kernel32.GetStdHandle(handle_id)
            mode = ctypes.c_uint32()
            if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                kernel32.SetConsoleMode(handle, mode.value | 0x0004)
        _ANSI_ENABLED = True
    except Exception:                                # noqa: BLE001
        pass


def has_console() -> bool:
    return sys.stdout is not None and sys.stderr is not None


def setup_console(log_level: str = "info",
                  log_file: str | Path | None = None) -> None:
    for stream in (sys.stdout, sys.stderr):
        if stream is None:                           # pythonw: no console
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):         # redirected / not a TTY
            pass
    _enable_ansi()

    level = getattr(logging, log_level.upper(), logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-5s %(name)-12s %(message)s", "%H:%M:%S")

    root = logging.getLogger()
    root.setLevel(level)
    for h in list(root.handlers):                    # idempotent re-setup
        root.removeHandler(h)

    if sys.stderr is not None:
        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setFormatter(fmt)
        root.addHandler(stream_handler)

    if log_file is not None:
        path = Path(log_file)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                path, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUPS,
                encoding="utf-8")
            file_handler.setFormatter(logging.Formatter(
                "%(asctime)s %(levelname)-5s %(name)-12s %(message)s"))
            root.addHandler(file_handler)
        except OSError:
            pass                                     # never block startup on a log

    # OpenVINO and friends are chatty at INFO.
    for noisy in ("openvino", "onnxruntime", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def install_crash_log(log_file: str | Path) -> None:
    """Make an unhandled exception visible when there is no console.

    Without this, double-clicking the desktop shortcut on a broken install
    just does nothing at all.
    """
    log = logging.getLogger("crash")

    def hook(exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        log.critical("unhandled exception", exc_info=(exc_type, exc, tb))
        if not has_console():
            try:
                import ctypes

                ctypes.windll.user32.MessageBoxW(
                    None,
                    f"{exc_type.__name__}: {exc}\n\n详细日志：\n{log_file}",
                    "同声传译 — 启动失败", 0x10)
            except Exception:                        # noqa: BLE001
                pass

    sys.excepthook = hook
