"""Console setup.

Windows consoles default to a legacy ANSI code page (GBK on a Chinese install),
so printing the Chinese translation -- the entire point of this app -- raises
UnicodeEncodeError. Every entry point calls `setup_console()` first.

Also enables ANSI escape processing so the CLI frontend can redraw a line in
place on older consoles.
"""
from __future__ import annotations

import logging
import sys

_ANSI_ENABLED = False


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


def setup_console(log_level: str = "info") -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):         # redirected/not a TTY
            pass
    _enable_ansi()

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-5s %(name)-12s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    # OpenVINO and friends are chatty at INFO.
    for noisy in ("openvino", "onnxruntime", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
