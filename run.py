"""Entry point.

    python run.py --cli                 terminal frontend (default)
    python run.py --gui                 floating subtitle overlay
    python run.py --source mic          override config
    python run.py --device Realtek      pick a capture device by name
    python run.py --devices             list capture devices and exit
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

BOOT_LOG = ROOT / "logs" / "startup_error.txt"


def _die_visibly(stage: str, exc: BaseException) -> None:
    """Report a failure that happens before logging exists.

    Under pythonw there is no stdout or stderr, so an exception raised while
    importing is written nowhere and the process just vanishes. Double-clicking
    the shortcut then appears to do nothing at all -- no window, no error, no
    log line -- which is exactly what was reported, twice, and what cost two
    long debugging sessions.

    So the very first thing this file does is make that case loud.
    """
    import traceback

    detail = f"{stage}\n\n{type(exc).__name__}: {exc}\n\n" \
             + traceback.format_exc()
    try:
        BOOT_LOG.parent.mkdir(parents=True, exist_ok=True)
        BOOT_LOG.write_text(detail, encoding="utf-8")
    except OSError:
        pass
    if sys.stderr is not None:
        print(detail, file=sys.stderr)
    else:
        try:
            import ctypes

            ctypes.windll.user32.MessageBoxW(
                None,
                f"{stage}\n\n{type(exc).__name__}: {exc}\n\n详细信息：\n{BOOT_LOG}",
                "同声传译 — 启动失败", 0x10)
        except Exception:                                  # noqa: BLE001
            pass
    sys.exit(1)


# app.console imports nothing but the standard library, so it is safe to pull
# in before the streams are repaired -- and repairing them is what makes the
# heavier imports below survivable under pythonw.
try:
    from app.console import (LOG_RELATIVE, ensure_std_streams,   # noqa: E402
                             has_console, install_crash_log, setup_console)
except BaseException as _e:                               # noqa: BLE001
    _die_visibly("导入 app.console 失败 / failed to import app.console", _e)

ensure_std_streams(BOOT_LOG.parent)

try:
    from app import config                                # noqa: E402
except BaseException as _e:                               # noqa: BLE001
    _die_visibly("导入依赖失败 / failed to import dependencies", _e)

LOG_FILE = ROOT / LOG_RELATIVE


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="run.py", description="Real-time English->Chinese interpretation")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--cli", action="store_true", help="terminal frontend")
    mode.add_argument("--gui", action="store_true", help="overlay frontend")
    p.add_argument("--config", default=None, help="path to config.yaml")
    p.add_argument("--source", choices=("loopback", "mic"), default=None)
    p.add_argument("--device", default=None, help="device name substring")
    p.add_argument("--model", default=None, help="override mt.model path")
    p.add_argument("--devices", action="store_true",
                   help="list capture devices and exit")
    p.add_argument("--no-metrics", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    setup_console(log_file=LOG_FILE)
    install_crash_log(LOG_FILE)

    if args.devices:
        from app.audio.capture import list_devices
        for d in sorted(list_devices(), key=lambda x: (not x.is_loopback, x.index)):
            print(" ", d)
        return 0

    try:
        cfg = config.load(args.config, root=ROOT)
    except (ValueError, FileNotFoundError) as e:
        logging.getLogger("config").critical("%s", e)
        if not has_console():
            import ctypes
            ctypes.windll.user32.MessageBoxW(
                None, f"配置错误：\n{e}", "同声传译", 0x10)
        else:
            print(f"config error: {e}", file=sys.stderr)
        return 2

    if args.source:
        cfg.audio.source = args.source
    if args.device:
        cfg.audio.device = args.device
    if args.model:
        cfg.mt.model = args.model
    if args.no_metrics:
        cfg.runtime.metrics = False

    setup_console(cfg.runtime.log_level, LOG_FILE)

    # Bound the on-disk footprint before anything writes to it.
    from app import storage
    storage.prepare(cfg.cache_path, cfg.runtime.cache_budget_gb)

    # PySide6 and the OpenVINO runtime are imported lazily here, and they
    # pull in native DLLs -- the most likely thing to fail under a different
    # PATH than the one this was developed in.
    try:
        if args.gui:
            from app.ui.overlay import run_gui
            return run_gui(cfg)
        return run_cli(cfg)
    except BaseException as e:                            # noqa: BLE001
        logging.getLogger("startup").critical("frontend failed", exc_info=True)
        _die_visibly("启动界面失败 / frontend failed to start", e)
        return 1


def run_cli(cfg) -> int:
    from app.cli import CliRenderer
    from app.pipeline import Pipeline

    pipe = Pipeline(cfg)
    stop = threading.Event()

    def on_sigint(_sig, _frm):
        stop.set()

    signal.signal(signal.SIGINT, on_sigint)

    renderer = CliRenderer(pipe.bus, pipe.metrics, cfg.ui.show_source,
                           cfg.runtime.metrics)

    print("loading models (first run compiles for the GPU; "
          "later runs use .ovcache) …")
    try:
        pipe.start()
    except Exception as e:                                # noqa: BLE001
        print(f"\nstartup failed: {e}", file=sys.stderr)
        return 1

    print("Ctrl+C to stop.\n")
    try:
        renderer.run(stop.is_set)
    finally:
        pipe.stop()
        print("\n" + "-" * 64)
        print(pipe.metrics.report())
        print("-" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
