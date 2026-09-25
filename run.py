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

from app import config                                    # noqa: E402
from app.console import (LOG_RELATIVE, has_console,            # noqa: E402
                         install_crash_log, setup_console)

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

    if args.gui:
        from app.ui.overlay import run_gui
        return run_gui(cfg)
    return run_cli(cfg)


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
