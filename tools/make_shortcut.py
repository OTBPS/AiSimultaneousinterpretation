"""Create (or refresh) the desktop shortcut.

Scripted rather than clicked so a reinstall or a new machine is one command,
and so the shortcut is always rebuilt in place -- it overwrites the existing
.lnk instead of leaving "同声传译 (2).lnk" behind.

    python tools/make_shortcut.py
    python tools/make_shortcut.py --remove
    python tools/make_shortcut.py --name "EN-ZH Subtitles" --console
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_NAME = "同声传译"


def desktop_dir() -> Path:
    """Resolve the real Desktop, which OneDrive often redirects."""
    try:
        import ctypes
        import ctypes.wintypes

        buf = ctypes.create_unicode_buffer(ctypes.wintypes.MAX_PATH)
        # CSIDL_DESKTOPDIRECTORY = 0x0010, SHGFP_TYPE_CURRENT = 0
        if ctypes.windll.shell32.SHGetFolderPathW(None, 0x0010, None, 0, buf) == 0:
            return Path(buf.value)
    except Exception:                                      # noqa: BLE001
        pass
    return Path(os.path.expanduser("~")) / "Desktop"


def interpreter(console: bool) -> Path:
    """pythonw.exe for a GUI launch (no console window), python.exe otherwise."""
    exe = Path(sys.executable)
    if console:
        return exe
    pythonw = exe.with_name("pythonw.exe")
    return pythonw if pythonw.exists() else exe


def build(name: str, console: bool, cli: bool) -> Path:
    try:
        from win32com.client import Dispatch            # type: ignore
        shell = Dispatch("WScript.Shell")
    except ImportError:
        import subprocess

        return _build_via_powershell(name, console, cli)

    path = desktop_dir() / f"{name}.lnk"
    lnk = shell.CreateShortCut(str(path))
    lnk.TargetPath = str(interpreter(console))
    lnk.Arguments = f'"{ROOT / "run.py"}" {"--cli" if cli else "--gui"}'
    lnk.WorkingDirectory = str(ROOT)
    lnk.IconLocation = f"{ROOT / 'assets' / 'app.ico'},0"
    lnk.Description = "同声传译 EN→ZH — 本机核显实时翻译"
    lnk.Save()
    return path


def _build_via_powershell(name: str, console: bool, cli: bool) -> Path:
    """pywin32 is not a dependency, so fall back to PowerShell's COM access."""
    import subprocess

    path = desktop_dir() / f"{name}.lnk"
    icon = ROOT / "assets" / "app.ico"
    script = f"""
$ws = New-Object -ComObject WScript.Shell
$lnk = $ws.CreateShortcut({_ps(str(path))})
$lnk.TargetPath = {_ps(str(interpreter(console)))}
$lnk.Arguments = {_ps(f'"{ROOT / "run.py"}" ' + ('--cli' if cli else '--gui'))}
$lnk.WorkingDirectory = {_ps(str(ROOT))}
$lnk.IconLocation = {_ps(f'{icon},0')}
$lnk.Description = {_ps('同声传译 EN→ZH — 本机核显实时翻译')}
$lnk.Save()
"""
    subprocess.run(["powershell", "-NoProfile", "-Command", script],
                   check=True, capture_output=True)
    return path


def _ps(value: str) -> str:
    """Quote a string for PowerShell (single quotes, doubled to escape)."""
    return "'" + value.replace("'", "''") + "'"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=DEFAULT_NAME)
    ap.add_argument("--console", action="store_true",
                    help="launch with a visible console (python.exe)")
    ap.add_argument("--cli", action="store_true",
                    help="shortcut runs the terminal frontend (implies --console)")
    ap.add_argument("--remove", action="store_true")
    args = ap.parse_args()

    path = desktop_dir() / f"{args.name}.lnk"
    if args.remove:
        if path.exists():
            path.unlink()
            print(f"removed {path}")
        else:
            print(f"not present: {path}")
        return 0

    icon = ROOT / "assets" / "app.ico"
    if not icon.exists():
        print("assets/app.ico missing -- run: python tools/make_icon.py",
              file=sys.stderr)
        return 1

    created = build(args.name, args.console or args.cli, args.cli)
    print(f"shortcut: {created}")
    print(f"  target : {interpreter(args.console or args.cli)}")
    print(f"  workdir: {ROOT}")
    print(f"  mode   : {'CLI' if args.cli else 'GUI overlay'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
