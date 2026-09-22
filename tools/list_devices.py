"""List capture devices, so config audio.device can be set to a name substring."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.audio.capture import list_devices          # noqa: E402
from app.console import setup_console                # noqa: E402

setup_console()

devs = list_devices("pyaudiowpatch")
print("=== pyaudiowpatch ===")
for d in sorted(devs, key=lambda x: (not x.is_loopback, x.index)):
    print(" ", d)
print(f"  ({sum(d.is_loopback for d in devs)} loopback, "
      f"{sum(not d.is_loopback for d in devs)} input)")
