"""Measure what this app actually costs in watts.

Battery life is a product requirement for a laptop app, so it gets measured
the same way latency did, rather than guessed at.

Two instruments, because neither alone is sufficient:

* Intel RAPL counters (\\Energy Meter\\...) report SoC power per domain --
  PKG (whole package), PP0 (CPU cores), PP1 (iGPU). These work while plugged
  in and are the only way to see how the cost splits between CPU and iGPU.
  They do NOT include the display, wi-fi, fans or SSD, so they understate
  whole-machine draw -- often by a lot, since the panel alone can be 3-8 W.
* Battery discharge rate (root\\wmi BatteryStatus) is whole-machine ground
  truth, but reads 0 while on AC. Run with --battery after unplugging.

Method is a phased A/B so the app's marginal cost is isolated from whatever
else the machine is doing:

    1. baseline     nothing running
    2. idle         app running, silence on the wire
    3. audio-only   speech playing out the speakers, app NOT running
    4. active       speech playing, app running and translating

Phase 3 exists because loopback capture needs real sound coming out of the
speakers, and a laptop speaker at moderate volume is worth 1-3 W. Comparing
`active` against `baseline` charges that to the app; comparing it against
`audio-only` does not. An earlier run without this phase produced a
whole-machine result that contradicted RAPL, and uncontrolled speaker volume
was the leading suspect.

    python tools/measure_power.py
    python tools/measure_power.py --battery          (unplug first)
    python tools/measure_power.py --seconds 45
"""
from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config                                    # noqa: E402
from app.console import setup_console                     # noqa: E402

RAPL = {
    "pkg": r"\Energy Meter(RAPL_Package0_PKG)\Power",
    "cpu": r"\Energy Meter(RAPL_Package0_PP0)\Power",
    "igpu": r"\Energy Meter(RAPL_Package0_PP1)\Power",
}


@dataclass
class Phase:
    name: str
    label: str
    samples: dict[str, list[float]] = field(default_factory=dict)
    battery_mw: list[float] = field(default_factory=list)
    cap_start: float = 0.0          # RemainingCapacity, mWh
    cap_end: float = 0.0
    wall_s: float = 0.0

    def mean(self, key: str) -> float:
        xs = self.samples.get(key) or []
        return statistics.mean(xs) if xs else 0.0

    def battery(self) -> float:
        """Whole-machine watts from accumulated energy.

        NOT the mean of DischargeRate: this machine's gauge latches a new
        value only about every 30 s, so averaging it over a short phase just
        reports whichever value happened to be latched. Energy drawn from the
        pack over the window is immune to that lag.
        """
        if self.cap_start and self.cap_end and self.wall_s > 0:
            drained_mwh = self.cap_start - self.cap_end
            if drained_mwh > 0:
                return drained_mwh * 3600.0 / self.wall_s
        return 0.0

    def battery_quantisation_w(self, step_mwh: float = 110.0) -> float:
        """Worst-case error from the gauge's coarse steps, in watts."""
        return step_mwh * 3600.0 / self.wall_s if self.wall_s else 0.0


def _ps(script: str) -> str:
    out = subprocess.run(["powershell", "-NoProfile", "-Command", script],
                         capture_output=True, text=True, encoding="utf-8",
                         errors="replace")
    return out.stdout


def sample_loop(phase: Phase, stop: threading.Event, battery: bool) -> None:
    """Sample RAPL (and optionally battery) once a second until stopped.

    One long-lived PowerShell process does the sampling: spawning one per
    second would itself cost more power than the thing being measured.
    """
    paths = " , ".join(f"'{p}'" for p in RAPL.values())
    n = 100000
    script = (
        f"Get-Counter -Counter {paths} -SampleInterval 1 -MaxSamples {n} | "
        "ForEach-Object { "
        "  $v = $_.CounterSamples | ForEach-Object { $_.CookedValue }; "
        "  $bat = 0; "
        "  if ($env:MEASURE_BATTERY -eq '1') { "
        "    $s = Get-CimInstance -Namespace root\\wmi -ClassName BatteryStatus "
        "         -ErrorAction SilentlyContinue; "
        "    if ($s) { $bat = $s.DischargeRate } }; "
        "  Write-Output (($v -join ',') + ',' + $bat) }"
    )
    proc = subprocess.Popen(
        ["powershell", "-NoProfile", "-Command", script],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, encoding="utf-8", errors="replace",
        env={**__import__("os").environ,
             "MEASURE_BATTERY": "1" if battery else "0"},
    )
    keys = list(RAPL)
    try:
        first = True
        while not stop.is_set():
            line = proc.stdout.readline()
            if not line:
                break
            parts = [p.strip() for p in line.strip().split(",") if p.strip() != ""]
            if len(parts) < len(keys):
                continue
            if first:                       # first sample has no interval yet
                first = False
                continue
            try:
                vals = [float(x) for x in parts[:len(keys)]]
            except ValueError:
                continue
            for k, v in zip(keys, vals):
                phase.samples.setdefault(k, []).append(v)
            if battery and len(parts) > len(keys):
                try:
                    phase.battery_mw.append(float(parts[len(keys)]))
                except ValueError:
                    pass
    finally:
        proc.kill()


def remaining_capacity() -> float:
    out = _ps(r"(Get-CimInstance -Namespace root\wmi -ClassName BatteryStatus)"
              ".RemainingCapacity")
    try:
        return float(out.strip().splitlines()[0])
    except Exception:                                      # noqa: BLE001
        return 0.0


def measure(phase: Phase, seconds: float, battery: bool) -> Phase:
    stop = threading.Event()
    t = threading.Thread(target=sample_loop, args=(phase, stop, battery),
                         daemon=True)
    t.start()

    if battery:
        phase.cap_start = remaining_capacity()
    t_start = time.perf_counter()

    tty = sys.stdout.isatty()
    if not tty:                    # a carriage-return countdown spams a log
        print(f"  {phase.label:<34s} …", flush=True)
    for remaining in range(int(seconds), 0, -1):
        if tty:
            print(f"\r  {phase.label:<34s} {remaining:3d}s ", end="", flush=True)
        time.sleep(1)

    stop.set()
    phase.wall_s = time.perf_counter() - t_start
    if battery:
        phase.cap_end = remaining_capacity()
    t.join(timeout=5)

    print(f"{'' if not tty else chr(13)}  {phase.label:<34s} done "
          f"({len(phase.samples.get('pkg', []))} samples"
          + (f", {phase.cap_start - phase.cap_end:.0f} mWh drawn"
             if battery else "") + ")")
    return phase


def kill_app() -> None:
    _ps("Get-Process pythonw -ErrorAction SilentlyContinue | Stop-Process -Force")
    time.sleep(2)


def start_app() -> None:
    subprocess.Popen(
        ["powershell", "-NoProfile", "-Command",
         f"Start-Process -FilePath '{sys.executable.replace('python.exe', 'pythonw.exe')}' "
         f"-ArgumentList '\"{ROOT / 'run.py'}\"','--gui' "
         f"-WorkingDirectory '{ROOT}'"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def play_wav(path: Path, stop: threading.Event) -> threading.Thread:
    def run():
        import numpy as np
        import pyaudiowpatch as pyaudio
        import soundfile as sf
        import soxr

        data, sr = sf.read(path, dtype="float32")
        if data.ndim == 1:
            data = np.stack([data, data], axis=1)
        if sr != 48000:
            data = soxr.resample(data, sr, 48000).astype(np.float32)
        pa = pyaudio.PyAudio()
        try:
            stream = pa.open(format=pyaudio.paFloat32, channels=2,
                             rate=48000, output=True)
            while not stop.is_set():                 # loop the clip
                for i in range(0, len(data), 1024):
                    if stop.is_set():
                        break
                    stream.write(data[i:i + 1024].tobytes())
            stream.stop_stream()
            stream.close()
        finally:
            pa.terminate()

    t = threading.Thread(target=run, daemon=True, name="wav-player")
    t.start()
    return t


def battery_info() -> tuple[float, float]:
    """(full charge capacity mWh, current percent)."""
    out = _ps(
        "$f = Get-CimInstance -Namespace root\\wmi -ClassName BatteryFullChargedCapacity "
        "-ErrorAction SilentlyContinue; "
        "$b = Get-CimInstance Win32_Battery; "
        "Write-Output ($f.FullChargedCapacity, $b.EstimatedChargeRemaining -join ',')")
    try:
        cap, pct = out.strip().splitlines()[0].split(",")
        return float(cap), float(pct)
    except Exception:                                      # noqa: BLE001
        return 0.0, 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=40.0,
                    help="sampling window per phase")
    ap.add_argument("--battery", action="store_true",
                    help="also record battery discharge (unplug the charger)")
    ap.add_argument("--wav", default="bench/speech.wav")
    ap.add_argument("--skip-idle", action="store_true",
                    help="skip the silent-idle phase (RAPL shows it costs ~0)")
    ap.add_argument("--skip-audio-ref", action="store_true",
                    help="skip the audio-only reference phase; the app's cost "
                         "then includes the speakers, which is usually wrong")
    args = ap.parse_args()

    setup_console()
    cfg = config.load(root=ROOT)
    cap_mwh, pct = battery_info()

    ac = "PowerOnline" in _ps(
        "$s = Get-CimInstance -Namespace root\\wmi -ClassName BatteryStatus; "
        "if ($s.PowerOnline) { 'PowerOnline' } else { 'OnBattery' }")
    print(f"power source : {'AC (plugged in)' if ac else 'battery'}")
    print(f"battery      : {pct:.0f}%  full-charge capacity {cap_mwh/1000:.1f} Wh")
    if args.battery and ac:
        print("\n  --battery was requested but the charger is still connected;\n"
              "  discharge rate will read 0. Unplug and re-run.\n")
    print(f"window       : {args.seconds:.0f}s per phase\n")

    print("measuring:")
    kill_app()
    baseline = measure(Phase("baseline", "1/3 baseline (app off)"),
                       args.seconds, args.battery)

    start_app()
    print("  waiting for models to load …", end="", flush=True)
    time.sleep(16)
    print(" ok")
    idle = Phase("idle", "app idle (skipped)")
    if not args.skip_idle:
        idle = measure(Phase("idle", "2/3 app idle (silence)"),
                       args.seconds, args.battery)

    # Audio-only reference: the speakers cost watts, and they are not the
    # app's fault. Measured with the app killed so only playback is running.
    audio_ref = Phase("audio-only", "audio only (app off)")
    if not args.skip_audio_ref:
        kill_app()
        stop_ref = threading.Event()
        play_wav(ROOT / args.wav, stop_ref)
        time.sleep(3)
        audio_ref = measure(Phase("audio-only", "audio only (app off)"),
                            args.seconds, args.battery)
        stop_ref.set()
        time.sleep(1)
        start_app()
        print("  reloading models …", end="", flush=True)
        time.sleep(16)
        print(" ok")

    stop_audio = threading.Event()
    play_wav(ROOT / args.wav, stop_audio)
    time.sleep(3)
    active = measure(Phase("active", "app translating"),
                     args.seconds, args.battery)
    stop_audio.set()
    time.sleep(1)
    kill_app()

    # ---------------------------------------------------------- report
    print("\n" + "=" * 68)
    print("SoC power (Intel RAPL -- excludes display, wi-fi, fans)")
    print("=" * 68)
    print(f"{'':<26s}{'PKG':>10s}{'CPU':>10s}{'iGPU':>10s}")
    for ph in (p for p in (baseline, idle, audio_ref, active) if p.samples):
        print(f"{ph.name:<26s}"
              f"{ph.mean('pkg')/1000:>9.2f}W"
              f"{ph.mean('cpu')/1000:>9.2f}W"
              f"{ph.mean('igpu')/1000:>9.2f}W")
    print("-" * 68)
    d_idle = ((idle.mean("pkg") - baseline.mean("pkg")) / 1000
              if idle.samples else float("nan"))
    ref = audio_ref if audio_ref.samples else baseline
    d_act = (active.mean("pkg") - ref.mean("pkg")) / 1000
    if idle.samples:
        print(f"{'app cost, idle':<26s}{d_idle:>9.2f}W")
    print(f"{'app cost, translating':<26s}{d_act:>9.2f}W"
          f"   (vs {ref.name})")

    if args.battery and any(p.battery() for p in (baseline, idle, active)):
        print("\n" + "=" * 68)
        print("Whole-machine draw (battery discharge)")
        print("=" * 68)
        print("(from energy drawn out of the pack, not the lagging "
              "DischargeRate reading)")
        for ph in (p for p in (baseline, idle, audio_ref, active)
                   if p.battery()):
            w = ph.battery() / 1000
            hours = (cap_mwh / ph.battery()) if ph.battery() else 0
            err = ph.battery_quantisation_w() / 1000
            print(f"{ph.name:<26s}{w:>9.2f}W ±{err:.2f}   "
                  f"~{hours:>5.1f} h of battery")
        bref = audio_ref if audio_ref.battery() else baseline
        extra = (active.battery() - bref.battery()) / 1000
        worst = (active.battery_quantisation_w()
                 + bref.battery_quantisation_w()) / 1000
        print("-" * 68)
        print(f"{'app cost, translating':<26s}{extra:>9.2f}W ±{worst:.2f}"
              f"   (vs {bref.name})")
        if worst > abs(extra) * 0.5:
            print("\n  WARNING: the gauge's step size is large relative to the")
            print("  difference being measured. Re-run with a longer --seconds")
            print("  (300+) before trusting this number.")
    elif cap_mwh:
        est = d_act
        print(f"\nRAPL-only estimate: translating adds ~{est:.1f} W to the SoC.")
        print("Run with --battery (unplugged) for the whole-machine number,")
        print("which is what actually decides battery life.")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
