"""Rolling latency stats.

Everything is measured from the speaker's last voiced frame -- not from when
the pipeline decided the utterance was over. The endpoint hangover is the
single largest term in the budget, so letting it sit outside the metric would
make the metric flattering and useless.

The physical floor is roughly:

    endpoint hangover      550 ms   (config: vad.endpoint_silence_ms)
  + Whisper on the segment ~400 ms  (measured p95 on this machine)
  + MT first token         ~200 ms  (measured)
  = ~1150 ms before a single character can appear

Anything above that is queueing: one segment can split into several sentences,
and with a single GPU worker (E1: concurrency is 4.69x worse) sentence 2 waits
for sentence 1. So the p95 gate is set at 2500 ms, not at the floor.
"""
from __future__ import annotations

import statistics
import threading
from collections import deque

# Gates for the Phase-1 acceptance run (tools/replay.py).
TTFT_P95_GATE_MS = 2500.0
LINE_P95_GATE_MS = 3500.0


class Metrics:
    def __init__(self, window: int = 200, endpoint_ms: int = 0) -> None:
        self._ttft: deque[float] = deque(maxlen=window)
        self._total: deque[float] = deque(maxlen=window)
        self._asr: deque[float] = deque(maxlen=window)
        self._lock = threading.Lock()
        self.endpoint_ms = endpoint_ms
        self.dropped = 0
        self.underruns = 0
        self.lines = 0

    def record_line(self, ttft_ms: float, total_ms: float) -> None:
        with self._lock:
            self._ttft.append(ttft_ms)
            self._total.append(total_ms)
            self.lines += 1

    def record_asr(self, ms: float) -> None:
        with self._lock:
            self._asr.append(ms)

    @staticmethod
    def _p(xs: deque[float], q: int) -> float:
        if not xs:
            return 0.0
        if len(xs) == 1:
            return xs[0]
        return statistics.quantiles(sorted(xs), n=100)[q - 1]

    def snapshot(self) -> dict[str, float]:
        with self._lock:
            return {
                "lines": self.lines,
                "ttft_p50": self._p(self._ttft, 50),
                "ttft_p95": self._p(self._ttft, 95),
                "total_p50": self._p(self._total, 50),
                "total_p95": self._p(self._total, 95),
                "asr_p50": self._p(self._asr, 50),
                "asr_p95": self._p(self._asr, 95),
                "dropped": self.dropped,
                "underruns": self.underruns,
            }

    def report(self) -> str:
        s = self.snapshot()
        if not s["lines"]:
            return "no lines translated"

        def gate(value: float, limit: float) -> str:
            return "PASS" if 0 < value < limit else "FAIL"

        queueing = max(0.0, s["ttft_p50"] - self.endpoint_ms - s["asr_p50"])
        lines = [
            f"lines={int(s['lines'])}  dropped={int(s['dropped'])}  "
            f"underruns={int(s['underruns'])}",
            "  measured from the speaker's last voiced frame:",
            f"    first-token    p50 {s['ttft_p50']:7.0f} ms   "
            f"p95 {s['ttft_p95']:7.0f} ms   "
            f"[<{TTFT_P95_GATE_MS:.0f} {gate(s['ttft_p95'], TTFT_P95_GATE_MS)}]",
            f"    line-complete  p50 {s['total_p50']:7.0f} ms   "
            f"p95 {s['total_p95']:7.0f} ms   "
            f"[<{LINE_P95_GATE_MS:.0f} {gate(s['total_p95'], LINE_P95_GATE_MS)}]",
            "  where the p50 first-token time goes:",
            f"    endpoint hangover {self.endpoint_ms:7.0f} ms  (vad.endpoint_silence_ms)",
            f"    ASR               {s['asr_p50']:7.0f} ms",
            f"    queue + MT TTFT   {queueing:7.0f} ms",
            f"  no drops/underruns                             "
            f"[{'PASS' if s['dropped'] == 0 and s['underruns'] == 0 else 'FAIL'}]",
        ]
        return "\n".join(lines)
