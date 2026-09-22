"""Compare translation models on a fixed test set.

Exists because "did the fine-tune help?" is unanswerable without a baseline
captured beforehand. Run this against the current model now, run it again
against the distilled one later, diff the two.

Deliberately NOT a BLEU score. We have no reference translations, and BLEU on
50 sentences would be noise dressed up as a number. Instead:

  * automatic regression checks that catch the failure modes actually observed
    on this project -- English leaking into Chinese output, <think> tags
    escaping the filter, glossary terms ignored, degenerate/empty output;
  * a side-by-side dump for human judgement, which is still the only honest
    way to compare translation quality at this scale.

    python tools/eval_translation.py
    python tools/eval_translation.py --models D:\\AI\\Models\\Qwen3-4B-int4-ov \\
                                              D:\\AI\\Models\\Qwen3-8B-int4-ov
    python tools/eval_translation.py --out eval_before.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config                                    # noqa: E402
from app.console import setup_console                     # noqa: E402
from app.context import Context                           # noqa: E402
from app.glossary import Glossary                         # noqa: E402
from app.mt import MtEngine                               # noqa: E402

# Cases chosen to cover the failure modes this project actually hit, not a
# generic benchmark. `expect` terms are checked automatically.
TEST_SET: list[dict] = [
    {"id": "colloquial", "en": "Yeah, no, I mean — we could ship it Friday, "
                               "but honestly? Gonna be rough."},
    {"id": "terminology", "en": "The scheduler preempts the running thread when "
                                "a higher-priority task becomes runnable, then "
                                "performs a context switch.",
     "expect": ["上下文切换"]},
    {"id": "prefill", "en": "Prefill is compute-bound while decode is limited "
                            "by memory bandwidth.",
     "expect": ["预填充"], "forbid": ["预读取", "预取"]},
    {"id": "long_clause", "en": "Although the initial benchmarks suggested that "
                                "the new allocator would reduce fragmentation, "
                                "subsequent testing under sustained load revealed "
                                "that it actually increased peak memory usage by "
                                "roughly twelve percent."},
    {"id": "pronoun", "en": "Context: We evaluated the Arc 140T integrated GPU "
                            "against a discrete card.\nTranslate this: It turned "
                            "out to be much better at prefill, though it lagged "
                            "on decode."},
    {"id": "idiom_ocean", "en": "Let's not boil the ocean here — just get a rough "
                                "cut in front of the team by Thursday.",
     "forbid": ["煮沸", "烧开", "海洋"]},
    {"id": "idiom_fruit", "en": "That integration is low-hanging fruit, we should "
                                "grab it this sprint.",
     "forbid": ["低垂", "水果"]},
    # Both Qwen3-4B and Qwen2.5-1.5B rendered "quarter over quarter" as 同比
    # (year over year) here. The automated checks missed it; a human caught
    # it. Now it is pinned, and the glossary carries the correct term.
    {"id": "numbers", "en": "Revenue grew 12.5 percent quarter over quarter, from "
                            "$4.2 million to $4.7 million.",
     "expect": ["12.5", "环比"], "forbid": ["同比"]},
    {"id": "fragment", "en": "copies between stages.", "is_continuation": True},
    {"id": "disfluency", "en": "So the— sorry, the thing is, um, we haven't "
                               "actually validated that on production hardware yet."},
    {"id": "acronyms", "en": "The NPU handles INT8 inference while the iGPU runs "
                             "the FP16 encoder.",
     "expect": ["NPU"]},
    {"id": "question", "en": "Could you walk me through why the p95 latency "
                             "regressed after the cache change?"},
]

_LATIN_RUN = re.compile(r"[A-Za-z]{4,}")
# Technical tokens that SHOULD stay in Latin script.
#
# Bounded with explicit lookarounds, not \b: CJK characters are word
# characters under Unicode, so "而iGPU则" has no word boundary before the "i"
# and \biGPU\b silently fails to match -- which showed up as a false
# "english leaked" on a perfectly good translation.
_ALLOWED_LATIN = re.compile(
    r"(?<![A-Za-z0-9])(NPU|GPU|CPU|iGPU|INT8|INT4|FP16|BF16|API|LLM|ASR|RAM|"
    r"VRAM|SSD|Arc|Intel|Qwen|Whisper|OpenVINO|p95|p50)(?![A-Za-z0-9])",
    re.IGNORECASE)


@dataclass
class Result:
    id: str
    en: str
    zh: str
    ms: float
    ttft_ms: float
    issues: list[str]


def check(case: dict, zh: str) -> list[str]:
    issues: list[str] = []
    if not zh.strip():
        issues.append("empty output")
        return issues
    if "<think>" in zh or "</think>" in zh:
        issues.append("think tag leaked")

    residual = _ALLOWED_LATIN.sub("", zh)
    leaks = _LATIN_RUN.findall(residual)
    if leaks:
        issues.append(f"english leaked: {leaks[:3]}")

    for term in case.get("expect", []):
        if term not in zh:
            issues.append(f"missing expected term: {term}")
    for term in case.get("forbid", []):
        if term in zh:
            issues.append(f"contains forbidden literal: {term}")

    # A translation far longer than its source is usually the model rambling
    # or restating, which we saw with continuation fragments.
    if len(zh) > max(60, len(case["en"]) * 1.4):
        issues.append(f"suspiciously long ({len(zh)} chars for "
                      f"{len(case['en'])} source chars)")
    return issues


def run_model(model_path: str, cfg, use_glossary: bool) -> list[Result]:
    glossary = Glossary(cfg.glossary_path, cfg.glossary.enabled and use_glossary)
    engine = MtEngine(model_path, cfg.mt.device, cfg.ov_props(),
                      cfg.mt.max_new_tokens, Context(pairs=0, reset_every=999),
                      glossary)
    engine.warmup()

    out: list[Result] = []
    for case in TEST_SET:
        chunks: list[str] = []
        t0 = time.perf_counter()
        first = {"t": None}

        def sink(text: str):
            if first["t"] is None:
                first["t"] = time.perf_counter()
            chunks.append(text)

        zh, _ = engine.translate(case["en"], sink, use_context=True,
                                 is_continuation=case.get("is_continuation", False))
        total = (time.perf_counter() - t0) * 1000
        ttft = ((first["t"] or time.perf_counter()) - t0) * 1000
        out.append(Result(case["id"], case["en"], zh, total, ttft, check(case, zh)))
    engine.close()
    del engine
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=None,
                    help="model dirs to compare (default: mt.model from config)")
    ap.add_argument("--no-glossary", action="store_true",
                    help="measure the raw model, without glossary injection")
    ap.add_argument("--out", default=None, help="write results as JSON")
    args = ap.parse_args()

    setup_console()
    cfg = config.load(root=ROOT)
    models = args.models or [cfg.mt.model]

    all_results: dict[str, list[Result]] = {}
    for m in models:
        name = Path(m).name
        print(f"\n=== {name} ===", flush=True)
        try:
            res = run_model(m, cfg, not args.no_glossary)
        except Exception as e:                             # noqa: BLE001
            print(f"  FAILED to load/run: {e}")
            continue
        all_results[name] = res
        for r in res:
            flag = "  " if not r.issues else "!!"
            print(f"{flag} [{r.id:12s}] {r.ms:6.0f}ms  {r.zh[:70]}")
            for issue in r.issues:
                print(f"       -> {issue}")
        bad = sum(1 for r in res if r.issues)
        avg = sum(r.ms for r in res) / len(res) if res else 0
        print(f"  --- {len(res) - bad}/{len(res)} clean, avg {avg:.0f} ms/sentence")

    if len(all_results) > 1:
        print("\n" + "=" * 70)
        print("SIDE BY SIDE (human judgement is still the real metric)")
        print("=" * 70)
        for case in TEST_SET:
            print(f"\n[{case['id']}] {case['en'][:90]}")
            for name, res in all_results.items():
                r = next((x for x in res if x.id == case["id"]), None)
                if r:
                    print(f"  {name[:28]:<30s} {r.zh[:80]}")

    print("\n" + "=" * 70)
    print(f"{'model':<34s}{'clean':>8s}{'avg ms':>10s}")
    for name, res in all_results.items():
        bad = sum(1 for r in res if r.issues)
        avg = sum(r.ms for r in res) / len(res) if res else 0
        print(f"{name[:33]:<34s}{len(res)-bad:>4d}/{len(res):<3d}{avg:>10.0f}")
    print("=" * 70)

    if args.out:
        payload = {name: [asdict(r) for r in res]
                   for name, res in all_results.items()}
        Path(args.out).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
