"""Step 1 (DESKTOP / 4090): generate EN->ZH training pairs with the 14B teacher.

RUNS ON THE DESKTOP, NOT THE LAPTOP. Needs CUDA PyTorch; it is not exercised
by this repo's test suite and has not been run on the laptop -- treat the
first run as a shakedown and check the output pairs by hand before training
on them.

Input is MONOLINGUAL English: one sentence or short paragraph per line. The
Chinese side is produced here, which is why no human parallel corpus is
needed.

The prompt deliberately matches app/mt.py's SYSTEM_PROMPT. The student is
being taught to respond to the exact prompt it will see in production; if the
two drift apart the fine-tune teaches the wrong conditional.

    python 1_generate.py --corpus corpus.txt --out train.jsonl \
                         --teacher Qwen/Qwen3-14B --max-samples 20000
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

# Keep in sync with app/mt.py. Imported literally rather than duplicated when
# the repo is available, so the two cannot silently diverge.
try:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from app.mt import NO_THINK, SYSTEM_PROMPT
except Exception:                                          # noqa: BLE001
    SYSTEM_PROMPT = (
        "You are a professional simultaneous interpreter. Translate the user's "
        "English into natural, fluent Chinese.\n"
        "Rules:\n"
        "- Output ONLY the Chinese translation. No explanation, no pinyin, no "
        "quotes, no English.\n"
        "- Translate meaning, not words. Render idioms as idiomatic Chinese.\n"
        "- Keep technical terms accurate and consistent.\n"
        "- If a [Terms] line is given, you MUST use those translations.\n"
    )
    NO_THINK = " /no_think"

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_HAN = re.compile(r"[一-鿿]")
_LATIN_RUN = re.compile(r"[A-Za-z]{5,}")


def load_corpus(path: Path, min_chars: int, max_chars: int) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = " ".join(raw.split())
        if not (min_chars <= len(s) <= max_chars):
            continue
        if not _LATIN_RUN.search(s):          # skip lines with no real words
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def clean(zh: str) -> str:
    zh = _THINK.sub("", zh).strip()
    for p in ("翻译：", "翻译:", "译文：", "译文:", "Translation:"):
        if zh.startswith(p):
            zh = zh[len(p):].lstrip()
    return zh.strip().strip('"“”')


def acceptable(en: str, zh: str) -> tuple[bool, str]:
    """Reject teacher outputs that would poison the student."""
    if not zh:
        return False, "empty"
    if not _HAN.search(zh):
        return False, "no chinese"
    if "<think>" in zh:
        return False, "think leaked"
    han = len(_HAN.findall(zh))
    if han < 2:
        return False, "almost no chinese"
    # A sane EN->ZH character ratio is roughly 0.3-1.0; far outside that means
    # the teacher rambled, refused, or echoed the source.
    ratio = len(zh) / max(1, len(en))
    if not 0.15 <= ratio <= 1.6:
        return False, f"length ratio {ratio:.2f}"
    if en.lower()[:40] in zh.lower():
        return False, "echoed source"
    return True, ""


def gen_vllm(prompts: list[str], model: str, max_new: int, tp: int):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    texts = [
        tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": p + NO_THINK}],
            tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
        for p in prompts
    ]
    llm = LLM(model=model, tensor_parallel_size=tp, max_model_len=2048,
              gpu_memory_utilization=0.90, trust_remote_code=True)
    params = SamplingParams(temperature=0.0, max_tokens=max_new)
    for out in llm.generate(texts, params):
        yield out.outputs[0].text


def gen_hf(prompts: list[str], model: str, max_new: int, batch: int):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True,
                                        padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(
        model, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True)
    mdl.eval()

    for i in range(0, len(prompts), batch):
        chunk = prompts[i:i + batch]
        texts = [
            tok.apply_chat_template(
                [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": p + NO_THINK}],
                tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
            for p in chunk
        ]
        enc = tok(texts, return_tensors="pt", padding=True,
                  truncation=True, max_length=1024).to(mdl.device)
        with torch.inference_mode():
            out = mdl.generate(**enc, max_new_tokens=max_new,
                               do_sample=False,
                               pad_token_id=tok.pad_token_id)
        for j in range(len(chunk)):
            gen = out[j][enc["input_ids"].shape[1]:]
            yield tok.decode(gen, skip_special_tokens=True)
        print(f"  {min(i + batch, len(prompts))}/{len(prompts)}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="monolingual English, one per line")
    ap.add_argument("--out", default="train.jsonl")
    ap.add_argument("--teacher", default="Qwen/Qwen3-14B")
    ap.add_argument("--max-samples", type=int, default=20000)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--min-chars", type=int, default=25)
    ap.add_argument("--max-chars", type=int, default=400)
    ap.add_argument("--batch", type=int, default=16, help="HF fallback batch size")
    ap.add_argument("--tp", type=int, default=1, help="vLLM tensor parallel")
    ap.add_argument("--no-vllm", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    lines = load_corpus(Path(args.corpus), args.min_chars, args.max_chars)
    random.Random(args.seed).shuffle(lines)
    lines = lines[:args.max_samples]
    print(f"corpus: {len(lines)} unique lines after filtering")
    if not lines:
        print("nothing to do", file=sys.stderr)
        return 1

    use_vllm = not args.no_vllm
    if use_vllm:
        try:
            import vllm                                    # noqa: F401
        except ImportError:
            print("vllm not installed, falling back to transformers "
                  "(much slower)")
            use_vllm = False

    stream = (gen_vllm(lines, args.teacher, args.max_new, args.tp) if use_vllm
              else gen_hf(lines, args.teacher, args.max_new, args.batch))

    kept = 0
    rejected: dict[str, int] = {}
    with open(args.out, "w", encoding="utf-8") as f:
        for en, raw in zip(lines, stream):
            zh = clean(raw)
            ok, why = acceptable(en, zh)
            if not ok:
                rejected[why] = rejected.get(why, 0) + 1
                continue
            f.write(json.dumps({"en": en, "zh": zh}, ensure_ascii=False) + "\n")
            kept += 1

    print(f"\nkept {kept} / {len(lines)} pairs -> {args.out}")
    if rejected:
        print("rejected:")
        for why, n in sorted(rejected.items(), key=lambda kv: -kv[1]):
            print(f"  {n:6d}  {why}")
    print("\nSpot-check the first ~50 lines by hand before training on this.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
