"""Step 2 (DESKTOP / 4090): QLoRA fine-tune Qwen3-4B on the teacher's output.

RUNS ON THE DESKTOP, NOT THE LAPTOP. Needs CUDA PyTorch. Not covered by this
repo's tests -- verify the loss curve and the held-out samples before
shipping the adapter.

Loss is computed on the Chinese completion only, not on the prompt. Training
on the prompt tokens teaches the model to generate the instructions, which
wastes capacity and blurs the thing we actually want.

The prompt format is byte-identical to what app/mt.py sends at inference
time, including the trailing ` /no_think`. A mismatch here is the classic
way a fine-tune looks fine in training and does nothing in production.

    python 2_train_lora.py --data train.jsonl --base Qwen/Qwen3-4B \
                           --out lora_adapter --epochs 2
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

try:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from app.mt import NO_THINK, SYSTEM_PROMPT
except Exception:                                          # noqa: BLE001
    from importlib import import_module
    SYSTEM_PROMPT = import_module("1_generate").SYSTEM_PROMPT   # type: ignore
    NO_THINK = " /no_think"


def build_examples(path: Path, tok, max_len: int):
    """Tokenise into prompt+completion with the prompt masked out of the loss."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        prompt = tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": rec["en"] + NO_THINK}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)

        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        c_ids = tok(rec["zh"] + tok.eos_token, add_special_tokens=False)["input_ids"]
        ids = p_ids + c_ids
        if len(ids) > max_len:
            continue
        labels = [-100] * len(p_ids) + c_ids        # loss on the answer only
        rows.append({"input_ids": ids, "labels": labels,
                     "attention_mask": [1] * len(ids)})
    return rows


class Collator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch):
        import torch

        n = max(len(b["input_ids"]) for b in batch)
        out = {"input_ids": [], "labels": [], "attention_mask": []}
        for b in batch:
            pad = n - len(b["input_ids"])
            out["input_ids"].append(b["input_ids"] + [self.pad_id] * pad)
            out["labels"].append(b["labels"] + [-100] * pad)
            out["attention_mask"].append(b["attention_mask"] + [0] * pad)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in out.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--base", default="Qwen/Qwen3-4B")
    ap.add_argument("--out", default="lora_adapter")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--bf16-base", action="store_true",
                    help="load the base in bf16 instead of 4-bit "
                         "(~16 GB instead of ~10 GB, slightly better quality)")
    args = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig, Trainer, TrainingArguments)

    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    rows = build_examples(Path(args.data), tok, args.max_len)
    random.Random(0).shuffle(rows)
    n_val = max(1, int(len(rows) * args.val_frac))
    val, train = rows[:n_val], rows[n_val:]
    print(f"examples: {len(train)} train / {len(val)} val")
    if not train:
        print("no usable examples", file=sys.stderr)
        return 1

    if args.bf16_base:
        model = AutoModelForCausalLM.from_pretrained(
            args.base, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True)
    else:
        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.base, quantization_config=quant, device_map="auto",
            trust_remote_code=True)
        model = prepare_model_for_kbit_training(model)

    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    lora = LoraConfig(
        r=args.rank, lora_alpha=args.alpha, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir=args.out + "_run",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=20,
        eval_strategy="steps" if val else "no",
        eval_steps=200,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=1,          # overwrite in place, do not pile up ckpts
        bf16=True,
        optim="paged_adamw_8bit" if not args.bf16_base else "adamw_torch",
        gradient_checkpointing=True,
        report_to=[],
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train,
                      eval_dataset=val or None,
                      data_collator=Collator(tok.pad_token_id))
    trainer.train()

    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"\nadapter written to {args.out}")
    print("Next: python 3_export_openvino.py --base "
          f"{args.base} --adapter {args.out} --out Qwen3-4B-ft-int4-ov")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
