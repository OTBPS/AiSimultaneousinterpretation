"""Step 3 (DESKTOP): merge the LoRA, export to OpenVINO IR, quantise to INT4.

RUNS ON THE DESKTOP, NOT THE LAPTOP -- it needs PyTorch to merge the adapter.
The laptop only ever receives the finished IR folder.

Quantisation is the step that can quietly eat the fine-tune. INT4 with a
group size of 128 and a ratio of 1.0 matches how the stock
OpenVINO/Qwen3-4B-int4-ov was built, so the comparison against the baseline
stays apples-to-apples. If the fine-tuned model measures worse than the
baseline afterwards, re-export with --int8 before concluding the training
failed -- you may be measuring quantisation damage, not training damage.

    python 3_export_openvino.py --base Qwen/Qwen3-4B --adapter lora_adapter \
                                --out Qwen3-4B-ft-int4-ov
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-4B")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", default="Qwen3-4B-ft-int4-ov")
    ap.add_argument("--merged", default=None,
                    help="where to keep the merged fp16 model "
                         "(default: <out>_merged, deleted afterwards)")
    ap.add_argument("--int8", action="store_true",
                    help="export INT8 instead of INT4 (bigger and slower on "
                         "the laptop, but loses less of the fine-tune)")
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--keep-merged", action="store_true")
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    merged_dir = Path(args.merged or (args.out + "_merged"))
    out_dir = Path(args.out)

    print(f"[1/3] merging {args.adapter} into {args.base} …")
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.float16, device_map="cpu",
        trust_remote_code=True)
    model = PeftModel.from_pretrained(base, args.adapter)
    model = model.merge_and_unload()
    if merged_dir.exists():
        shutil.rmtree(merged_dir)           # overwrite, never accumulate
    model.save_pretrained(merged_dir, safe_serialization=True)
    tok.save_pretrained(merged_dir)
    del model, base
    print(f"      merged -> {merged_dir}")

    print("[2/3] exporting to OpenVINO IR and quantising …")
    from optimum.intel import OVModelForCausalLM, OVWeightQuantizationConfig

    if args.int8:
        qcfg = OVWeightQuantizationConfig(bits=8, sym=False)
    else:
        # Mirrors how OpenVINO's own *-int4-ov models are produced, so the
        # fine-tuned model is comparable with the stock baseline.
        qcfg = OVWeightQuantizationConfig(
            bits=4, group_size=args.group_size, ratio=1.0, sym=True)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    ov_model = OVModelForCausalLM.from_pretrained(
        merged_dir, export=True, quantization_config=qcfg,
        trust_remote_code=True)
    ov_model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)

    print("[3/3] exporting the tokenizer for openvino_genai …")
    try:
        from openvino_tokenizers import convert_tokenizer
        import openvino as ov

        ov_tok, ov_detok = convert_tokenizer(tok, with_detokenizer=True)
        ov.save_model(ov_tok, out_dir / "openvino_tokenizer.xml")
        ov.save_model(ov_detok, out_dir / "openvino_detokenizer.xml")
    except Exception as e:                                 # noqa: BLE001
        print(f"      WARNING: tokenizer export failed: {e}")
        print("      openvino_genai.LLMPipeline needs openvino_tokenizer.xml "
              "and openvino_detokenizer.xml in the folder.")

    if not args.keep_merged and merged_dir.exists():
        shutil.rmtree(merged_dir)

    size = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file())
    print(f"\ndone: {out_dir}  ({size/1e9:.2f} GB)")
    print("\nCopy that folder to the laptop's D:\\AI\\Models\\, then:")
    print("  1. point config.yaml's mt.model at it")
    print("  2. python tools/eval_translation.py --models "
          "D:\\AI\\Models\\Qwen3-4B-int4-ov <new folder>")
    print("  3. compare against eval_baseline.json before trusting it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
