"""LoRA SFT of a small student on validator-filtered teacher data.

Runs in the TRAINING venv (.lab/venv-train: torch cu132 + transformers/peft/
trl), not the harness venv:

    .lab/venv-train/Scripts/python research/train_sft.py --base ibm-granite/granite-4.2-3b \
        --data .lab/datasets/sft-v1 --out .lab/train/granite3b-sft-v1 --qlora

Recipe (docs/research/techniques-2026-11.md §2): LoRA on all linear layers,
rsLoRA, rank 16, loss on the completion only, cosine schedule, 2 epochs.
QLoRA (4-bit NF4 base) when the bf16 base would not fit next to the desktop's
~2.5 GB of VRAM. After training the adapter is merged into a bf16 copy of
the base and saved for GGUF conversion (research/export_gguf.py), so the
tuned and untuned models are compared through the same converter and quant.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--max-len", type=int, default=3072)
    ap.add_argument("--bsz", type=int, default=1)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--qlora", action="store_true")
    ap.add_argument("--max-steps", type=int, default=-1, help="smoke test: stop after N steps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-merge", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(a.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.bfloat16,
                              bnb_4bit_use_double_quant=True) if a.qlora else None
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16, device_map={"": 0},
                                                 attn_implementation="sdpa", quantization_config=qcfg)
    ds = load_dataset("json", data_files={"train": str(Path(a.data) / "train.jsonl"),
                                          "val": str(Path(a.data) / "val.jsonl")})
    keep = ["prompt", "completion"]
    ds = ds.remove_columns([c for c in ds["train"].column_names if c not in keep])
    # A right-truncated row loses its completion and trains on nothing: measure
    # and drop over-length rows instead of letting the collator cut them.
    def _len(row):
        ids = tok.apply_chat_template(row["prompt"] + row["completion"], tokenize=True)
        ids = ids["input_ids"] if isinstance(ids, dict) else ids
        return {"n_tok": len(ids)}
    ds = ds.map(_len)
    lens = sorted(ds["train"]["n_tok"])
    length_report = {"train_tokens_p50": lens[len(lens) // 2] if lens else 0,
                     "train_tokens_p95": lens[int(len(lens) * 0.95)] if lens else 0,
                     "train_tokens_max": lens[-1] if lens else 0,
                     "dropped_over_max_len": {s: sum(1 for n in ds[s]["n_tok"] if n > a.max_len) for s in ds}}
    print("lengths:", length_report, flush=True)
    ds = ds.filter(lambda r: r["n_tok"] <= a.max_len).remove_columns(["n_tok"])

    peft_cfg = LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=0.0, target_modules="all-linear",
                          use_rslora=True, task_type="CAUSAL_LM")
    cfg = SFTConfig(output_dir=str(out / "ckpt"), num_train_epochs=a.epochs, max_steps=a.max_steps,
                    per_device_train_batch_size=a.bsz, gradient_accumulation_steps=a.accum,
                    learning_rate=a.lr, lr_scheduler_type="cosine", warmup_ratio=0.03,
                    logging_steps=5, eval_strategy="epoch" if a.max_steps < 0 else "no",
                    save_strategy="no", bf16=True, gradient_checkpointing=True,
                    max_length=a.max_len, completion_only_loss=True, report_to="none", seed=a.seed,
                    dataloader_num_workers=0)
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds["train"],
                         eval_dataset=ds["val"] if len(ds["val"]) else None,
                         processing_class=tok, peft_config=peft_cfg)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    res = trainer.train()
    train_s = time.time() - t0
    ev = trainer.evaluate() if len(ds["val"]) and a.max_steps < 0 else {}
    trainer.model.save_pretrained(str(out / "adapter"))
    metrics = {"base": a.base, "data": a.data, "train_rows": len(ds["train"]), "val_rows": len(ds["val"]),
               "rank": a.rank, "alpha": a.alpha, "lr": a.lr, "epochs": a.epochs, "qlora": a.qlora,
               "max_len": a.max_len, **length_report, "train_seconds": train_s, "train_loss": res.training_loss,
               "eval": ev, "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
               "log_history": trainer.state.log_history}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in metrics.items() if k != "log_history"}, indent=2, default=str))

    if a.no_merge or a.max_steps > 0:
        return
    # Merge into a bf16 copy of the base (never into the 4-bit one).
    del trainer, model
    torch.cuda.empty_cache()
    base = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16, device_map={"": "cpu"})
    merged = PeftModel.from_pretrained(base, str(out / "adapter")).merge_and_unload()
    merged.save_pretrained(str(out / "merged"), safe_serialization=True)
    tok.save_pretrained(str(out / "merged"))
    print("merged ->", out / "merged")


if __name__ == "__main__":
    main()
