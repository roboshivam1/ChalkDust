"""GRPO on the beat-fill job, rewarded by the ChalkDust validator.

Runs in .lab/venv-train with plain transformers generation (no vLLM, so no
WSL). Needs the reward server running in the harness venv:

    .lab/venv-harness/Scripts/python -m research.reward_server --port 8099
    .lab/venv-train/Scripts/python research/train_grpo.py --base .lab/train/<sft>/merged \
        --data .lab/datasets/sft-v1 --out .lab/train/<sft>-grpo

Prompts are the fill prompts from the SFT set (rows of type beat, round 0),
so the policy is trained on exactly what it sees at inference. They come from
the validation families first (`val.jsonl`: families SFT never trained on),
then train rows. On rows SFT just fitted, every sample in a group scored the
same, so the advantage and the gradient were zero (smoke step 2:
frac_reward_zero_std 1.0; 2026-11-21 training review). Reward is
research/reward_server.reward_of: validator gate x content floor. beta=0
(no reference model; TRL's default), loss_type left at TRL's default.
Unconstrained generation: the grammar is an inference-time aid, and RL
should teach the model to need it less.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer

REWARD_URL = "http://127.0.0.1:8099/score"
LOG: list[dict] = []


def validator_reward(completions, component, **kwargs):
    items = [{"component": comp, "completion": c[0]["content"] if isinstance(c, list) else c}
             for c, comp in zip(completions, component)]
    r = requests.post(REWARD_URL, json={"items": items}, timeout=600).json()
    LOG.append({"t": time.time(), "rewards": r["rewards"],
                "levels": [d.get("level") for d in r["details"]],
                "clean": [d.get("clean") for d in r["details"]]})
    return r["rewards"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--gens", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max-completion", type=int, default=600)
    ap.add_argument("--max-prompts", type=int, default=400)
    ap.add_argument("--qlora", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompt-splits", default="val,train",
                    help="SFT-set splits to draw fill prompts from, in priority order")
    ap.add_argument("--max-zero-std", type=float, default=0.8,
                    help="exit 3 if the mean share of groups with equal rewards exceeds this: no learning signal")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(a.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    from datasets import concatenate_datasets
    parts, split_counts = [], {}
    for split in [s.strip() for s in a.prompt_splits.split(",") if s.strip()]:
        f = Path(a.data) / f"{split}.jsonl"
        if not f.exists() or not f.stat().st_size:
            split_counts[split] = 0
            continue
        d = load_dataset("json", data_files=str(f))["train"].filter(lambda r: r["type"] == "beat")
        d = d.remove_columns([c for c in d.column_names if c not in ("prompt", "component")])
        parts.append(d.shuffle(seed=a.seed))      # shuffle within a split; splits stay in priority order
        split_counts[split] = len(d)
    ds = concatenate_datasets(parts)
    seen, keep = set(), []
    for i, r in enumerate(ds):      # one row per distinct prompt
        k = r["prompt"][-1]["content"]
        if k not in seen:
            seen.add(k)
            keep.append(i)
    ds = ds.select(keep[: a.max_prompts])
    print("prompts:", len(ds), "from", split_counts, flush=True)

    qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.bfloat16) if a.qlora else None
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16, device_map={"": 0},
                                                 attn_implementation="sdpa", quantization_config=qcfg)
    cfg = GRPOConfig(output_dir=str(out / "ckpt"), max_steps=a.steps, learning_rate=a.lr,
                     per_device_train_batch_size=a.gens, gradient_accumulation_steps=1,
                     num_generations=a.gens, max_completion_length=a.max_completion,
                     temperature=0.8, beta=0.0, bf16=True, gradient_checkpointing=True,
                     logging_steps=1, save_strategy="no", report_to="none", seed=a.seed,
                     log_completions=False, chat_template_kwargs={"enable_thinking": False})
    peft_cfg = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, target_modules="all-linear", use_rslora=True,
                          task_type="CAUSAL_LM")
    trainer = GRPOTrainer(model=model, reward_funcs=validator_reward, args=cfg, train_dataset=ds,
                          processing_class=tok, peft_config=peft_cfg)
    # TRL casts the LoRA weights to bf16 under a quantized base; at lr ~1e-5 most
    # bf16 LoRA-A updates round to nothing (training review: 81% never change).
    # Keep the trainable weights in fp32; the optimizer is only built in train().
    n_cast = 0
    for _, p in trainer.model.named_parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
            n_cast += 1
    print("trainable tensors cast to fp32:", n_cast, flush=True)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer.train()
    secs = time.time() - t0
    trainer.model.save_pretrained(str(out / "adapter"))
    zs = [h["frac_reward_zero_std"] for h in trainer.state.log_history if "frac_reward_zero_std" in h]
    zero_std = sum(zs) / len(zs) if zs else None
    metrics = {"base": a.base, "steps": a.steps, "gens": a.gens, "lr": a.lr, "prompts": len(ds),
               "prompt_splits": split_counts, "fp32_cast_tensors": n_cast, "mean_frac_reward_zero_std": zero_std,
               "train_seconds": secs, "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
               "log_history": trainer.state.log_history, "reward_log": LOG}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1, default=str), encoding="utf-8")
    first = [sum(x["rewards"]) / len(x["rewards"]) for x in LOG[:10]]
    last = [sum(x["rewards"]) / len(x["rewards"]) for x in LOG[-10:]]
    print(json.dumps({"train_seconds": secs, "peak_vram_gib": metrics["peak_vram_gib"],
                      "mean_frac_reward_zero_std": zero_std,
                      "reward_first10": sum(first) / max(len(first), 1),
                      "reward_last10": sum(last) / max(len(last), 1)}, indent=2))
    if zero_std is not None and zero_std > a.max_zero_std:
        # Saved above for the record; a run with no reward spread is not evaluated as a result.
        raise SystemExit(3)


if __name__ == "__main__":
    main()
