"""GRPO on the beat-fill job, rewarded by the ChalkDust validator.

Runs in .lab/venv-train with plain transformers generation (no vLLM, so no
WSL). Needs the reward server running in the harness venv:

    .lab/venv-harness/Scripts/python -m research.reward_server --port 8099
    .lab/venv-train/Scripts/python research/train_grpo.py --base .lab/train/<sft>/merged \
        --data .lab/datasets/sft-v1 --out .lab/train/<sft>-grpo

Prompts are the fill prompts from the SFT set (rows of type beat, round 0),
so the policy is trained on exactly what it sees at inference. Reward is
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
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(a.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    ds = load_dataset("json", data_files=str(Path(a.data) / "train.jsonl"))["train"]
    ds = ds.filter(lambda r: r["type"] == "beat")
    seen, keep = set(), []
    for i, r in enumerate(ds):      # one row per distinct prompt
        k = r["prompt"][-1]["content"]
        if k not in seen:
            seen.add(k)
            keep.append(i)
    ds = ds.select(keep[: a.max_prompts]).shuffle(seed=a.seed)
    ds = ds.remove_columns([c for c in ds.column_names if c not in ("prompt", "component")])

    qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.bfloat16) if a.qlora else None
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16, device_map={"": 0},
                                                 attn_implementation="sdpa", quantization_config=qcfg)
    cfg = GRPOConfig(output_dir=str(out / "ckpt"), max_steps=a.steps, learning_rate=a.lr,
                     per_device_train_batch_size=a.gens, gradient_accumulation_steps=1,
                     num_generations=a.gens, max_completion_length=a.max_completion,
                     temperature=0.8, beta=0.0, bf16=True, gradient_checkpointing=True,
                     logging_steps=1, save_strategy="no", report_to="none", seed=a.seed,
                     log_completions=False)
    peft_cfg = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, target_modules="all-linear", use_rslora=True,
                          task_type="CAUSAL_LM")
    trainer = GRPOTrainer(model=model, reward_funcs=validator_reward, args=cfg, train_dataset=ds,
                          processing_class=tok, peft_config=peft_cfg)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer.train()
    secs = time.time() - t0
    trainer.model.save_pretrained(str(out / "adapter"))
    metrics = {"base": a.base, "steps": a.steps, "gens": a.gens, "lr": a.lr, "prompts": len(ds),
               "train_seconds": secs, "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
               "log_history": trainer.state.log_history, "reward_log": LOG}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1, default=str), encoding="utf-8")
    first = [sum(x["rewards"]) / len(x["rewards"]) for x in LOG[:10]]
    last = [sum(x["rewards"]) / len(x["rewards"]) for x in LOG[-10:]]
    print(json.dumps({"train_seconds": secs, "peak_vram_gib": metrics["peak_vram_gib"],
                      "reward_first10": sum(first) / max(len(first), 1),
                      "reward_last10": sum(last) / max(len(last), 1)}, indent=2))


if __name__ == "__main__":
    main()
