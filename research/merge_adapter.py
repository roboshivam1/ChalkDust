"""Merge a LoRA adapter into a bf16 copy of its base, on the CPU, for GGUF export.

    .lab/venv-train/Scripts/python research/merge_adapter.py BASE_DIR ADAPTER_DIR OUT_DIR

Used after GRPO (train_grpo.py saves only the adapter). The adapter is merged
into the bf16 weights even when it was trained against a 4-bit (QLoRA) load
of the same base, the usual practice; the small mismatch is the same one the
SFT model's merge has.
"""

import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_dir, adapter_dir, out_dir = sys.argv[1:4]
base = AutoModelForCausalLM.from_pretrained(base_dir, dtype=torch.bfloat16, device_map={"": "cpu"})
merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
merged.save_pretrained(out_dir, safe_serialization=True)
AutoTokenizer.from_pretrained(base_dir).save_pretrained(out_dir)
print("merged ->", out_dir)
