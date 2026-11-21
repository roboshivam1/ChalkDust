"""Template parity: the prompt the trainer formats must be the prompt
llama-server formats at inference, or SFT trains on text the model never sees.

    .lab/venv-train/Scripts/python research/check_template.py --hf .lab/hf/granite-4.2-3b \
        --url http://127.0.0.1:8091 --data .lab/datasets/sft-v1/train.jsonl

Renders the first N rows' prompts with the HF tokenizer's chat template
(add_generation_prompt=True) and with llama-server's /apply-template, and
prints the first difference. Exit 1 on any mismatch.
"""

from __future__ import annotations

import argparse
import json
import sys

import requests
from transformers import AutoTokenizer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True)
    ap.add_argument("--url", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("-n", type=int, default=5)
    ap.add_argument("--template-kwargs", default='{"enable_thinking": false}')
    a = ap.parse_args()
    tkw = json.loads(a.template_kwargs)
    tok = AutoTokenizer.from_pretrained(a.hf)
    rows = [json.loads(l) for l in open(a.data, encoding="utf-8")][: a.n]
    bad = 0
    for i, r in enumerate(rows):
        hf = tok.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True, **tkw)
        srv = requests.post(a.url + "/apply-template", json={"messages": r["prompt"], "chat_template_kwargs": tkw},
                            timeout=60).json()["prompt"]
        if hf == srv:
            print(f"row {i}: identical ({len(hf)} chars)")
            continue
        bad += 1
        j = next((k for k in range(min(len(hf), len(srv))) if hf[k] != srv[k]), min(len(hf), len(srv)))
        print(f"row {i}: DIFFER at char {j} (hf {len(hf)} / server {len(srv)})")
        print("  hf    :", repr(hf[max(0, j - 80): j + 120]))
        print("  server:", repr(srv[max(0, j - 80): j + 120]))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
