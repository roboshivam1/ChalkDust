"""HF safetensors -> GGUF (f16) -> Q4_K_M, with the pinned llama.cpp b12466 tools.

    .lab/venv-train/Scripts/python research/export_gguf.py --hf .lab/hf/granite-4.2-3b \
        --key granite-4.2-3b-self

Writes .lab/models/<key>/<key>-Q4_K_M.gguf and registers nothing on disk: the
research/models.py registry picks up .lab/models/<key>/*-Q4_K_M.gguf for keys
ending in -self or -sft (local_entries()). The untuned base and every tuned
model go through this same converter and quantiser, with no importance
matrix, so the tuned-vs-untuned comparison is not confounded by converting
differently (a downloaded GGUF may use an imatrix).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / ".lab" / "bin" / "llama-src" / "llama.cpp-b12466"
QUANT = REPO / ".lab" / "bin" / "llama-b12466" / "llama-quantize.exe"
MODELS = REPO / ".lab" / "models"


def export(hf_dir: Path, key: str, qtype: str = "Q4_K_M", keep_f16: bool = False) -> Path:
    out_dir = MODELS / key
    out_dir.mkdir(parents=True, exist_ok=True)
    f16 = out_dir / f"{key}-F16.gguf"
    q = out_dir / f"{key}-{qtype}.gguf"
    t0 = time.time()
    env_py = sys.executable
    subprocess.run([env_py, str(SRC / "convert_hf_to_gguf.py"), str(hf_dir), "--outfile", str(f16),
                    "--outtype", "f16"], check=True,
                   env={**__import__("os").environ, "PYTHONPATH": str(SRC / "gguf-py")})
    t1 = time.time()
    subprocess.run([str(QUANT), str(f16), str(q), qtype], check=True, cwd=str(QUANT.parent))
    t2 = time.time()
    if not keep_f16:
        f16.unlink()
    meta = {"hf_dir": str(hf_dir), "key": key, "qtype": qtype, "convert_s": t1 - t0, "quantize_s": t2 - t1,
            "size_bytes": q.stat().st_size, "llama_cpp": "b12466", "imatrix": None}
    (out_dir / "export.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return q


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--qtype", default="Q4_K_M")
    ap.add_argument("--keep-f16", action="store_true")
    a = ap.parse_args()
    export(Path(a.hf), a.key, a.qtype, a.keep_f16)
