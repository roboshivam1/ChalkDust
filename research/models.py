"""Candidate local models: where the GGUF comes from, how to serve it, and the
licence facts the findings rely on (each checked against the model card or
LICENSE file named in docs/research/models-2026-11.md).

A model is fetched on first use into .lab/models/<key>/ (git-ignored).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO / ".lab" / "models"


@dataclass(frozen=True)
class ModelEntry:
    key: str
    repo: str                 # Hugging Face repo holding the GGUF
    file: str                 # GGUF filename (first shard if split)
    licence: str
    params_b: float           # total parameters (billions)
    active_b: float           # active parameters per token (MoE) or = params_b
    tiers: tuple[str, ...]    # which target machines it is a candidate for (A/B/C)
    ctx: int = 32768
    extra: tuple[str, ...] = ()           # llama-server flags (offload is left to llama.cpp --fit)
    chat_extra: dict = field(default_factory=dict)  # per-request extras (e.g. disable thinking)
    note: str = ""

    @property
    def path(self) -> Path:
        return MODELS_DIR / self.key / self.file


REGISTRY: dict[str, ModelEntry] = {}


def add(e: ModelEntry) -> None:
    REGISTRY[e.key] = e


NOTHINK = {"chat_template_kwargs": {"enable_thinking": False}}

add(ModelEntry("qwen3-4b-2507", "unsloth/Qwen3-4B-Instruct-2507-GGUF",
               "Qwen3-4B-Instruct-2507-Q4_K_M.gguf", "apache-2.0", 4.0, 4.0, ("A", "B", "C"),
               note="baseline from Aug 2025; thin-slice model"))
# Tier C (8-16 GB RAM, CPU/iGPU)
add(ModelEntry("qwen3.5-2b", "unsloth/Qwen3.5-2B-GGUF", "Qwen3.5-2B-Q4_K_M.gguf", "apache-2.0",
               2.0, 2.0, ("C",), chat_extra=NOTHINK))
add(ModelEntry("qwen3.5-4b", "unsloth/Qwen3.5-4B-GGUF", "Qwen3.5-4B-Q4_K_M.gguf", "apache-2.0",
               4.0, 4.0, ("C", "B"), chat_extra=NOTHINK))
add(ModelEntry("granite-4.2-3b", "bartowski/granite-4.2-3b-GGUF", "granite-4.2-3b-Q4_K_M.gguf",
               "apache-2.0", 3.0, 3.0, ("C",)))
add(ModelEntry("gemma-4-e2b", "unsloth/gemma-4-E2B-it-GGUF", "gemma-4-E2B-it-Q4_K_M.gguf",
               "apache-2.0", 5.1, 2.3, ("C",)))
# Tier B (M1 16 GB)
add(ModelEntry("qwen3.5-9b", "unsloth/Qwen3.5-9B-GGUF", "Qwen3.5-9B-Q4_K_M.gguf", "apache-2.0",
               9.0, 9.0, ("B",), chat_extra=NOTHINK))
add(ModelEntry("granite-4.2-8b", "bartowski/granite-4.2-8b-GGUF", "granite-4.2-8b-Q4_K_M.gguf",
               "apache-2.0", 8.0, 8.0, ("B",)))
add(ModelEntry("gemma-4-e4b", "unsloth/gemma-4-E4B-it-GGUF", "gemma-4-E4B-it-Q4_K_M.gguf",
               "apache-2.0", 8.0, 4.5, ("B",)))
# Tier A (12 GB VRAM + 32 GB RAM)
add(ModelEntry("gemma-4-12b", "unsloth/gemma-4-12B-it-GGUF", "gemma-4-12b-it-Q4_K_M.gguf",
               "apache-2.0", 12.0, 12.0, ("A",)))
add(ModelEntry("qwen3.6-35b-a3b", "unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
               "apache-2.0", 35.0, 3.0, ("A",), chat_extra=NOTHINK,
               note="does not fit 12 GB: --fit moves expert weights to RAM"))
add(ModelEntry("qwen3.8-27b", "unsloth/Qwen3.8-27B-GGUF", "Qwen3.8-27B-UD-IQ3_XXS.gguf", "apache-2.0",
               27.0, 27.0, ("A",), chat_extra=NOTHINK))
add(ModelEntry("gpt-oss-20b", "ggml-org/gpt-oss-20b-GGUF", "gpt-oss-20b-MXFP4.gguf", "apache-2.0",
               21.0, 3.6, ("A",), chat_extra={"chat_template_kwargs": {"reasoning_effort": "low"}}))


def local_entries() -> None:
    """Models this run converted itself (research/export_gguf.py): keys ending
    in -self (an untuned base) or containing -sft (tuned). Licence and size
    are inherited from the base entry named in export.json's key prefix."""
    import json as _json
    for d in MODELS_DIR.glob("*"):
        if not (d.name.endswith("-self") or "-sft" in d.name) or d.name in REGISTRY:
            continue
        ggufs = sorted(d.glob("*-Q4_K_M.gguf"))
        if not ggufs:
            continue
        base = next((REGISTRY[k] for k in sorted(REGISTRY, key=len, reverse=True) if d.name.startswith(k)), None)
        meta = {}
        if (d / "export.json").exists():
            meta = _json.loads((d / "export.json").read_text(encoding="utf-8"))
        add(ModelEntry(d.name, "local", ggufs[0].name, base.licence if base else "?",
                       base.params_b if base else 0, base.active_b if base else 0,
                       base.tiers if base else (), chat_extra=base.chat_extra if base else {},
                       note=f"converted locally from {meta.get('hf_dir', '?')}"))


local_entries()


def ensure(key: str) -> Path:
    e = REGISTRY[key]
    if e.repo == "local":
        return e.path
    if e.path.exists():
        return e.path
    from huggingface_hub import hf_hub_download, list_repo_files
    files = list_repo_files(e.repo)
    want = [f for f in files if f == e.file or (e.file.split("-00001-of-")[0] in f and f.endswith(".gguf")
                                                  and "-of-" in e.file)]
    for f in want or [e.file]:
        hf_hub_download(e.repo, f, local_dir=str(MODELS_DIR / e.key))
    return e.path
