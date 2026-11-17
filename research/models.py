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
    extra: tuple[str, ...] = ()           # llama-server flags (e.g. MoE offload)
    chat_extra: dict = field(default_factory=dict)  # per-request extras (e.g. disable thinking)
    note: str = ""

    @property
    def path(self) -> Path:
        return MODELS_DIR / self.key / self.file


REGISTRY: dict[str, ModelEntry] = {}


def add(e: ModelEntry) -> None:
    REGISTRY[e.key] = e


add(ModelEntry("qwen3-4b-2507", "unsloth/Qwen3-4B-Instruct-2507-GGUF",
               "Qwen3-4B-Instruct-2507-Q4_K_M.gguf", "apache-2.0", 4.0, 4.0, ("A", "B", "C"),
               note="baseline from Aug 2025; thin-slice model"))


def ensure(key: str) -> Path:
    e = REGISTRY[key]
    if e.path.exists():
        return e.path
    from huggingface_hub import hf_hub_download, list_repo_files
    files = list_repo_files(e.repo)
    want = [f for f in files if f == e.file or (e.file.split("-00001-of-")[0] in f and f.endswith(".gguf")
                                                  and "-of-" in e.file)]
    for f in want or [e.file]:
        hf_hub_download(e.repo, f, local_dir=str(MODELS_DIR / e.key))
    return e.path
