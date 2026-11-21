"""A thin, measured client for llama.cpp's llama-server.

Every call returns what the experiment log needs: the text, token counts,
llama.cpp's own prompt/generation timings, wall time and the seed. The server
manager samples GPU memory while a model is loaded, so each run records the
VRAM it actually used rather than a number from a model card.

The GPU is one lane: run one LlamaServer at a time.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import psutil
import requests

REPO = Path(__file__).resolve().parents[1]
LLAMA_DIR = REPO / ".lab" / "bin" / "llama-b12466"
LLAMA_SERVER = LLAMA_DIR / "llama-server.exe"

# Per-model request extras (e.g. turn thinking off), set by the runner for the
# model currently being served.
REQUEST_EXTRA: dict = {}


def gpu_used_mib() -> int | None:
    """Total used VRAM on GPU 0 (all processes). None without nvidia-smi."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
        return int(out[0])
    except Exception:
        return None


class GpuSampler:
    """Polls total VRAM in a thread; peak minus the pre-load baseline is what
    the model (weights + KV cache + compute buffers) costs."""

    def __init__(self, period: float = 0.5) -> None:
        self.period = period
        self.baseline = gpu_used_mib()
        self.peak = self.baseline or 0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            v = gpu_used_mib()
            if v is not None:
                self.peak = max(self.peak, v)
            self._stop.wait(self.period)

    def start(self) -> "GpuSampler":
        self._t.start()
        return self

    def stop(self) -> dict:
        self._stop.set()
        self._t.join(timeout=5)
        return {"vram_baseline_mib": self.baseline, "vram_peak_mib": self.peak,
                "vram_model_mib": (self.peak - self.baseline) if self.baseline is not None else None}


@dataclass
class ServerConfig:
    model_path: str
    ctx: int = 16384
    n_gpu_layers: int | None = None  # None = llama.cpp '--fit' decides; 0 = CPU only (student proxy)
    parallel: int = 4              # slots: concurrent requests are batched
    threads: int | None = None     # CPU threads (None = llama.cpp default)
    port: int = 8091
    flash_attn: str = "on"
    # Host-RAM caps. llama-server b12466 keeps a cross-request prompt cache of
    # up to 8 GiB in system RAM by default (--cache-ram 8192) and up to 32
    # context checkpoints per slot for SWA/hybrid models (Gemma 4, Qwen3.5).
    # The cache is off: under Gemma 4 12B one idle slot's saved state is
    # 0.9-1.2 GB, so a 1 GiB cap held at most one entry and copied ~1 GB out of
    # the GPU on almost every request (2026-11-21 server.log), while system
    # commit reached 51.3 of 53.4 GB. Each slot still reuses its own KV prefix.
    # Note: llama-server's private bytes sit near its VRAM use even when
    # everything is on the GPU (10.1 GB after loading Gemma 4 12B, with 0.6 GB
    # of host buffers in the load log); our inference is that Windows charges
    # GPU allocations to commit, so judge memory by growth after load.
    cache_ram_mib: int = 0
    ctx_checkpoints: int = 4
    extra: tuple[str, ...] = ()    # e.g. ("--n-cpu-moe", "20") or draft-model flags

    def argv(self) -> list[str]:
        a = [str(LLAMA_SERVER), "-m", self.model_path, "-c", str(self.ctx),
             "--parallel", str(self.parallel),
             "--port", str(self.port), "--host", "127.0.0.1", "--jinja",
             "-fa", self.flash_attn, "--no-webui", "--metrics",
             "--cache-ram", str(self.cache_ram_mib), "--ctx-checkpoints", str(self.ctx_checkpoints)]
        if self.n_gpu_layers is not None:
            a += ["-ngl", str(self.n_gpu_layers)]
        if self.threads:
            a += ["-t", str(self.threads)]
        return a + list(self.extra)


class LlamaServer:
    """Start llama-server for one model, wait until healthy, stop on exit."""

    def __init__(self, cfg: ServerConfig, log_path: Path) -> None:
        self.cfg = cfg
        self.log_path = log_path
        self.proc: subprocess.Popen | None = None
        self.sampler: GpuSampler | None = None
        self.load_seconds: float | None = None
        self.peak_rss_mib = 0

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.cfg.port}"

    def __enter__(self) -> "LlamaServer":
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.sampler = GpuSampler().start()
        t0 = time.perf_counter()
        self._log = open(self.log_path, "w", encoding="utf-8", errors="replace")
        self.proc = subprocess.Popen(self.cfg.argv(), stdout=self._log, stderr=subprocess.STDOUT,
                                     cwd=str(LLAMA_DIR),
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        deadline = time.time() + 600
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited {self.proc.returncode}; see {self.log_path}")
            try:
                if requests.get(self.url + "/health", timeout=2).status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(0.5)
        else:
            self.__exit__(None, None, None)
            raise TimeoutError(f"llama-server not healthy after 600 s; see {self.log_path}")
        self.load_seconds = time.perf_counter() - t0
        return self

    def rss_mib(self) -> int:
        try:
            r = int(psutil.Process(self.proc.pid).memory_info().rss / 2**20)
            self.peak_rss_mib = max(self.peak_rss_mib, r)
            return r
        except Exception:
            return 0

    def placement(self) -> dict:
        """What llama.cpp actually put on the GPU (from its log): with --fit
        the layer split is chosen at load time, so it must be recorded."""
        try:
            log = self.log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return {}
        out = {}
        m = re.search(r"offloaded (\d+)/(\d+) layers to GPU", log)
        if m:
            out["layers_on_gpu"], out["layers_total"] = int(m.group(1)), int(m.group(2))
        out["fit_lines"] = [l.strip()[:200] for l in log.splitlines() if "fit" in l.lower() and "llama_params_fit" in l][:8]
        m = re.search(r"n_cpu_moe\D+(\d+)", log)
        if m:
            out["n_cpu_moe"] = int(m.group(1))
        return out

    def stats(self) -> dict:
        s = self.sampler.stop() if self.sampler else {}
        self.sampler = None
        return {**s, "load_seconds": self.load_seconds, "server_peak_rss_mib": self.peak_rss_mib,
                **self.placement()}

    def __exit__(self, *exc) -> None:
        if self.proc and self.proc.poll() is None:
            self.rss_mib()
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if self.sampler:
            self.sampler.stop()
        try:
            self._log.close()
        except Exception:
            pass


@dataclass
class Completion:
    text: str
    ok: bool
    error: str | None
    prompt_tokens: int
    completion_tokens: int
    prompt_ms: float
    gen_ms: float
    gen_tok_s: float
    wall_s: float
    finish_reason: str | None
    seed: int
    temperature: float
    constrained: bool
    reasoning_chars: int = 0     # thinking text returned separately: must be 0 (thinking off)

    def to_dict(self) -> dict:
        return asdict(self)


def chat(url: str, messages: list[dict], *, schema: dict | None = None,
         temperature: float = 0.7, top_p: float = 0.95, seed: int = 0,
         max_tokens: int = 2048, timeout: float = 900,
         extra: dict | None = None) -> Completion:
    """One chat completion. With `schema`, decoding is grammar-constrained to
    it (llama.cpp converts the JSON schema to a GBNF grammar)."""
    body = {"messages": messages, "temperature": temperature, "top_p": top_p,
            "seed": seed, "max_tokens": max_tokens, "cache_prompt": True}
    if schema is not None:
        body["response_format"] = {"type": "json_schema",
                                   "json_schema": {"name": "out", "schema": schema, "strict": True}}
    if REQUEST_EXTRA:
        body.update(REQUEST_EXTRA)
    if extra:
        body.update(extra)
    t0 = time.perf_counter()
    try:
        r = requests.post(url + "/v1/chat/completions", json=body, timeout=timeout)
        wall = time.perf_counter() - t0
        if r.status_code != 200:
            return Completion("", False, f"HTTP {r.status_code}: {r.text[:500]}", 0, 0, 0, 0, 0,
                              wall, None, seed, temperature, schema is not None)
        d = r.json()
        ch = d["choices"][0]
        tm = d.get("timings", {})
        us = d.get("usage", {})
        rc = len(ch["message"].get("reasoning_content") or "")
        return Completion(ch["message"].get("content") or "", True, None,
                          int(us.get("prompt_tokens", tm.get("prompt_n", 0))),
                          int(us.get("completion_tokens", tm.get("predicted_n", 0))),
                          float(tm.get("prompt_ms", 0)), float(tm.get("predicted_ms", 0)),
                          float(tm.get("predicted_per_second", 0)), wall,
                          ch.get("finish_reason"), seed, temperature, schema is not None, rc)
    except Exception as exc:  # network / timeout: a harness fault, recorded as such
        return Completion("", False, f"{type(exc).__name__}: {exc}", 0, 0, 0, 0, 0,
                          time.perf_counter() - t0, None, seed, temperature, schema is not None)


def chat_many(url: str, messages: list[dict], n: int, *, seed0: int, **kw) -> list[Completion]:
    """n samples of the same prompt, sent concurrently so llama-server batches
    them across its slots (best-of-N at close to single-sample latency)."""
    with ThreadPoolExecutor(max_workers=n) as ex:
        futs = [ex.submit(chat, url, messages, seed=seed0 + i, **kw) for i in range(n)]
        return [f.result() for f in futs]


def parse_json(text: str) -> tuple[object | None, str | None]:
    """Parse model output as JSON; tolerate a ```json fence or prose around one
    object (unconstrained decoding). Returns (value, error)."""
    s = text.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s
        s = s.rsplit("```", 1)[0]
    try:
        return json.loads(s), None
    except json.JSONDecodeError as exc:
        a, b = s.find("{"), s.rfind("}")
        if a != -1 and b > a:
            try:
                return json.loads(s[a:b + 1]), None
            except json.JSONDecodeError as exc2:
                return None, f"json: {exc2}"
        return None, f"json: {exc}"
