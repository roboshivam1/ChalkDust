"""Run an experiment: models x methods x topics x seeds.

    python -m research.run --exp E1 --models qwen3-4b-2507 \
        --methods staged staged:n=4 oneshot --topics dev --seeds 0 1

Each (model, method, topic, seed) appends ONE record to
research/results/<exp>.jsonl (committed; append-only, so runs stay
comparable) and writes its raw trace -- every prompt, every sample, every
validator finding -- under .lab/runs/<run_id>/ (git-ignored).

Method strings: "staged", "staged:n=4,repair=1,calc=1", "oneshot",
"oneshot:free=1" (free=1 turns grammar-constrained decoding off).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from research.content import (answer_candidates, check_answer,
                              derivation_consistency, structure_checks)
from research.generate import generate_oneshot, generate_staged
from research.llm import LlamaServer, ServerConfig
from research.models import REGISTRY, ensure
from research.score import assemble_spec, ensure_miktex_first, slug, validate_spec_cli
from research.topics import topic_set

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "research" / "results"
RUNS = REPO / ".lab" / "runs"


def parse_method(s: str) -> tuple[str, dict]:
    name, _, opts = s.partition(":")
    kw = {}
    for kv in filter(None, opts.split(",")):
        k, _, v = kv.partition("=")
        kw[k] = int(v) if v.lstrip("-").isdigit() else float(v) if v.replace(".", "", 1).isdigit() else v
    return name, kw


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return "?"


def summarise(topic, gen: dict, cli: dict | None) -> dict:
    scores = gen["scores"]
    beats = gen["beats"]
    levels = Counter(s["level"] for s in scores)
    r1 = Counter(e.split(":")[0].split(".")[0] if e else "?" for s in scores for e in s["rung1_errors"])
    kinds = Counter(f["kind"] for s in scores for f in s["findings"])
    p_tok, g_tok = gen["trace"].tokens
    comps = gen["trace"].calls
    tok_s = [c["gen_tok_s"] for call in comps for c in call["completions"] if c["gen_tok_s"]]
    answer = check_answer(topic.key, answer_candidates(beats)) if topic.key else {"status": "n/a"}
    # What the student HEARS: the answer in the last two narrations, checked on its own
    # (the first video showed a correct AnswerBox under narration saying a wrong number).
    spoken = check_answer(topic.key, [" ".join(str(b.get("narration", "")) for b in beats[-2:])])         if (topic.key and beats) else {"status": "n/a"}
    valid = sum(1 for s in scores if s["ok"])
    clean = sum(1 for s in scores if s.get("clean"))
    flags = Counter(f.split(":")[0] for s in scores for f in s.get("flags", []))
    return {
        "plan_ok": gen["plan"] is not None and bool(gen["plan"].get("beats") if gen["plan"] else False) or (bool(beats)),
        "n_beats": len(beats),
        "beats_valid": valid,
        "beat_valid_rate": valid / len(beats) if beats else 0.0,
        "all_beats_valid": bool(beats) and valid == len(beats),
        "beats_clean": clean,
        "all_beats_clean": bool(beats) and clean == len(beats),
        "flag_counts": dict(flags),
        "spec_valid_cli": None if cli is None else cli["ok"],
        "levels": {str(k): v for k, v in sorted(levels.items())},
        "rung1_error_fields": dict(r1.most_common(8)),
        "finding_kinds": dict(kinds),
        "harness_faults": sum(1 for s in scores if s["harness_fault"]),
        "escape_repairs": sum(s["escape_repairs"] for s in scores),
        "mech_repaired_beats": sum(1 for s in scores if s["repaired_mechanically"]),
        "components": [s["component"] for s in scores],
        "text_chars_mean": (sum(s["text_chars"] for s in scores) / len(scores)) if scores else 0,
        "narration_words_mean": (sum(s["narration_words"] for s in scores) / len(scores)) if scores else 0,
        "answer": answer,
        "answer_spoken": spoken,
        "derivation": derivation_consistency(beats, topic.key),
        "structure": structure_checks(gen["plan"], beats, topic.key),
        "samples": sum(p.get("samples", 1) for p in gen["per_beat"]) or len(beats),
        "repair_rounds": sum(p.get("repair_rounds", 0) for p in gen["per_beat"]),
        "llm_seconds": gen["trace"].llm_seconds,
        "validator_seconds": sum(p.get("validator_seconds", 0) for p in gen["per_beat"]) or sum(s["seconds"] for s in scores),
        "wall_seconds": gen["wall_s"],
        "prompt_tokens": p_tok,
        "gen_tokens": g_tok,
        "gen_tok_s_mean": (sum(tok_s) / len(tok_s)) if tok_s else 0,
        "answer_calc_value": (gen["plan"] or {}).get("answer_calc_value") if gen["plan"] else None,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--methods", nargs="+", default=["staged"])
    ap.add_argument("--topics", default="dev")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--ngl", type=int, default=999, help="GPU layers; 0 = CPU-only proxy")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--ctx", type=int, default=None)
    ap.add_argument("--server-extra", default="", help="extra llama-server flags, space separated")
    ap.add_argument("--no-cli", action="store_true", help="skip chalkdust validate on all-valid specs")
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)

    ensure_miktex_first()
    RESULTS.mkdir(parents=True, exist_ok=True)
    run_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S") + (f"-{a.tag}" if a.tag else "")
    topics = topic_set(a.topics)
    out_path = RESULTS / f"{a.exp}.jsonl"
    commit = git_commit()
    for mkey in a.models:
        entry = REGISTRY[mkey]
        path = ensure(mkey)
        ctx = a.ctx or max(8192 * a.parallel, 16384)
        cfg = ServerConfig(model_path=str(path), ctx=ctx, n_gpu_layers=a.ngl, parallel=a.parallel,
                           threads=a.threads, extra=tuple(entry.extra) + tuple(a.server_extra.split()))
        run_dir = RUNS / run_id / mkey
        with LlamaServer(cfg, run_dir / "server.log") as srv:
            print(f"[{mkey}] loaded in {srv.load_seconds:.1f}s", flush=True)
            for mstr in a.methods:
                mname, kw = parse_method(mstr)
                for topic in topics:
                    for seed in a.seeds:
                        tdir = run_dir / slug(mstr) / f"{topic.id}_s{seed}"
                        tdir.mkdir(parents=True, exist_ok=True)
                        t0 = time.perf_counter()
                        if mname == "staged":
                            gen = generate_staged(srv.url, topic, tdir / "work", seed=seed,
                                                  temperature=a.temperature, n=int(kw.get("n", 1)),
                                                  repair=int(kw.get("repair", 0)),
                                                  constrained=not kw.get("free"), calc=bool(kw.get("calc")))
                        elif mname == "oneshot":
                            gen = generate_oneshot(srv.url, topic, tdir / "work", seed=seed,
                                                   temperature=a.temperature, constrained=not kw.get("free"))
                        else:
                            raise SystemExit(f"unknown method {mname}")
                        srv.rss_mib()
                        spec = assemble_spec(slug(f"{topic.id}-{mkey}-s{seed}"), gen["beats"])
                        (tdir / "spec.json").write_text(json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8")
                        (tdir / "trace.json").write_text(json.dumps(
                            {"plan": gen["plan"], "scores": gen["scores"], "per_beat": gen["per_beat"],
                             "calls": gen["trace"].calls}, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
                        all_valid = bool(gen["scores"]) and all(s["ok"] for s in gen["scores"])
                        cli = validate_spec_cli(tdir / "spec.json", tdir / "cli-work") if (all_valid and not a.no_cli) else None
                        rec = {
                            "exp": a.exp, "run_id": run_id, "ts": dt.datetime.now().isoformat(timespec="seconds"),
                            "commit": commit, "host": platform.node(), "model": mkey, "quant": entry.file,
                            "method": mstr, "topic": topic.id, "subject": topic.subject, "seed": seed,
                            "temperature": a.temperature, "ngl": a.ngl, "threads": a.threads, "parallel": a.parallel,
                            "trace_dir": str(tdir.relative_to(REPO)),
                            **summarise(topic, gen, cli),
                            "cli_seconds": cli["seconds"] if cli else None,
                            "cli_output": cli["output"][-600:] if cli and not cli["ok"] else None,
                            "total_seconds": time.perf_counter() - t0,
                        }
                        with open(out_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                        print(f"[{mkey}] {mstr} {topic.id} s{seed}: beats {rec['beats_valid']}/{rec['n_beats']} "
                              f"clean {rec['beats_clean']} cli={rec['spec_valid_cli']} answer={rec['answer'].get('status')}/{rec['answer_spoken'].get('status')} "
                              f"struct={rec['structure']['score']:.2f} llm={rec['llm_seconds']:.0f}s "
                              f"val={rec['validator_seconds']:.0f}s tok/s={rec['gen_tok_s_mean']:.0f}", flush=True)
            st = srv.stats()
        # server-level stats as a separate record type
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"exp": a.exp, "run_id": run_id, "model": mkey, "record": "server",
                                "ngl": a.ngl, "threads": a.threads, "ctx": ctx, "parallel": a.parallel,
                                **st}) + "\n")
        print(f"[{mkey}] server stats: {st}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
