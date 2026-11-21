"""Final-report tables from research/results/*.jsonl (read-only).

The T2 block is the training result: tuned against untuned on HELDOUT topics,
paired by (topic, seed, method). It reports a paired bootstrap 95% CI and an
exact sign test. The primary metric is the first-sample clean rate (METHODOLOGY.md
§6), read from each trace's round-0 call, so best-of-N and repair can't mask it.

    python -m research.report                      # E2, T2 (+paired), E4, E5
    python -m research.report --exp E1c-work       # one experiment, general table
    python -m research.report --exp smoke-pipeline --paired-exp smoke-pipeline \
        --tuned granite-4.2-3b-sft-smoke --base granite-4.2-3b-self

Reuses research.analyze (load / rescore / table / to_markdown). Nothing is written.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from math import comb
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from research import analyze                      # noqa: E402
from research.llm import parse_json               # noqa: E402
from research.models import REGISTRY              # noqa: E402
from research.score import quality_flags          # noqa: E402

COLS = ["n", "valid", "clean", "allclean", "ans", "spk", "struct", "wall_s", "tok/s", "vram"]
METRICS = [("first-sample clean, per beat (primary)", "fs"),
           ("first-sample all-clean, per video", "fs_all"),("allclean", "allclean"), ("ans", "ans"), ("spk", "spk")]


def get(exp):
    """(recs, servers, n_unrescored) or None when the experiment has no file."""
    if not (analyze.RESULTS / f"{exp}.jsonl").exists():
        return None
    recs, servers = analyze.load(exp)
    ok = [r for r in recs if r.get("rescored")]
    return ok, servers, len(recs) - len(ok)


def cfg(r):
    """Scoring workers / prompt-cache MiB; 'pre' = record predates the fields (E1-E1c)."""
    if "score_workers" not in r and "cache_ram_mib" not in r:
        return "pre"
    return f"w{r.get('score_workers')}/c{r.get('cache_ram_mib')}"


def md(rows, cols):
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(out)


def model_rows(recs, servers):
    """One row per model x method via analyze.table, plus licence, tier, settings flag."""
    cells = {}
    for t in analyze.table(recs, servers, ["model", "method"]):
        cells[(t["model"], t["method"])] = t
    cfgs = defaultdict(Counter)
    for r in recs:
        cfgs[(r["model"], r["method"])][cfg(r)] += 1
    rows = []
    for (m, meth), t in cells.items():
        e = REGISTRY.get(m)
        c = cfgs[(m, meth)]
        row = {k: (f"{v:.0f}" if k in ("wall_s", "tok/s", "vram") and isinstance(v, float)
                   else f"{v:.2f}" if isinstance(v, float) else ("" if v is None else v))
               for k, v in t.items()}
        row.update({"licence": e.licence if e else "?", "tier": "/".join(e.tiers) if e else "?",
                    "cfg": next(iter(c)) if len(c) == 1 else "MIXED " + ", ".join(f"{k} x{v}" for k, v in c.items())})
        rows.append(row)
    return sorted(rows, key=lambda r: (r["tier"], r["model"], r["method"]))


def tables(exp, by_tier=False):
    d = get(exp)
    if d is None:
        return f"_{exp}: not run yet (no results file)._"
    recs, servers, bad = d
    if not recs:
        return f"_{exp}: no video records yet._"
    rows = model_rows(recs, servers)
    note = f"\n\n_{len(recs)} videos; {bad} excluded (saved spec/trace missing)._" if bad else f"\n\n_{len(recs)} videos._"
    if any(r["cfg"].startswith("MIXED") for r in rows):
        note += " Rows marked MIXED combine different (score_workers, cache_ram_mib); wall_s is not like-for-like."
    cols = ["model", "licence", "method"] + COLS + ["cfg"]
    if not by_tier:
        return md(rows, cols) + note
    out = []
    for tier in sorted({r["tier"] for r in rows}):
        out.append(f"**Tier {tier}**\n\n" + md([r for r in rows if r["tier"] == tier], cols))
    return "\n\n".join(out) + note


# --- paired comparison ------------------------------------------------------------

def first_sample_clean(r):
    """Share of beats whose FIRST sample (round 0, completion 0) was valid and
    flag-free. Source: trace.json calls 'beatN.r0' (generate.fill_beat -> Trace.add,
    scores per sample). oneshot has one sample, so the record's own beats_clean is it."""
    if r["method"].startswith("oneshot"):
        return r["beats_clean"] / r["n_beats"] if r.get("beats_clean") is not None and r["n_beats"] else 0.0
    try:
        calls = json.loads((REPO / r["trace_dir"] / "trace.json").read_text(encoding="utf-8"))["calls"]
    except (OSError, KeyError):
        return None
    hits = []
    for c in calls:
        if not re.fullmatch(r"beat\d+\.r0", c["step"]) or not c.get("scores"):
            continue
        sc, comp = c["scores"][0], c["completions"][0]
        v, _ = parse_json(comp["text"]) if comp.get("text") else (None, "empty")
        v = v if isinstance(v, dict) else {}
        beat = {"component": sc.get("component"), "narration": v.get("narration", ""),
                "params": v.get("params", {}) if isinstance(v.get("params"), dict) else {}}
        hits.append(sc.get("level") == 3 and not quality_flags(beat))
    return sum(hits) / len(hits) if hits else None


def video_metrics(r):
    fs = first_sample_clean(r)
    return {"fs": fs, "fs_all": None if fs is None else float(fs == 1.0),
            "allclean": 1.0 if r.get("all_beats_clean") else 0.0,
            "ans": None if r["answer"]["status"] == "n/a" else float(r["answer"]["status"] == "correct"),
            "spk": float(r["answer_spoken"]["status"] == "correct")
            if r["answer_spoken"]["status"] in ("correct", "wrong", "not_found") else None}


def sign_p(diffs):
    d = [x for x in diffs if x != 0]
    n, k = len(d), sum(x > 0 for x in d)
    if not n:
        return None
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(k, n - k) + 1)) / 2 ** n)


def boot_ci(diffs, B=10_000, seed=20261121):
    rng, n = random.Random(seed), len(diffs)
    means = sorted(sum(rng.choices(diffs, k=n)) / n for _ in range(B))
    return means[int(0.025 * B)], means[int(0.975 * B) - 1]


def paired_block(recs, a, b, label):
    """a minus b, paired by (topic, seed, method); a pair needs the metric on both sides."""
    side = {a: {}, b: {}}
    for r in recs:
        if r["model"] in side:
            side[r["model"]][(r["topic"], r["seed"], r["method"])] = video_metrics(r)
    keys = sorted(set(side[a]) & set(side[b]))
    head = f"**{label}: {a} vs {b}** ({len(keys)} pairs of {len(side[a])} / {len(side[b])} videos)"
    if not keys:
        return head + "\n\n_no matching (topic, seed, method) pairs yet._"
    rows = []
    for scope in [None] + sorted({k[2] for k in keys}):
        ks = [k for k in keys if scope is None or k[2] == scope]
        for name, m in METRICS:
            ps = [(side[a][k][m], side[b][k][m]) for k in ks]
            ps = [p for p in ps if p[0] is not None and p[1] is not None]
            if not ps:
                rows.append({"method": scope or "all", "metric": name, "n": 0})
                continue
            diffs = [x - y for x, y in ps]
            lo, hi = boot_ci(diffs)
            p = sign_p(diffs)
            rows.append({"method": scope or "all", "metric": name, "n": len(ps),
                         a: f"{sum(x for x, _ in ps) / len(ps):.3f}", b: f"{sum(y for _, y in ps) / len(ps):.3f}",
                         "diff": f"{sum(diffs) / len(ps):+.3f}", "95% CI": f"[{lo:+.3f}, {hi:+.3f}]",
                         "sign p": "n/a (all ties)" if p is None else f"{p:.4f}",
                         "ties": sum(x == 0 for x in diffs)})
    return head + "\n\n" + md(rows, ["method", "metric", "n", a, b, "diff", "95% CI", "sign p", "ties"])


def paired_section(exp, tuned, base, grpo):
    d = get(exp)
    if d is None:
        return f"_{exp}: not run yet (no results file)._"
    recs, servers, bad = d
    out = [tables(exp)]
    if not recs:
        return out[0]
    out.append("_First-sample clean = share of beats whose round-0 first sample is valid and flag-free, read from "
               "each trace.json (`beatN.r0` scores[0], re-flagged with current `quality_flags`; oneshot: the single "
               "sample). allclean/ans/spk are the final video, equal to the first sample only when n=1, repair=0. "
               "Bootstrap: 10,000 resamples, RNG seed 20261121; sign test exact two-sided, ties dropped._")
    out.append(paired_block(recs, tuned, base, "Tuned vs untuned"))
    out.append(paired_block(recs, grpo, tuned, "GRPO vs SFT"))
    known = {tuned, base, grpo}
    try:
        teacher = json.loads((REPO / ".lab" / "queue" / "teacher.json").read_text(encoding="utf-8")).get("teacher")
    except (OSError, ValueError):
        teacher = None
    teach = [r for r in recs if (r["model"] == teacher if teacher else r["model"] not in known)]
    src = f"teacher = {teacher}, from teacher.json" if teacher else "no teacher.json: every model other than tuned/base/grpo"
    if teach:
        out.append(f"**Extra rows: teacher ceiling ({src}; seeds 0-1 expected)**\n\n"
                   + md([r for r in model_rows(teach, servers)], ["model", "licence", "method"] + COLS + ["cfg"]))
    else:
        out.append("_Teacher-ceiling rows: none yet._")
    return "\n\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", nargs="+", default=["E2-models", "T2-heldout", "E4-split", "E5-cpu"])
    ap.add_argument("--tuned", default="granite-4.2-3b-sft-v1")
    ap.add_argument("--base", default="granite-4.2-3b-self")
    ap.add_argument("--grpo", help="default: <tuned>-grpo")
    ap.add_argument("--paired-exp", default="T2-heldout")
    a = ap.parse_args()
    grpo = a.grpo or a.tuned + "-grpo"
    exps = list(a.exp)
    for e in exps:
        print(f"## {e}\n")
        print(paired_section(e, a.tuned, a.base, grpo) if e == a.paired_exp
              else tables(e, by_tier=(e == "E2-models")))
        print()


if __name__ == "__main__":
    main()
