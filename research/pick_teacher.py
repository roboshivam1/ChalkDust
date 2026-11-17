"""Choose the teacher by measured yield: usable SFT videos per GPU-hour.

    python -m research.pick_teacher --exp E2-models --method staged:n=4,repair=1,calc=1,work=1 \
        --models gemma-4-12b qwen3.6-35b-a3b gpt-oss-20b --out .lab/queue/teacher.json

A video is usable when build_sft would keep its plan: the answer passes
(`build_sft.video_ok`, re-scored with the current checkers) and at least 80%
of its beats are clean. Yield per hour = usable / sum(wall_seconds) * 3600,
because the teacher run is time-boxed on one GPU and the filter, not the
teacher, decides what reaches the student. A model needs a usable fraction
of at least `--min-frac` to be eligible, so a fast teacher that is mostly
wrong cannot win on speed alone. Only Apache-2.0 / MIT models are eligible
(their outputs may train a shipped model; docs/research/models-2026-11.md).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.build_sft import load_records, video_ok
from research.models import REGISTRY

OPEN = {"apache-2.0", "mit"}


def score(exp: str, method: str, models: list[str]) -> list[dict]:
    recs = [r for r in load_records(exp) if r.get("method") == method and r.get("model") in models]
    out = []
    for m in models:
        rs = [r for r in recs if r["model"] == m]
        if not rs:
            out.append({"model": m, "n": 0})
            continue
        usable = sum(1 for r in rs if video_ok(r) and r.get("n_beats")
                     and r.get("beats_clean", 0) / r["n_beats"] >= 0.8)
        secs = sum(r.get("wall_seconds", 0) for r in rs)
        out.append({"model": m, "n": len(rs), "usable": usable, "usable_frac": usable / len(rs),
                    "wall_s_mean": secs / len(rs), "usable_per_hour": usable / secs * 3600 if secs else 0,
                    "licence": REGISTRY[m].licence})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--min-frac", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = score(a.exp, a.method, a.models)
    for r in rows:
        print(json.dumps(r))
    ok = [r for r in rows if r.get("n") and r["usable_frac"] >= a.min_frac and r["licence"].lower() in OPEN]
    if not ok:
        raise SystemExit(f"no eligible teacher (usable fraction >= {a.min_frac}, open licence)")
    best = max(ok, key=lambda r: (r["usable_per_hour"], r["usable_frac"]))
    res = {"teacher": best["model"], "rule": "max usable_per_hour with usable_frac >= min_frac",
           "min_frac": a.min_frac, "exp": a.exp, "method": a.method, "candidates": rows}
    Path(a.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("teacher:", best["model"])


if __name__ == "__main__":
    main()
