"""Pick and render videos for people to watch.

    python -m research.watchlist --exp E2-models --model gemma-4-12b \
        --method staged:n=4,repair=1,calc=1,work=1 --n 3 --quality draft --tag teacher

Picks the run's best videos by hard checks only -- every beat clean (re-flagged
with the current checks), the answer right on screen and (where checkable)
in speech -- one per subject first so the list is not all maths, then
renders each with research.render and appends to .lab/videos/WATCHLIST.jsonl
(path, topic, model, method, the checks it passed, render time). With
--worst it picks the opposite (valid but wrong answer) as a cautionary
example: what the validator alone lets through.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from research.analyze import load
from research.render import OUT, render


def _good(r: dict) -> bool:
    a, s = r["answer"]["status"], r["answer_spoken"]["status"]
    return bool(r.get("all_beats_clean")) and a in ("correct", "n/a") and s in ("correct", "n/a")


def _cautionary(r: dict) -> bool:
    return r.get("beats_valid") == r.get("n_beats") and r["answer"]["status"] == "wrong"


def pick(exp: str, model: str, method: str, n: int, worst: bool = False) -> list[dict]:
    recs, _ = load(exp)
    rs = [r for r in recs if r["model"] == model and r["method"] == method]
    rs = [r for r in rs if (_cautionary(r) if worst else _good(r))]
    rs.sort(key=lambda r: (-r["structure"]["score"], r["wall_seconds"]))
    out, subjects = [], set()
    for r in rs:                      # one per subject first
        if r["subject"] not in subjects:
            out.append(r)
            subjects.add(r["subject"])
    out += [r for r in rs if r not in out]
    return out[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--quality", default="draft", choices=["draft", "final"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--worst", action="store_true")
    a = ap.parse_args()
    for r in pick(a.exp, a.model, a.method, a.n, a.worst):
        name = f"{a.tag + '-' if a.tag else ''}{r['topic']}-{r['model']}-s{r['seed']}"
        t0 = time.time()
        res = render(Path(r["trace_dir"]), name, a.quality)
        entry = {"name": name, "mp4": res.get("mp4"), "rc": res["rc"], "contact": res.get("contact"),
                 "duration_s": res.get("duration_s"), "render_seconds": time.time() - t0,
                 "exp": a.exp, "model": r["model"], "method": r["method"], "topic": r["topic"], "seed": r["seed"],
                 "quality": a.quality, "cautionary": a.worst,
                 "checks": {"all_beats_clean": r.get("all_beats_clean"), "answer": r["answer"],
                            "answer_spoken": r["answer_spoken"], "structure": r["structure"]["score"]}}
        with open(OUT / "WATCHLIST.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        print(json.dumps({k: entry[k] for k in ("name", "rc", "duration_s", "render_seconds")}), flush=True)


if __name__ == "__main__":
    main()
