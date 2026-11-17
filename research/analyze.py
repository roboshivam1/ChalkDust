"""Summarise experiment results, re-scoring content with the current checkers.

    python -m research.analyze E1-methods [--by model,method] [--md out.md]

The raw records in research/results/*.jsonl are never edited. Content
metrics (answer shown, answer spoken, structure) and beat cleanliness
(validator level from the trace + the current quality flags) are recomputed
here from each run's saved spec.json + trace.json, so every row of a table
is scored by the same checker version even when a checker was fixed after
the run. Validator levels, speed and memory come from the record as
written. A record whose artifacts are missing is marked rescored=False.

Columns (all per video, averaged over the group):
  n            videos in the group
  valid        share of beats that pass validator rungs 1-3
  clean        share of beats valid AND free of quality flags (LaTeX read
               aloud, LaTeX drawn literally in prose, stray $)
  allclean     share of videos where every beat is clean
  ans          share of keyed videos whose shown answer is correct (an answer
               the checker cannot read counts as not correct: unverified)
  unp          share of keyed videos whose answer could not be read at all
  spk          share of keyed numeric videos whose narration's final stated
               value is the answer (n/a when the key is in the request)
  both         share of keyed videos correct on screen AND (spoken or n/a)
  struct       mean lesson-structure score (8 checks, research/content.py)
  llm_s        mean LLM seconds per video (max over concurrent samples)
  val_s        mean validator seconds per video
  wall_s       mean wall seconds per video
  tok/s        mean generation tokens/s per request
  vram         model VRAM (peak minus pre-load baseline, MiB) from the
               server record
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from research.content import answer_candidates, check_answer, check_spoken, final_narration, structure_checks
from research.score import quality_flags
from research.topics import BY_ID, topic_set

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "research" / "results"


def _topic(tid: str):
    if tid in BY_ID:
        return BY_ID[tid]
    for t in topic_set("train"):
        if t.id == tid:
            return t
    return None


def rescore(rec: dict) -> dict:
    """Content metrics from the saved artifacts, with the current checkers."""
    tdir = REPO / rec["trace_dir"]
    topic = _topic(rec["topic"])
    try:
        spec = json.loads((tdir / "spec.json").read_text(encoding="utf-8"))
        tr = json.loads((tdir / "trace.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"rescored": False}
    beats = spec.get("beats", [])
    key = topic.key if topic else None
    req = topic.request if topic else ""
    ans = check_answer(key, answer_candidates(beats), final_narration(beats), req) if key else {"status": "n/a"}
    spk = check_spoken(key, final_narration(beats), req) if (key and beats) else {"status": "n/a"}
    struct = structure_checks(tr.get("plan"), beats, key)
    out = {"answer": ans, "answer_spoken": spk, "structure": struct, "rescored": True}
    scores = tr.get("scores", [])
    if beats and len(scores) == len(beats):
        clean = [s.get("level") == 3 and not quality_flags(b) for s, b in zip(scores, beats)]
        out.update({"beats_clean_stored": rec.get("beats_clean"), "beats_clean": sum(clean),
                    "all_beats_clean": all(clean)})
    return out


def load(exp: str) -> tuple[list[dict], list[dict]]:
    p = RESULTS / f"{exp}.jsonl"
    recs, servers = [], []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        (servers if d.get("record") == "server" else recs).append(d)
    for r in recs:
        r.update(rescore(r))
    return recs, servers


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def table(recs: list[dict], servers: list[dict], by: list[str]) -> list[dict]:
    groups = defaultdict(list)
    for r in recs:
        groups[tuple(r.get(k) for k in by)].append(r)
    vram = defaultdict(list)
    for s in servers:
        vram[s["model"]].append(s.get("vram_model_mib"))
    rows = []
    for g, rs in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        keyed = [r for r in rs if r["answer"]["status"] != "n/a"]
        spk_app = [r for r in rs if r["answer_spoken"]["status"] in ("correct", "wrong", "not_found")]
        row = dict(zip(by, g))
        row.update({
            "n": len(rs),
            "valid": _mean([r["beats_valid"] / r["n_beats"] if r["n_beats"] else 0 for r in rs]),
            "clean": _mean([r.get("beats_clean", 0) / r["n_beats"] if r["n_beats"] else 0 for r in rs]),
            "allclean": _mean([1.0 if r.get("all_beats_clean") else 0.0 for r in rs]),
            "ans": _mean([1.0 if r["answer"]["status"] == "correct" else 0.0 for r in keyed]),
            "unp": _mean([1.0 if r["answer"]["status"] == "unparsed" else 0.0 for r in keyed]),
            "spk": _mean([1.0 if r["answer_spoken"]["status"] == "correct" else 0.0 for r in spk_app]),
            "both": _mean([1.0 if (r["answer"]["status"] == "correct"
                                   and r["answer_spoken"]["status"] in ("correct", "n/a")) else 0.0
                           for r in keyed]),
            "struct": _mean([r["structure"]["score"] for r in rs]),
            "llm_s": _mean([r["llm_seconds"] for r in rs]),
            "val_s": _mean([r["validator_seconds"] for r in rs]),
            "wall_s": _mean([r["wall_seconds"] for r in rs]),
            "tok/s": _mean([r["gen_tok_s_mean"] for r in rs]),
            "vram": _mean(vram.get(rs[0]["model"], [])) if "model" in by else None,
        })
        rows.append(row)
    return rows


def to_markdown(rows: list[dict]) -> str:
    if not rows:
        return "(no rows)"
    cols = list(rows[0].keys())
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append(f"{v:.0f}" if c in ("llm_s", "val_s", "wall_s", "tok/s", "vram") else f"{v:.2f}")
            else:
                cells.append("" if v is None else str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", nargs="+")
    ap.add_argument("--by", default="model,method")
    ap.add_argument("--md")
    a = ap.parse_args()
    recs, servers = [], []
    for e in a.exp:
        r, s = load(e)
        recs += r
        servers += s
    md = to_markdown(table(recs, servers, a.by.split(",")))
    print(md)
    if a.md:
        Path(a.md).write_text(md + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
