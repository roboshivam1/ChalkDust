"""Turn a teacher run into an SFT dataset, keeping only what passed.

    python -m research.build_sft --exp T1-teacher --out .lab/datasets/sft-v1

Filters, all hard:
  video   the topic's answer (shown AND spoken) is correct, or the topic has
          no key; otherwise every example from that video is dropped
  plan    kept only if the video passed and >= 80% of its beats are clean
  beat    a sample is kept only if it is clean (validator rungs 1-3 pass and
          no quality flag) and needed no LaTeX-escape repair; at most 2
          distinct clean samples per beat, so the set is not dominated by
          near-copies

Each example is {"prompt": [system, user], "completion": [assistant]} with
the exact prompts the student will see at inference, plus metadata. The
validation split holds out whole families, so val loss is not measured on
near-twins of training rows.

Only the TEACHER's completions become targets. Prompts were written by
Claude (inputs); see research/generate.py.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from research.topics import topic_set

REPO = Path(__file__).resolve().parents[1]


def load_records(exp: str) -> list[dict]:
    p = REPO / "research" / "results" / f"{exp}.jsonl"
    recs = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [r for r in recs if r.get("record") != "server"]


def _is_plan(text: str) -> bool:
    """Mirror make_plan's filter: a dict with at least one allowed beat."""
    from research.catalog import ALLOWED
    from research.llm import parse_json
    v, _ = parse_json(text)
    return (isinstance(v, dict) and isinstance(v.get("beats"), list)
            and any(isinstance(b, dict) and b.get("component") in ALLOWED for b in v["beats"]))


def video_ok(rec: dict, require_spoken: bool = True) -> bool:
    """The answer shown is right and (for a numeric key) the narration says
    it. Re-scored from the saved spec with the current checkers, so a record
    written before a checker fix is judged like every other."""
    from research.analyze import rescore
    rec = {**rec, **rescore(rec)}
    a, s = rec["answer"].get("status"), rec.get("answer_spoken", {}).get("status")
    if a == "n/a":
        return True
    if a != "correct":
        return False
    return s in ("correct", "n/a") or not require_spoken


def _sample_ok(text: str, component: str | None, topic) -> tuple[bool, str]:
    """Re-check one clean sample with the CURRENT checks: quality flags (the
    stored ones may predate a flag fix), and, for an AnswerBox, the answer
    itself -- every kept sample, not only the one the video selected."""
    from research.content import check_answer
    from research.llm import parse_json
    from research.score import quality_flags
    v, _ = parse_json(text)
    if not isinstance(v, dict):
        return False, "unparsable"
    beat = {"component": component, "narration": v.get("narration", ""), "params": v.get("params", {})}
    if quality_flags(beat):
        return False, "flags"
    if component == "AnswerBox" and topic is not None and topic.key is not None:
        st = check_answer(topic.key, [str((v.get("params") or {}).get("value") or "")])["status"]
        if st != "correct":
            return False, f"answer_{st}"
    return True, ""


def build(exp: str, out: Path, val_families: int = 4, per_beat: int = 2, seed: int = 0,
          require_spoken: bool = True, prompt_hash: str | None = None, allow_non_train: bool = False) -> dict:
    from research.generate import PROMPT_HASH
    train = {t.id: t for t in topic_set("train")}
    prompt_hash = prompt_hash or PROMPT_HASH
    recs = load_records(exp)
    n_all = len(recs)
    # an SFT set must come from one prompt version: the one the student will be served
    if prompt_hash != "any":    # 'any' is for dry runs on traces older than the hash
        recs = [r for r in recs if r.get("prompt_hash") == prompt_hash]
    stray = sorted({r["topic"] for r in recs if r["topic"] not in train})
    if stray and not allow_non_train:
        raise SystemExit(f"topics outside the training pool: {stray[:5]} (use --allow-non-train for a dry run)")
    rows, stats = [], Counter({"records": n_all, "records_with_hash": len(recs)})
    for rec in recs:
        trace_p = REPO / rec["trace_dir"] / "trace.json"
        if not trace_p.exists():
            stats["missing_trace"] += 1
            continue
        tr = json.loads(trace_p.read_text(encoding="utf-8"))
        stats["videos"] += 1
        if not video_ok(rec, require_spoken):
            stats["videos_dropped_answer"] += 1
            continue
        from research.analyze import rescore
        topic = train.get(rec["topic"])
        fam = topic.family if topic else rec["topic"]
        meta = {"topic": rec["topic"], "family": fam, "teacher": rec["model"], "prompt_hash": prompt_hash}
        clean_now = rescore(rec).get("beats_clean", rec["beats_clean"])
        # plan
        if rec["n_beats"] and clean_now / rec["n_beats"] >= 0.8:
            for call in tr["calls"]:
                if call["step"] != "plan":
                    continue
                # vote()'s 'chosen' indexes the plans that parsed, not the raw
                # completions: map it back.
                chosen = (call.get("vote") or {}).get("chosen", 0) or 0
                parsed = [i for i, c in enumerate(call["completions"]) if c["ok"] and _is_plan(c["text"])]
                if chosen < len(parsed):
                    text = call["completions"][parsed[chosen]]["text"]
                    rows.append({"prompt": call["messages"],
                                 "completion": [{"role": "assistant", "content": text}],
                                 "type": "plan", **meta})
                    stats["plan_examples"] += 1
        # beats: at most per_beat distinct samples per BEAT, across its repair rounds
        kept, seen = Counter(), set()
        for call in tr["calls"]:
            if not call["step"].startswith("beat"):
                continue
            beat_key = call["step"].split(".r")[0]
            for c, sc in zip(call["completions"], call.get("scores", [])):
                stats["beat_samples"] += 1
                if not (sc.get("clean") and sc.get("escape_repairs", 0) == 0 and c["ok"]):
                    continue
                key = (beat_key, c["text"].strip())
                if key in seen or kept[beat_key] >= per_beat:
                    continue
                ok, why = _sample_ok(c["text"], sc.get("component"), topic)
                if not ok:
                    stats[f"beat_dropped_{why}"] += 1
                    continue
                seen.add(key)
                kept[beat_key] += 1
                rows.append({"prompt": call["messages"],
                             "completion": [{"role": "assistant", "content": c["text"]}],
                             "type": "beat_repair" if ".r0" not in call["step"] else "beat",
                             "component": sc.get("component"), **meta})
                stats["beat_examples"] += 1
                stats[f"component:{sc.get('component')}"] += 1
    fams = sorted({r["family"] for r in rows})
    rng = random.Random(seed)
    val_f = set(rng.sample(fams, min(val_families, max(len(fams) - 1, 0)))) if fams else set()
    out.mkdir(parents=True, exist_ok=True)
    tr_rows = [r for r in rows if r["family"] not in val_f]
    va_rows = [r for r in rows if r["family"] in val_f]
    for name, rs in (("train", tr_rows), ("val", va_rows)):
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {"exp": exp, "prompt_hash": prompt_hash, "train": len(tr_rows), "val": len(va_rows),
               "val_families": sorted(val_f), **dict(stats)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-beat", type=int, default=2)
    ap.add_argument("--val-families", type=int, default=4)
    ap.add_argument("--no-require-spoken", action="store_true")
    ap.add_argument("--prompt-hash", default=None, help="default: the current research.generate.PROMPT_HASH")
    ap.add_argument("--allow-non-train", action="store_true", help="dry runs on DEV traces only")
    a = ap.parse_args()
    print(json.dumps(build(a.exp, Path(a.out), val_families=a.val_families, per_beat=a.per_beat,
                           require_spoken=not a.no_require_spoken, prompt_hash=a.prompt_hash,
                           allow_non_train=a.allow_non_train), indent=2))
