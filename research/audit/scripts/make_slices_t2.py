"""Blind slices for (a) the T2 audit: every granite T2 video the checker did not mark
correct (shown or spoken), plus up to 2 checker-correct controls per (topic, arm); ids
are anonymised, the id -> model map goes to a private file; (b) T1 wave 2: SFT-eligible
teacher videos not graded in wave 1.
Usage (repo root): python make_slices_t2.py <t2_outdir> <t1_outdir> <private_map.json>"""
import json, os, random, sys
sys.path.insert(0, os.getcwd())
from research.build_sft import video_ok
from research.topics import BY_ID
from research.train_topics import TRAIN
import research.analyze as an

t2dir, t1dir, mapf = sys.argv[1:4]
os.makedirs(t2dir, exist_ok=True); os.makedirs(t1dir, exist_ok=True)


def topic_entry(t, set_):
    return {"set": set_, "request": t.request, "key_kind": t.key.kind if t.key else "none",
            "key_value": str(t.key.value) if t.key else "", "units": t.key.units if t.key else None,
            "rel_tol": t.key.rel_tol if t.key else None}


# --- T2 -----------------------------------------------------------------------------
recs, _ = an.load("T2-heldout")          # re-scored with the frozen checkers
recs = [r for r in recs if r["model"] in ("granite-4.2-3b-sft-v1", "granite-4.2-3b-self")]
rng = random.Random(20261121)
bad = [r for r in recs if r["answer"]["status"] not in ("correct", "n/a")
       or r["answer_spoken"]["status"] not in ("correct", "n/a")]
good = [r for r in recs if r not in bad]
ctrl = []
for key in sorted({(r["topic"], r["model"]) for r in good}):
    pool = [r for r in good if (r["topic"], r["model"]) == key]
    ctrl += rng.sample(pool, min(2, len(pool)))
pick = bad + ctrl
rng.shuffle(pick)
items, private, topics = [], {}, {}
for i, r in enumerate(pick):
    vid = f"T2-{i + 1:03d}"
    items.append({"id": vid, "spec": str((an.REPO / r["trace_dir"] / "spec.json").resolve()), "topic": r["topic"]})
    private[vid] = {"model": r["model"], "method": r["method"], "seed": r["seed"], "topic": r["topic"],
                    "checker_shown": r["answer"]["status"], "checker_spoken": r["answer_spoken"]["status"],
                    "control": r in ctrl, "trace_dir": r["trace_dir"]}
    topics[r["topic"]] = topic_entry(BY_ID[r["topic"]], "heldout")
k2 = max(1, round(len(items) / 8))
for i in range(k2):
    json.dump(items[i::k2], open(f"{t2dir}/slice{i + 1}.json", "w", encoding="utf-8"), indent=1)
json.dump(topics, open(f"{t2dir}/topics.json", "w", encoding="utf-8"), indent=1)
json.dump(private, open(mapf, "w", encoding="utf-8"), indent=1)
print(f"T2: {len(recs)} granite videos, {len(bad)} not-correct + {len(ctrl)} controls -> {len(items)} in {k2} slices")

# --- T1 wave 2 ----------------------------------------------------------------------
by = {t.id: t for t in TRAIN}
seen = set()
for f in os.listdir("research/audit/t1-wave1"):
    if f.startswith("slice"):
        seen |= {x["id"] for x in json.load(open(f"research/audit/t1-wave1/{f}", encoding="utf-8"))}
t1 = [json.loads(l) for l in open("research/results/T1-teacher.jsonl", encoding="utf-8") if l.strip()]
t1 = [r for r in t1 if r.get("record") != "server" and r.get("topic") in by]
ok = [r for r in t1 if video_ok(r)]
items1, topics1 = [], {}
for r in ok:
    vid = f"T1|{r['model']}|{r['topic']}|s{r.get('seed', 0)}"
    spec = os.path.join(r["trace_dir"], "spec.json")
    if vid in seen or not os.path.exists(spec):
        continue
    items1.append({"id": vid, "spec": os.path.abspath(spec), "topic": r["topic"]})
    topics1[r["topic"]] = topic_entry(by[r["topic"]], "train")
k1 = max(1, round(len(items1) / 7))
for i in range(k1):
    json.dump(items1[i::k1], open(f"{t1dir}/slice{i + 1}.json", "w", encoding="utf-8"), indent=1)
json.dump(topics1, open(f"{t1dir}/topics.json", "w", encoding="utf-8"), indent=1)
print(f"T1: {len(t1)} videos, {len(ok)} pass video_ok, {len(seen)} graded in wave 1, {len(items1)} new in {k1} slices")
