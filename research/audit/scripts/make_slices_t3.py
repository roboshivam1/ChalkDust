"""Blind slices for the T3 audit: every GRPO held-out video the frozen checker did not
pass (shown or spoken), plus one checker-passed control per topic. Ids are anonymised;
the id -> run map goes to a private file until grading is done.

    python research/audit/scripts/make_slices_t3.py research/audit/t3-heldout <private_map.json>
"""
import json
import os
import random
import sys

sys.path.insert(0, os.getcwd())
import research.analyze as an                      # noqa: E402
from research.topics import BY_ID                  # noqa: E402

out, mapf = sys.argv[1:3]
os.makedirs(out, exist_ok=True)
recs, _ = an.load("T2-heldout")
recs = [r for r in recs if r["model"] == "granite-4.2-3b-sft-v1-grpo"]
bad = [r for r in recs if r["answer"]["status"] not in ("correct", "n/a")
       or r["answer_spoken"]["status"] not in ("correct", "n/a")]
good = [r for r in recs if r not in bad]
rng = random.Random(20261122)
ctrl = []
for t in sorted({r["topic"] for r in good}):
    ctrl += rng.sample([r for r in good if r["topic"] == t], 1)
pick = bad + ctrl
rng.shuffle(pick)
items, private, topics = [], {}, {}
for i, r in enumerate(pick):
    vid = f"T3-{i + 1:03d}"
    items.append({"id": vid, "spec": str((an.REPO / r["trace_dir"] / "spec.json").resolve()), "topic": r["topic"]})
    private[vid] = {"model": r["model"], "method": r["method"], "seed": r["seed"], "topic": r["topic"],
                    "checker_shown": r["answer"]["status"], "checker_spoken": r["answer_spoken"]["status"],
                    "control": r in ctrl, "trace_dir": r["trace_dir"]}
    t = BY_ID[r["topic"]]
    topics[r["topic"]] = {"set": "heldout", "request": t.request, "key_kind": t.key.kind if t.key else "none",
                          "key_value": str(t.key.value) if t.key else "", "units": t.key.units if t.key else None,
                          "rel_tol": t.key.rel_tol if t.key else None}
k = max(1, round(len(items) / 8))
for i in range(k):
    json.dump(items[i::k], open(f"{out}/slice{i + 1}.json", "w", encoding="utf-8"), indent=1)
json.dump(topics, open(f"{out}/topics.json", "w", encoding="utf-8"), indent=1)
json.dump(private, open(mapf, "w", encoding="utf-8"), indent=1)
print(f"T3: {len(recs)} GRPO videos, {len(bad)} not-correct + {len(ctrl)} controls -> {len(items)} in {k} slices")
