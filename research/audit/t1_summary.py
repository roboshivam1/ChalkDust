"""Summarise the blind grading of T1 teacher videos (the SFT-eligible ones).

    python research/audit/t1_summary.py research/audit/t1-wave1

Reads labels_slice*_{A,B}.jsonl and adjudicated.jsonl in that folder. Reports:
- grader agreement on the shown and spoken answer;
- the current checkers against the graders' consensus, where both graders agree;
- how many videos show a false line that an adjudicator confirmed teaches an error.

The wrong-step count is a MEASUREMENT of the SFT data, not a filter: hard checks
decide what is kept (FINDINGS item 9)."""

import glob
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.getcwd())
from research.content import answer_candidates, check_answer, check_spoken, final_narration  # noqa: E402
from research.train_topics import TRAIN  # noqa: E402

d = sys.argv[1]
labels = {}
for f in sorted(glob.glob(os.path.join(d, "labels_slice*_*.jsonl"))):
    who = f.rsplit("_", 1)[-1][0]
    for line in open(f, encoding="utf-8"):
        if line.strip():
            lab = json.loads(line)
            labels.setdefault(lab["id"], {})[who] = lab
adj = {}
for line in open(os.path.join(d, "adjudicated.jsonl"), encoding="utf-8"):
    if line.strip():
        a = json.loads(line)
        adj[a["id"]] = a["steps"]
specs = {}
for f in glob.glob(os.path.join(d, "slice*.json")):
    for it in json.load(open(f, encoding="utf-8")):
        specs[it["id"]] = it
topics = {t.id: t for t in TRAIN}

agree, cmp, n = Counter(), Counter(), 0
for vid, ab in sorted(labels.items()):
    if "A" not in ab or "B" not in ab:
        continue
    n += 1
    t = topics[specs[vid]["topic"]]
    beats = json.load(open(specs[vid]["spec"], encoding="utf-8"))["beats"]
    for side in ("shown", "spoken"):
        a, b = ab["A"][side], ab["B"][side]
        agree[(side, a == b)] += 1
        if a != b or a == "n/a" or t.key is None:
            continue
        if side == "shown":
            got = check_answer(t.key, answer_candidates(beats), final_narration(beats), t.request)["status"]
        else:
            got = check_spoken(t.key, final_narration(beats), t.request)["status"]
        if got == "n/a":
            continue
        truth_ok, chk_ok = a == "correct", got == "correct"
        cmp[(side, "false_pass" if chk_ok and not truth_ok else "false_fail" if truth_ok and not chk_ok
             else "agree")] += 1

flag = {vid for vid, ab in labels.items() if any(ab.get(w, {}).get("wrong_steps") for w in "AB")}
both = {vid for vid, ab in labels.items() if all(ab.get(w, {}).get("wrong_steps") for w in "AB")}
teach = {vid for vid, st in adj.items() if any(s["false_on_screen"] and s["severity"] == "teaches_error" for s in st)}
slip = {vid for vid, st in adj.items() if any(s["false_on_screen"] for s in st)} - teach
print(f"videos double-graded: {n}")
for side in ("shown", "spoken"):
    print(f"  {side}: graders agree {agree[(side, True)]}/{agree[(side, True)] + agree[(side, False)]}; "
          f"checker vs consensus: agree {cmp[(side, 'agree')]}, false pass {cmp[(side, 'false_pass')]}, "
          f"false fail {cmp[(side, 'false_fail')]}")
print(f"wrong steps flagged by either grader: {len(flag)}/{n}; by both: {len(both)}/{n}")
print(f"adjudicated: teaches an error {len(teach)}/{n}; harmless slip only {len(slip)}/{n}")
for vid in sorted(teach):
    for s in adj[vid]:
        if s["false_on_screen"] and s["severity"] == "teaches_error":
            print(f"  - {vid.split('|')[2]}: {s['line'][:70]!r}")
