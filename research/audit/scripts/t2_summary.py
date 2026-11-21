"""Write T2-audit and T1-wave-2 labels from the workflow output, then summarise.
Usage (repo root): python t2_summary.py <workflow output file> <private map>"""
import json, os, re, sys
from collections import Counter

out_file, map_file = sys.argv[1:3]
raw = open(out_file, encoding="utf-8").read()
start = raw.index("{")
doc = json.loads(raw[start:])
res = doc.get("result", doc)
if isinstance(res, str):
    res = json.loads(res)
priv = json.load(open(map_file, encoding="utf-8"))


def consensus(ab, adj, field):
    a, b = (ab.get("A") or {}).get(field), (ab.get("B") or {}).get(field)
    if a is not None and a == b:
        return a, "agree"
    for v in adj:
        if v.get("field") == field:
            return v["verdict"], "adjudicated"
    return None, "unresolved"


def write(dirname, rows_by_slice, kind):
    for sl in rows_by_slice:
        n = sl["slice"]
        for who in ("A", "B"):
            with open(f"{dirname}/labels_slice{n}_{who}.jsonl", "w", encoding="utf-8") as f:
                for vid, ab in sorted(sl["byId"].items()):
                    if ab.get(who):
                        f.write(json.dumps(ab[who], ensure_ascii=False) + "\n")
    with open(f"{dirname}/adjudicated.jsonl", "w", encoding="utf-8") as f:
        for sl in rows_by_slice:
            for a in sl["adjudicated"]:
                f.write(json.dumps({"slice": sl["slice"], **a}, ensure_ascii=False) + "\n")


A = "research/audit"
write(f"{A}/t2-heldout", res["t2"], "t2")
write(f"{A}/t1-wave2", res["t1"], "t1")

# --- T2: checker vs consensus, per arm --------------------------------------------
rows = []
for sl in res["t2"]:
    for vid, ab in sl["byId"].items():
        adj = [v for v in sl["adjudicated"] if v.get("id") == vid]
        p = priv[vid]
        sh, sh_how = consensus(ab, adj, "shown")
        sp, sp_how = consensus(ab, adj, "spoken")
        ws = set((ab.get("A") or {}).get("wrong_steps", [])) & set((ab.get("B") or {}).get("wrong_steps", []))
        rows.append({"id": vid, **p, "graders_shown": sh, "shown_how": sh_how, "graders_spoken": sp,
                     "spoken_how": sp_how, "wrong_steps_both": sorted(ws)})
with open(f"{A}/t2-heldout/consensus.jsonl", "w", encoding="utf-8") as f:
    for r in sorted(rows, key=lambda r: r["id"]):
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print("T2 audit:", len(rows), "videos")
for arm in ("granite-4.2-3b-self", "granite-4.2-3b-sft-v1"):
    rs = [r for r in rows if r["model"] == arm]
    for field, ck in (("shown", "checker_shown"), ("spoken", "checker_spoken")):
        g = "graders_" + field
        c = Counter((r[ck], r[g]) for r in rs)
        print(f"  {arm[-6:]} {field:6} checker->graders:", dict(sorted(c.items(), key=str)))
    ctrl = [r for r in rs if r["control"]]
    print(f"  {arm[-6:]} controls {len(ctrl)}: shown graders-correct {sum(r['graders_shown'] == 'correct' for r in ctrl)}, "
          f"spoken graders-correct-or-n/a {sum(r['graders_spoken'] in ('correct', 'n/a') for r in ctrl)}")
    print(f"  {arm[-6:]} videos with a wrong step flagged by both graders: {sum(bool(r['wrong_steps_both']) for r in rs)}/{len(rs)}")

# --- T1 wave 2 -----------------------------------------------------------------
t1 = []
for sl in res["t1"]:
    for vid, ab in sl["byId"].items():
        adj = [s for s in sl["adjudicated"] if s.get("id") == vid]
        t1.append({"id": vid, "A_shown": (ab.get("A") or {}).get("shown"), "B_shown": (ab.get("B") or {}).get("shown"),
                   "A_spoken": (ab.get("A") or {}).get("spoken"), "B_spoken": (ab.get("B") or {}).get("spoken"),
                   "teaches_error": any(s["false_on_screen"] and s["severity"] == "teaches_error" for s in adj),
                   "slip": any(s["false_on_screen"] and s["severity"] == "typo_or_notation" for s in adj)})
print("T1 wave 2:", len(t1), "videos")
print("  shown, both graders:", Counter((r["A_shown"], r["B_shown"]) for r in t1))
print("  spoken, both graders:", Counter((r["A_spoken"], r["B_spoken"]) for r in t1))
print("  confirmed error-teaching line:", sum(r["teaches_error"] for r in t1), "| slips only:",
      sum(r["slip"] and not r["teaches_error"] for r in t1))
for r in t1:
    if r["teaches_error"] or "wrong" in (r["A_shown"], r["B_shown"], r["A_spoken"], r["B_spoken"]):
        print("   ", r)
