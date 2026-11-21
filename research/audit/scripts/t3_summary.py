"""Write T3-audit labels from the workflow output and compare the checker with the
graders' consensus for the GRPO arm.

    python research/audit/scripts/t3_summary.py <workflow output file> <private map>
"""
import json
import sys
from collections import Counter

out_file, map_file = sys.argv[1:3]
raw = open(out_file, encoding="utf-8").read()
doc = json.loads(raw[raw.index("{"):])
res = doc.get("result", doc)
priv = json.load(open(map_file, encoding="utf-8"))
D = "research/audit/t3-heldout"


def consensus(ab, adj, field):
    a, b = (ab.get("A") or {}).get(field), (ab.get("B") or {}).get(field)
    if a is not None and a == b:
        return a, "agree"
    for v in adj:
        if v.get("field") == field:
            return v["verdict"], "adjudicated"
    return None, "unresolved"


rows = []
with open(f"{D}/adjudicated.jsonl", "w", encoding="utf-8") as fa:
    for sl in res["t3"]:
        n = sl["slice"]
        for who in ("A", "B"):
            with open(f"{D}/labels_slice{n}_{who}.jsonl", "w", encoding="utf-8") as f:
                for vid, ab in sorted(sl["byId"].items()):
                    if ab.get(who):
                        f.write(json.dumps(ab[who], ensure_ascii=False) + "\n")
        for a in sl["adjudicated"]:
            fa.write(json.dumps({"slice": n, **a}, ensure_ascii=False) + "\n")
        for vid, ab in sl["byId"].items():
            adj = [v for v in sl["adjudicated"] if v.get("id") == vid]
            sh, shh = consensus(ab, adj, "shown")
            sp, sph = consensus(ab, adj, "spoken")
            ws = set((ab.get("A") or {}).get("wrong_steps", [])) & set((ab.get("B") or {}).get("wrong_steps", []))
            rows.append({"id": vid, **priv[vid], "graders_shown": sh, "shown_how": shh, "graders_spoken": sp,
                         "spoken_how": sph, "wrong_steps_both": sorted(ws)})
with open(f"{D}/consensus.jsonl", "w", encoding="utf-8") as f:
    for r in sorted(rows, key=lambda r: r["id"]):
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print("T3 audit:", len(rows), "videos")
for field, ck in (("shown", "checker_shown"), ("spoken", "checker_spoken")):
    print(f"  {field:6} checker->graders:", dict(sorted(Counter((r[ck], r['graders_' + field]) for r in rows).items(), key=str)))
ctrl = [r for r in rows if r["control"]]
print(f"  controls {len(ctrl)}: shown graders-correct {sum(r['graders_shown'] == 'correct' for r in ctrl)}")
print(f"  videos with a wrong step flagged by both graders: {sum(bool(r['wrong_steps_both']) for r in rows)}/{len(rows)}"
      f" (controls {sum(bool(r['wrong_steps_both']) for r in ctrl)}/{len(ctrl)})")
