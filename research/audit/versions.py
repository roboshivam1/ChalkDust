"""Score the audited specs with three checker versions against the graders' consensus.

Run from the repo root, after extracting the two older checkers:
    git show 20eb828^:research/content.py > <dir>/content_pre.py
    git show 20eb828:research/content.py  > <dir>/content_mut.py
    .lab/venv-harness/Scripts/python.exe research/audit/versions.py research/audit <dir>
"""
import importlib.util, json, os, sys
sys.path.insert(0, os.getcwd())
from research.topics import BY_ID
A = sys.argv[1]; D = sys.argv[2]
def mod(name, path):
    s = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
from research import content as NEW
V = [("pre-audit (FINDINGS, before 20eb828)", mod("c_pre", D + "/content_pre.py")),
     ("mutation fix 20eb828", mod("c_mut", D + "/content_mut.py")), ("audit fix 1a0b4f0", NEW)]
specs, lab = {}, {"A": {}, "B": {}}
for n in (1, 2, 3):
    for r in json.load(open(f"{A}/slice{n}.json", encoding="utf-8")):
        specs[r["id"]] = (r["topic"], r["spec"])
    for g in "AB":
        for l in open(f"{A}/labels_slice{n}_{g}.jsonl", encoding="utf-8"):
            if l.strip():
                d = json.loads(l); lab[g][d["id"]] = d
ok = lambda s: s == "correct"
for name, C in V:
    tab = {"shown": [0, 0, 0, 0], "spoken": [0, 0, 0, 0]}   # TP FP FN TN vs consensus correct
    for i, (tid, sp) in specs.items():
        a, b = lab["A"].get(i), lab["B"].get(i)
        if not a or not b:
            continue
        t = BY_ID[tid]; beats = json.load(open(sp, encoding="utf-8"))["beats"]
        ca = C.check_answer(t.key, C.answer_candidates(beats), C.final_narration(beats), t.request)["status"]
        cs = C.check_spoken(t.key, C.final_narration(beats), t.request)["status"]
        for f, c in (("shown", ca), ("spoken", cs)):
            if a[f] != b[f] or c == "n/a":
                continue
            g = ok(a[f]); k = ok(c)
            tab[f][0 if (k and g) else 1 if (k and not g) else 2 if (g and not k) else 3] += 1
    for f, (tp, fp, fn, tn) in tab.items():
        print(f"{name:38s} {f:6s} agree {tp+tn}/{tp+fp+fn+tn}  false pass {fp}  false fail {fn}")
