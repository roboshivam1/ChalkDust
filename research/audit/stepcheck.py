import glob, io, json, os, sys
from collections import Counter
sys.path.insert(0, ".")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from research import content as C, topics as T
from research.build_sft import video_ok

AUD = "research/audit"
TOPICS = {}
for n in ("dev", "heldout", "train"):
    try:
        for t in T.topic_set(n):
            TOPICS.setdefault(t.id, t)
    except Exception:
        pass
specs = {}
for n in (1, 2, 3):
    for r in json.load(open(f"{AUD}/slice{n}.json", encoding="utf-8")):
        specs[r["id"]] = r
lab = {"A": {}, "B": {}}
for n in (1, 2, 3):
    for g in "AB":
        p = f"{AUD}/labels_slice{n}_{g}.jsonl"
        if os.path.exists(p):
            for l in open(p, encoding="utf-8"):
                if l.strip():
                    d = json.loads(l); lab[g][d["id"]] = d
ids = [i for i in specs if i in lab["A"] and i in lab["B"]]
print("specs", len(specs), "labelled by both", len(ids))

recs = {}
for f in glob.glob("research/results/*.jsonl"):
    for l in open(f, encoding="utf-8"):
        d = json.loads(l)
        if d.get("record") != "server" and d.get("trace_dir"):
            recs[os.path.normcase(os.path.normpath(os.path.abspath(d["trace_dir"])))] = d

rows = []
for i in ids:
    sp = specs[i]
    beats = json.load(open(sp["spec"], encoding="utf-8"))["beats"]
    t = TOPICS[sp["topic"]]
    dc = C.derivation_consistency(beats, t.key)
    ws = {g: lab[g][i].get("wrong_steps") or [] for g in "AB"}
    key = os.path.normcase(os.path.normpath(os.path.dirname(sp["spec"])))
    rec = recs.get(key)
    if rec is not None:
        ok = video_ok(rec); how = "video_ok"
    else:
        a = C.check_answer(t.key, C.answer_candidates(beats), C.final_narration(beats), t.request)["status"]
        s = C.check_spoken(t.key, C.final_narration(beats), t.request)["status"]
        ok = (a == "n/a") or (a == "correct" and s in ("correct", "n/a")); how = "approx"
    rows.append(dict(id=i, topic=sp["topic"], dc=dc, ws=ws, ok=ok, how=how, beats=beats))

print("video_ok source:", Counter(r["how"] for r in rows))
print("derivation_consistency applicable:", Counter(r["dc"]["applicable"] for r in rows), "topics:",
      Counter(r["topic"] for r in rows if r["dc"]["applicable"]))
print("applicable bad>0:", sum(1 for r in rows if r["dc"].get("inconsistent", 0) > 0))

def matrix(name, pos, rs):
    tp = sum(1 for r in rs if pos(r) and r["dc"].get("inconsistent", 0) > 0)
    fn = sum(1 for r in rs if pos(r) and not r["dc"].get("inconsistent", 0) > 0)
    fp = sum(1 for r in rs if not pos(r) and r["dc"].get("inconsistent", 0) > 0)
    tn = len(rs) - tp - fn - fp
    pr = tp / (tp + fp) if tp + fp else float("nan"); rc = tp / (tp + fn) if tp + fn else float("nan")
    print(f"{name}: n={len(rs)} TP={tp} FN={fn} FP={fp} TN={tn} precision={pr:.2f} recall={rc:.2f}")

both = lambda r: bool(r["ws"]["A"]) and bool(r["ws"]["B"])
either = lambda r: bool(r["ws"]["A"]) or bool(r["ws"]["B"])
print()
for nm, rs in (("ALL specs", rows), ("applicable only (set key)", [r for r in rows if r["dc"]["applicable"]])):
    matrix(nm + " | both graders", both, rs)
    matrix(nm + " | either grader", either, rs)

print("\nGrader positive (either) counts by topic:")
for tp_, c in sorted(Counter(r["topic"] for r in rows if either(r)).items()):
    print(" ", tp_, c, "of", sum(1 for r in rows if r["topic"] == tp_), "(applicable:", T and TOPICS[tp_].key.kind == "set", ")")
print("grader positive: both", sum(both(r) for r in rows), "either", sum(either(r) for r in rows))

print("\n== applicable specs: detail ==")
for r in rows:
    if not r["dc"]["applicable"]:
        continue
    flag = r["dc"]["inconsistent"] > 0
    if flag or either(r):
        print(f"\n{r['id']}\n  dc={r['dc']} grader A={bool(r['ws']['A'])} B={bool(r['ws']['B'])}")
        print("  A wrong_steps:", r["ws"]["A"]); print("  B wrong_steps:", r["ws"]["B"])
        for b in r["beats"]:
            p = b.get("params", {})
            lines = p.get("steps", []) if b.get("component") == "EquationDerivation" else (
                p.get("work", []) if b.get("component") == "SolutionStep" and p.get("work_format", "math") == "math" else [])
            for ln in lines:
                parts = C.equation_parts(__import__("re").sub(r"\{\{|\}\}", "", str(ln)))
                print("   line:", ln, "| parts:", parts)

print("\n== SFT filter question ==")
okrows = [r for r in rows if r["ok"]]
print("video_ok True:", len(okrows), "of", len(rows))
for nm, pos in (("both", both), ("either", either)):
    hit = [r for r in okrows if pos(r)]
    caught = [r for r in hit if r["dc"].get("inconsistent", 0) > 0]
    print(f"video_ok & grader-wrong_steps ({nm}): {len(hit)}; of which derivation_consistency would flag: {len(caught)}")
rej = [r for r in okrows if r["dc"].get("inconsistent", 0) > 0]
print("video_ok specs the step filter would reject:", len(rej), " of which grader-clean (false rejects):",
      sum(1 for r in rej if not either(r)))
print("\nvideo_ok with grader wrong_steps (either):")
for r in okrows:
    if either(r):
        print(f"- {r['id']} [{r['topic']}] dc_applicable={r['dc']['applicable']}")
        for g in "AB":
            print(f"    {g}: {r['ws'][g]}")
