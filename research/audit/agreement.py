"""Grader-vs-checker agreement. Run from repo root:
  .lab/venv-harness/Scripts/python.exe <this file>
Read-only; pure Python + sympy (via research.content). Missing label files are skipped."""
import io, json, os, sys
from collections import Counter, defaultdict

sys.path.insert(0, ".")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from research import content as C
from research import topics as T

HERE = os.path.dirname(os.path.abspath(__file__))
TOPICS = {}
for name in ("dev", "heldout", "train"):
    try:
        for t in T.topic_set(name):
            TOPICS.setdefault(t.id, t)
    except Exception:
        pass

specs = {}   # id -> (topic_id, spec path)
for n in (1, 2, 3):
    p = os.path.join(HERE, f"slice{n}.json")
    if os.path.exists(p):
        for r in json.load(open(p, encoding="utf-8")):
            specs[r["id"]] = (r["topic"], r["spec"])

labels = {"A": {}, "B": {}}
loaded, bad = [], []
for n in (1, 2, 3):
    for g in "AB":
        p = os.path.join(HERE, f"labels_slice{n}_{g}.jsonl")
        if not os.path.exists(p):
            continue
        k = 0
        for ln, line in enumerate(open(p, encoding="utf-8"), 1):
            if not line.strip():
                continue
            try:
                d = json.loads(line)
                labels[g][d["id"]] = d
                k += 1
            except Exception as e:
                bad.append(f"{os.path.basename(p)}:{ln} {e}")
        loaded.append(f"slice{n}_{g}={k}")

print("# Grader vs checker agreement\n")
print("Loaded:", ", ".join(loaded) or "none")
if bad:
    print("\nPARSE PROBLEMS:", *bad, sep="\n- ")
unknown = [i for g in "AB" for i in labels[g] if i not in specs]
if unknown:
    print("\nLabel ids not in any slice file:", *unknown, sep="\n- ")

CATS = ["correct", "wrong", "absent", "ambiguous"]
def norm(v):
    return v if v in CATS else f"?{v}"

def kappa(pairs):
    n = len(pairs)
    if not n:
        return None
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[c] * cb[c] for c in set(ca) | set(cb)) / n / n
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)

both = [i for i in labels["A"] if i in labels["B"]]
print(f"\n## 1. Inter-grader agreement ({len(both)} specs labelled by both)\n")
consensus = {}
for field in ("shown", "spoken"):
    pairs = [(norm(labels["A"][i][field]), norm(labels["B"][i][field])) for i in both]
    if pairs:
        agree = sum(a == b for a, b in pairs)
        k = kappa(pairs)
        ks = "n/a" if k is None else f"{k:.3f}"
        print(f"- **{field}**: {agree}/{len(pairs)} = {100*agree/len(pairs):.1f}% raw, Cohen's kappa {ks}")
for field in ("shown", "spoken"):
    dis = [i for i in both if norm(labels["A"][i][field]) != norm(labels["B"][i][field])]
    print(f"\n### Disagreements on `{field}` ({len(dis)})\n")
    for i in dis:
        a, b = labels["A"][i], labels["B"][i]
        print(f"- `{i}`\n  - A: {a[field]} | {a.get(field + '_value')!r}\n  - B: {b[field]} | {b.get(field + '_value')!r}")
for i in both:
    consensus[i] = {f: (norm(labels["A"][i][f]) if norm(labels["A"][i][f]) == norm(labels["B"][i][f]) else "disputed")
                    for f in ("shown", "spoken")}

print(f"\n## 2. Consensus\n")
for field in ("shown", "spoken"):
    c = Counter(v[field] for v in consensus.values())
    print(f"- {field}: " + ", ".join(f"{k}={v}" for k, v in sorted(c.items())))

# ---- checkers, exactly as research/run.py summarise() does ----
def load_beats(i):
    return json.load(open(specs[i][1], encoding="utf-8")).get("beats", [])

def checkers(i):
    topic = TOPICS[specs[i][0]]
    beats = load_beats(i)
    ans = C.check_answer(topic.key, C.answer_candidates(beats), C.final_narration(beats), topic.request) \
        if topic.key else {"status": "n/a", "detail": ""}
    spk = C.check_spoken(topic.key, C.final_narration(beats), topic.request) \
        if (topic.key and beats) else {"status": "n/a", "detail": ""}
    return ans, spk

def ck(status):
    return None if status == "n/a" else ("correct" if status == "correct" else "not-correct")

def gr(label):
    return "correct" if label == "correct" else "not-correct"

print("\n## 3. Checker against consensus (disputed and n/a excluded)\n")
cases = {"shown": [], "spoken": []}
for field, which in (("shown", 0), ("spoken", 1)):
    mat = Counter()
    excl = Counter()
    for i in both:
        if i not in specs or specs[i][0] not in TOPICS:
            excl["no spec/topic"] += 1
            continue
        c = consensus[i][field]
        ans_spk = checkers(i)
        r = ans_spk[which]
        cs = ck(r["status"])
        if cs is None:
            excl["checker n/a"] += 1
            continue
        if c == "disputed":
            excl["disputed"] += 1
            continue
        g = gr(c)
        mat[(cs, g)] += 1
        if cs != g:
            cases[field].append((i, "FALSE PASS" if cs == "correct" else "FALSE FAIL", r, labels["A"][i], labels["B"][i]))
    print(f"### {field}  (checker `{'check_answer' if which == 0 else 'check_spoken'}`)\n")
    print("| checker \\ graders | correct | not-correct |\n|---|---|---|")
    for cs in ("correct", "not-correct"):
        print(f"| {cs} | {mat[(cs, 'correct')]} | {mat[(cs, 'not-correct')]} |")
    tot = sum(mat.values())
    print(f"\nagree {mat[('correct','correct')] + mat[('not-correct','not-correct')]}/{tot}; excluded: {dict(excl)}\n")

print("## 4. Cases\n")
for field in ("shown", "spoken"):
    for kind in ("FALSE PASS", "FALSE FAIL"):
        rows = [c for c in cases[field] if c[1] == kind]
        print(f"### {field}: {kind} ({len(rows)})\n")
        for i, _, r, a, b in rows:
            print(f"- `{i}`\n  - checker: {r['status']} | {r.get('detail')}\n"
                  f"  - A: {a[field]} | {a.get(field + '_value')!r}\n  - B: {b[field]} | {b.get(field + '_value')!r}")
        print()

print("## 5. wrong_steps / unspeakable per model/method\n")
agg = defaultdict(lambda: [0, 0, 0, 0, 0])   # n specs, specs with wrong_steps, total wrong steps, specs unspeakable, total unspeakable
for g in "AB":
    for i, d in labels[g].items():
        parts = i.split("|")
        key = (g, parts[1], parts[2]) if len(parts) >= 5 else (g, "?", i)
        a = agg[key]
        ws, us = d.get("wrong_steps") or [], d.get("unspeakable") or []
        a[0] += 1; a[1] += bool(ws); a[2] += len(ws); a[3] += bool(us); a[4] += len(us)
print("| grader | model | method | specs | specs w/ wrong_steps | wrong_steps | specs w/ unspeakable | unspeakable |\n|---|---|---|---|---|---|---|---|")
for (g, m, me), a in sorted(agg.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
    print(f"| {g} | {m} | {me} | {a[0]} | {a[1]} | {a[2]} | {a[3]} | {a[4]} |")
