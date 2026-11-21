"""Mutation tests of research.content.check_answer / check_spoken.
Run from repo root:  .lab/venv-harness/Scripts/python.exe <this file>
Read-only on the repo; no Manim/LaTeX/GPU."""
import copy, json, re, sys, io
from collections import defaultdict, Counter
sys.path.insert(0, ".")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from research import content as C
from research.topics import topic_set

TOPICS = {t.id: t for t in topic_set("dev")}
ORIG_ANS, ORIG_SPK = C.check_answer, C.check_spoken
NUM = re.compile(r"\d+(?:\.\d+)?")


def load():
    out = []
    for f in ("E1b-methods", "E1c-work"):
        for l in open(f"research/results/{f}.jsonl", encoding="utf-8"):
            d = json.loads(l)
            if d.get("record") == "server":
                continue
            try:
                spec = json.load(open(f"{d['trace_dir']}/spec.json", encoding="utf-8"))
            except Exception:
                continue
            if d["topic"] in TOPICS and spec.get("beats"):
                out.append((f, d, spec["beats"]))
    return out


def run(topic, beats):
    # replicates research/run.py summarise() lines 104-109
    a = C.check_answer(topic.key, C.answer_candidates(beats), C.final_narration(beats), topic.request)
    s = C.check_spoken(topic.key, C.final_narration(beats), topic.request)
    return a, s


def shown_loc(beats):
    for i in range(len(beats) - 1, -1, -1):
        if beats[i].get("component") == "AnswerBox":
            return (lambda b, i=i: b[i]["params"].get("value"),
                    lambda b, v, i=i: b[i]["params"].__setitem__("value", v))
    for i in range(len(beats) - 1, -1, -1):
        p = beats[i].get("params", {})
        if beats[i].get("component") == "SolutionStep" and p.get("work"):
            return (lambda b, i=i: b[i]["params"]["work"][-1],
                    lambda b, v, i=i: b[i]["params"]["work"].__setitem__(-1, v))
        if beats[i].get("component") == "EquationDerivation" and p.get("steps"):
            return (lambda b, i=i: b[i]["params"]["steps"][-1],
                    lambda b, v, i=i: b[i]["params"]["steps"].__setitem__(-1, v))
    return None


def wants(key):
    return [float(v) for v in (key.value if key.kind == "set" else [key.value])]


def near_any(v, ws, tol):
    return any(C._close(v, w, max(tol, 1e-6)) for w in ws)


def wrong_num(tok, ws, tol):
    v = float(tok)
    for cand in ((v * 1.5 if "." in tok else v + 1), v + 7, v + 11, v * 2.5):
        if not near_any(cand, ws, tol):
            return f"{cand:g}"


def m1(key, beats, req):
    loc = shown_loc(beats)
    if not loc or key.kind not in ("number", "set"):
        return None
    g, s = loc
    t = str(g(beats))
    ms = list(NUM.finditer(t))
    if not ms:
        return None
    near = [x for x in ms if near_any(float(x.group(0)), wants(key), key.rel_tol)]
    m = (near or ms)[-1]
    w = wrong_num(m.group(0), wants(key), key.rel_tol)
    nb = copy.deepcopy(beats)
    s(nb, t[:m.start()] + w + t[m.end():])
    return nb


def m2_add(key, beats, req):
    loc = shown_loc(beats)
    if key.kind != "set" or not loc:
        return None
    g, s = loc
    nb = copy.deepcopy(beats)
    s(nb, str(g(beats)) + " or x = 7")
    return nb


def m2_drop(key, beats, req):
    loc = shown_loc(beats)
    if key.kind != "set" or not loc:
        return None
    nb = copy.deepcopy(beats)
    loc[1](nb, "x = 2")
    return nb


def m3(key, beats, req):
    """Replace EVERY occurrence of the answer number (for a set: of one root) in the last two
    narrations. Not applicable if the answer is also written in words/fractions we cannot rewrite."""
    if key.kind not in ("number", "set"):
        return None
    ws = wants(key)
    victim = ws[-1]
    nb = copy.deepcopy(beats)
    done = False
    for bi in (len(beats) - 1, len(beats) - 2):
        if bi < 0:
            continue
        t = str(nb[bi].get("narration", ""))
        def sub(m):
            nonlocal done
            if near_any(float(m.group(0).replace(",", "")), [victim], key.rel_tol):
                done = True
                return wrong_num(m.group(0).replace(",", ""), ws, key.rel_tol)
            return m.group(0)
        nb[bi]["narration"] = re.sub(r"\d+(?:,\d{3})*(?:\.\d+)?", sub, t)
    if not done:
        return None
    left = [v for _, v in C.spoken_values(C.final_narration(nb)) if near_any(v, [victim], key.rel_tol)]
    return None if left else nb


def _w(key):
    return "7 and 11" if key.kind == "set" else wrong_num(repr(wants(key)[0]), wants(key), key.rel_tol)


def m4(key, beats, req):
    if key.kind not in ("number", "set"):
        return None
    nb = copy.deepcopy(beats)
    nb[-1]["narration"] = str(nb[-1].get("narration", "")).rstrip() + f" So the answer is {_w(key)}."
    return nb


def _req_nums(req):
    return NUM.findall(req)[:3]


def m5a(key, beats, req):
    if key.kind not in ("number", "set"):
        return None
    ns = _req_nums(req) or ["3"]
    nb = copy.deepcopy(beats)
    nb[-2]["narration"] = ""
    nb[-1]["narration"] = f"Starting from {', '.join(ns)}, we get {_w(key)}."
    return nb


def m5b(key, beats, req):
    if key.kind not in ("number", "set"):
        return None
    ns = _req_nums(req) or ["3"]
    nb = copy.deepcopy(beats)
    nb[-2]["narration"] = ""
    nb[-1]["narration"] = f"The answer is {_w(key)}, using {' and '.join(ns)} from the problem."
    return nb


def _equiv(key, text):
    """Independent oracle: is the mutated shown text still equivalent to the key?"""
    for part in reversed(C.equation_parts(text)):
        if C.X not in part.free_symbols:
            continue
        cand = C.sp.diff(part, C.X) if key.kind == "antiderivative" else part
        tgt = C.sp.diff(key.value, C.X) if key.kind == "antiderivative" else key.value
        for xv in (0.37, 1.1, -0.8, 1.9):
            try:
                if abs(complex(C.sp.N(cand.subs(C.X, xv))) - complex(C.sp.N(tgt.subs(C.X, xv)))) > 1e-6:
                    return False
            except Exception:
                return None
        return True
    return None


def m6_coef(key, beats, req):
    loc = shown_loc(beats)
    if key.kind not in ("expr", "antiderivative") or not loc:
        return None
    g, s = loc
    t = str(g(beats))
    cands = list(re.finditer(r"(?<![\^{\w.])\d+(?![\d.])", t)) or list(NUM.finditer(t))
    if not cands:
        return None
    m = cands[0]
    nt = t[:m.start()] + str(int(float(m.group(0))) + 1) + t[m.end():]
    if _equiv(key, nt) is not False:
        return None
    nb = copy.deepcopy(beats)
    s(nb, nt)
    return nb


def m6_sign(key, beats, req):
    loc = shown_loc(beats)
    if key.kind not in ("expr", "antiderivative") or not loc:
        return None
    g, s = loc
    t = str(g(beats))
    idx = [m.start() for m in re.finditer(r"[+\-]", t)]
    idx = [i for i in idx if i > 0 and not t[:i].rstrip().endswith("=")]
    if not idx:
        return None
    i = idx[-1]
    nt = t[:i] + ("-" if t[i] == "+" else "+") + t[i + 1:]
    if _equiv(key, nt) is not False:
        return None
    nb = copy.deepcopy(beats)
    s(nb, nt)
    return nb


MUTS = [("m1", m1), ("m2_add", m2_add), ("m2_drop", m2_drop), ("m3", m3), ("m4", m4),
        ("m5a", m5a), ("m5b", m5b), ("m6_coef", m6_coef), ("m6_sign", m6_sign)]


# ---- proposed fixes (monkeypatched in) ----
def fixed_check_answer(key, candidates, narration="", request=""):
    """F1: a shown set must EQUAL the key: any extra value that matches no root -> wrong."""
    r = ORIG_ANS(key, candidates, narration, request)
    if key.kind == "set" and r["status"] == "correct" and isinstance(r["source"], int):
        nums = C._set_values(str(candidates[r["source"]]))
        ws = wants(key)
        extras = [n for n in nums if not near_any(n, ws, key.rel_tol)]
        if extras:
            return {**r, "status": "wrong", "detail": r["detail"] + f" EXTRA {extras}"}
    return r


def _sentences(t):
    return [x for x in re.split(r"(?<=[.!?])\s+", t) if x.strip()]


def fixed_check_spoken(key, text, request=""):
    """F2: for sets, drop request givens (as the number branch does), take the trailing run of
    value-bearing sentences, and require exact equality with the roots."""
    if key.kind != "set":
        return ORIG_SPK(key, text, request)
    ws = wants(key)
    req = C.spoken_numbers(request) if request else []

    def own(t):
        return [v for p, v in C.spoken_values(t)
                if not (any(C._close(v, r, 1e-6) for r in req) and not near_any(v, ws, key.rel_tol)
                        and C._cue(t, p) != "strong")]
    vals = []
    for sent in reversed(_sentences(text)):
        o = own(sent)
        if o:
            vals = o + vals
        elif vals:
            break
    if not vals:
        return {"status": "not_found", "detail": "no set values"}
    hit = all(near_any(w, vals, key.rel_tol) for w in ws)
    extra = [v for v in vals if not near_any(v, ws, key.rel_tol)]
    st = "correct" if hit and not extra else ("wrong" if (hit or extra) else "not_found")
    return {"status": st, "detail": f"values {sorted(set(vals))[:10]} want {ws} extra {extra}"}


def score(recs, label, ex):
    tab = defaultdict(lambda: [0, 0, 0])
    base = {}
    for f, d, beats in recs:
        t = TOPICS[d["topic"]]
        a, s = run(t, beats)
        base[(f, d["trace_dir"])] = (a["status"], s["status"])
        for name, fn in MUTS:
            nb = fn(t.key, beats, t.request)
            if nb is None:
                continue
            ma, ms = run(t, nb)
            for ck, bs, r in (("answer", a["status"], ma), ("spoken", s["status"], ms)):
                if bs != "correct":
                    continue
                row = tab[(name, ck)]
                row[0] += 1
                if r["status"] == "correct":
                    row[1] += 1
                    ex.setdefault((label, name, ck), []).append((d["topic"], r["detail"], d["trace_dir"][-45:]))
                elif r["status"] == "unparsed":
                    row[2] += 1
    return tab, base


def main():
    recs = load()
    print("specs", len(recs))
    mism = 0
    for f, d, beats in recs:
        t = TOPICS[d["topic"]]
        a, s = run(t, beats)
        if d.get("answer", {}).get("status") != a["status"] or d.get("answer_spoken", {}).get("status") != s["status"]:
            mism += 1
            print("MISMATCH vs stored", d["topic"], d.get("answer", {}).get("status"), a["status"],
                  d.get("answer_spoken", {}).get("status"), s["status"])
    print(f"sanity: {len(recs) - mism}/{len(recs)} recomputed statuses equal stored record statuses")
    ex = {}
    base0 = None
    for label in ("current", "fixed"):
        if label == "fixed":
            C.check_answer, C.check_spoken = fixed_check_answer, fixed_check_spoken
        tab, base = score(recs, label, ex)
        if label == "current":
            base0 = base
            print("baseline answer", Counter(v[0] for v in base.values()), "spoken", Counter(v[1] for v in base.values()))
        else:
            ch = [(k, base0[k], base[k]) for k in base if base0[k] != base[k]]
            print(f"FIX baseline changes on unmutated specs: {len(ch)}")
            for k, o, nw in ch:
                print("  ", k[1][-60:], o, "->", nw)
        print(f"\n== {label}: false passes / applicable (unparsed) ==")
        for (name, ck), (a_, fp, up) in sorted(tab.items()):
            print(f"{name:8s} {ck:7s} {fp}/{a_}  (unparsed {up})")
    print("\n== examples (current) ==")
    for k, v in ex.items():
        if k[0] == "current":
            for e in v[:2]:
                print(k[1:], e)
    print("\n== residual false passes with fix ==")
    for k, v in ex.items():
        if k[0] == "fixed":
            for e in v[:2]:
                print(k[1:], e)


main()
