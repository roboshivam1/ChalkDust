"""Hard content checks: is the stated answer right, is the shown maths
consistent, is the lesson shaped like a lesson.

All of this is deterministic. Model-judged teaching quality is a weak signal
(docs/research/evaluation-2026-11.md); these are the checks the findings
lean on. An answer the checker cannot parse is reported as `unparsed`, never
folded into right or wrong.
"""

from __future__ import annotations

import math
import random
import re

import sympy as sp

from research.topics import Key

X = sp.Symbol("x")

# --- LaTeX -> sympy -------------------------------------------------------------

_UNIT_WORDS = r"(?:m|s|kg|N|A|V|J|W|Pa|L|mol|K|atm|g|cm|km|h|min|Hz|ohm|C|rad|deg)"


def _strip(s: str) -> str:
    s = s.replace("$", " ")
    s = re.sub(r"\\(mathrm|text|textrm|operatorname|mbox)\s*\{[^{}]*\}", " ", s)
    s = re.sub(r"\\(,|;|!|:| |quad|qquad)", " ", s)
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = s.replace("\\times", "\\cdot").replace("×", "\\cdot").replace("−", "-")
    s = s.replace("\\approx", "=").replace("≈", "=").replace("\\simeq", "=")
    s = re.sub(r"\^\s*\{?\\circ\}?", "", s).replace("°", "")
    s = re.sub(r"\\boxed\s*\{(.*)\}", r"\1", s)
    s = s.replace("²", "^2").replace("³", "^3")
    s = re.sub(r"\+\s*C\b", "", s)
    return s.strip()


def _latex_to_sympy(s: str):
    from sympy.parsing.latex import parse_latex
    s = s.strip()
    if not s:
        raise ValueError("empty")
    try:
        e = parse_latex(s)
    except Exception:
        # plain-text maths ("34.6 m", "1/6", "x**2")
        t = re.sub(rf"\b{_UNIT_WORDS}\b(\^-?\d)?", " ", s)
        t = t.replace("^", "**")
        e = sp.sympify(t, locals={"e": sp.E, "pi": sp.pi})
    return e.subs({sp.Symbol("e"): sp.E, sp.Symbol("pi"): sp.pi})


def equation_parts(line: str) -> list:
    """'a = expr = 3.27 m' -> sympy objects for each side that parses."""
    parts = [p for p in re.split(r"(?<![<>!\\])=(?!=)", _strip(line)) if p.strip()]
    out = []
    for p in parts:
        for piece in re.split(r"\\pm|±", p) if ("\\pm" in p or "±" in p) else [p]:
            try:
                out.append(_latex_to_sympy(piece))
            except Exception:
                pass
    return out


def _numbers_in(line: str) -> list[float]:
    vals = []
    for part in equation_parts(line):
        try:
            if not part.free_symbols:
                v = complex(sp.N(part))
                if abs(v.imag) < 1e-9:
                    vals.append(float(v.real))
        except Exception:
            pass
    return vals


def _close(a: float, b: float, rel: float) -> bool:
    return math.isclose(a, b, rel_tol=rel, abs_tol=1e-6 + rel * 1e-3)


def check_answer(key: Key, candidates: list[str]) -> dict:
    """Compare the answer the video states against the key.

    `candidates` are answer-bearing strings, most authoritative first (the
    AnswerBox value, then the last working line, then the last narration).
    Returns {status: correct|wrong|unparsed, source, detail}.
    """
    for src_i, text in enumerate(candidates):
        if not text or not str(text).strip():
            continue
        text = str(text)
        if key.kind in ("number", "set"):
            nums = _numbers_in(text)
            raw = [float(m) for m in re.findall(r"-?\d+(?:\.\d+)?(?:[eE]-?\d+)?",
                                                _strip(text).replace(",", " "))]
            if key.kind == "set" or not nums:
                nums = nums + [r for r in raw if r not in nums]
            if not nums:
                continue
            want = [float(v) for v in (key.value if key.kind == "set" else [key.value])]
            hit = all(any(_close(n, w, key.rel_tol) for n in nums) for w in want)
            return {"status": "correct" if hit else "wrong", "source": src_i,
                    "detail": f"found {nums[:6]} want {want}"}
        if key.kind in ("expr", "antiderivative"):
            for part in reversed(equation_parts(text)):
                if X not in part.free_symbols:
                    continue
                target = key.value
                cand = sp.diff(part, X) if key.kind == "antiderivative" else part
                tgt = sp.diff(target, X) if key.kind == "antiderivative" else target
                ok = True
                for _ in range(6):
                    xv = random.Random(len(text)).uniform(-2.5, 2.5) + _ * 0.37
                    try:
                        a = complex(sp.N(cand.subs(X, xv)))
                        b = complex(sp.N(tgt.subs(X, xv)))
                    except Exception:
                        ok = False
                        break
                    if abs(a - b) > 1e-6 * (1 + abs(b)):
                        ok = False
                        break
                return {"status": "correct" if ok else "wrong", "source": src_i,
                        "detail": f"{part} vs {key.value}"}
    return {"status": "unparsed", "source": None, "detail": "no parseable answer"}


def answer_candidates(beats: list[dict]) -> list[str]:
    """Answer-bearing strings from an assembled list of beats."""
    out = []
    for b in reversed(beats):
        if b.get("component") == "AnswerBox":
            p = b.get("params", {})
            out.append(" ".join(str(p.get(k) or "") for k in ("value",)))
            break
    for b in reversed(beats):
        p = b.get("params", {})
        if b.get("component") == "SolutionStep" and p.get("work"):
            out.append(str(p["work"][-1]))
            break
        if b.get("component") == "EquationDerivation" and p.get("steps"):
            out.append(str(p["steps"][-1]))
            break
    if beats:
        out.append(str(beats[-1].get("narration", "")))
        if len(beats) > 1:
            out.append(str(beats[-2].get("narration", "")))
    return out


def derivation_consistency(beats: list[dict], key: Key | None) -> dict:
    """For 'solve for x' topics: every equation line in the working should be
    satisfied by the true roots. Counts consistent / inconsistent / unparsed
    lines -- a crude detector of invented maths. Other topics: n/a."""
    if key is None or key.kind != "set":
        return {"applicable": False}
    roots = [float(v) for v in key.value]
    good = bad = unparsed = 0
    for b in beats:
        p = b.get("params", {})
        lines = []
        if b.get("component") == "EquationDerivation":
            lines = p.get("steps", [])
        elif b.get("component") == "SolutionStep" and p.get("work_format", "math") == "math":
            lines = p.get("work", [])
        for line in lines:
            line = re.sub(r"\{\{|\}\}", "", str(line))
            parts = equation_parts(line)
            if len(parts) < 2 or not any(X in q.free_symbols for q in parts):
                unparsed += 1
                continue
            try:
                # A line in a 'solve' derivation is consistent if it holds at
                # (at least) one true root: 'x = 2' is a correct line when the
                # roots are 2 and 3; 'x = 7' holds at neither.
                f = [sp.lambdify(X, parts[0] - q) for q in parts[1:]]
                if any(all(abs(complex(g(r))) < 1e-6 for g in f) for r in roots):
                    good += 1
                else:
                    bad += 1
            except Exception:
                unparsed += 1
    return {"applicable": True, "consistent": good, "inconsistent": bad, "unparsed": unparsed}


# --- lesson structure (from plan + beats) ------------------------------------------

def structure_checks(plan: dict | None, beats: list[dict], key: Key | None) -> dict:
    """Mechanical teaching-shape checks. Each is a yes/no a teacher would
    agree with; the score is the fraction passed. None of them proves the
    lesson teaches well -- they catch lessons that cannot."""
    comps = [b.get("component") for b in beats]
    words = [len(str(b.get("narration", "")).split()) for b in beats]
    purposes = [b.get("purpose") for b in (plan or {}).get("beats", [])] if plan else []
    narr = " ".join(str(b.get("narration", "")).lower() for b in beats)
    checks = {
        "beat_count_5_to_9": 5 <= len(beats) <= 9,
        "narration_20_to_66_words_each": bool(words) and all(20 <= w <= 66 for w in words),
        "distinct_components_ge_3": len(set(comps)) >= 3,
        "no_component_over_half": bool(comps) and max(comps.count(c) for c in set(comps)) <= len(comps) / 2,
        "opens_with_hook_or_problem": bool(comps) and comps[0] in ("TitleCard", "ProblemStatement", "SplitCompare", "GraphPlot"),
        "states_answer_if_problem": (key is None) or ("AnswerBox" in comps or "SolutionStep" in comps or "EquationDerivation" in comps),
        "addresses_misconception": ("misconception" in purposes) or any(w in narr for w in ("mistake", "tempting", "wrong", "trap", "common error", "careful")),
        "no_filler_greeting": not any(w in narr for w in ("hello everyone", "welcome back", "in this video", "don't forget to subscribe")),
    }
    return {"checks": checks, "score": sum(checks.values()) / len(checks)}
