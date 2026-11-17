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


def _trig_degrees(m: re.Match) -> str:
    fn, num = m.group(1), m.group(3)
    if fn.startswith("\\"):
        return f"{fn}({num} \\cdot \\frac{{\\pi}}{{180}})"
    return f"{fn}({num}*pi/180)"


def _strip(s: str) -> str:
    s = s.replace("$", " ")
    s = s.replace("{,}", "")                          # 1{,}275
    s = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", s)     # 1,275
    s = re.sub(r"\\(mathrm|text|textrm|operatorname|mbox)\s*\{[^{}]*\}", " ", s)
    s = re.sub(r"\\(,|;|!|:| |quad|qquad)", " ", s)
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = s.replace("\\times", "\\cdot").replace("×", "\\cdot").replace("−", "-")
    s = s.replace("\\approx", "=").replace("≈", "=").replace("\\simeq", "=")
    # degrees inside a trig function are converted, not dropped: sin(60°) is not sin(60)
    s = re.sub(r"(\\?(?:sin|cos|tan))\s*(\()?\s*(-?\d+(?:\.\d+)?)\s*(?:\^\s*\{?\\circ\}?|°)\s*(?(2)\))",
               _trig_degrees, s)
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
    if "\\" not in s:
        # plain-text maths first: the LaTeX grammar reads 'sin(x)' as s*i*n(x)
        from research.safe_math import safe_sympify
        try:
            t = re.sub(rf"\b{_UNIT_WORDS}\b(\^-?\d)?", " ", s).strip()
            return safe_sympify(t, {"x": X})
        except Exception:
            pass
    try:
        e = parse_latex(s)
    except Exception:
        # plain-text maths ("34.6 m", "1/6", "x**2")
        from research.safe_math import safe_sympify   # no eval of model text
        t = re.sub(rf"\b{_UNIT_WORDS}\b(\^-?\d)?", " ", s)
        e = safe_sympify(t.strip(), {"x": X})
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


# A condition after the value ('y = -5 at x = 2', '\text{ when } t = 3') is not
# the answer; cut it before reading the line's final value.
_TAIL = re.compile(r"\\text\s*\{\s*(?:at|when|for|where|if|with)\b.*$|\s(?:at|when|for|where)\s+[A-Za-z]\s*=.*$")
_SET_SPLIT = re.compile(r"\\text\s*\{\s*(?:or|and)\s*\}|\\quad|\bor\b|\band\b|,|;")


def _value_of(piece: str) -> float | None:
    """A real number if `piece` is a closed numeric expression, else None.
    Unevaluated integrals/derivatives are not values."""
    try:
        e = _latex_to_sympy(piece)
        if e.free_symbols or e.has(sp.Integral, sp.Derivative):
            return None
        v = complex(sp.N(e))
    except Exception:
        return None
    return float(v.real) if abs(v.imag) < 1e-9 else None


def final_value(line: str) -> float | None:
    """The value a line states: its LAST side ('P = 6/36 = 1/5' states 1/5,
    not 1/6). A line whose last side is a formula ('T = 2\\pi\\sqrt{L/g}')
    states no value. A plain-words line with no maths ('index 5') falls back
    to its last number."""
    line = _TAIL.sub("", str(line))
    parts = [p for p in re.split(r"(?<![<>!\\])=(?!=)", _strip(line)) if p.strip()]
    if parts:
        v = _value_of(parts[-1])
        if v is not None:
            return v
    if "=" in line or "\\" in line or re.search(r"(?<![A-Za-z])[A-Za-z](?![A-Za-z])", _strip(line)):
        return None
    raw = re.findall(r"-?\d+(?:\.\d+)?(?:[eE]-?\d+)?", _strip(line))
    return float(raw[-1]) if raw else None


def _set_values(line: str) -> list[float]:
    """Roots stated on a line ('x = 2 or x = 3', 'x = 2, 3'). A piece whose
    left side is an expression ('(x-2)(x-3) = 0') is an equation, not a root."""
    vals = []
    for piece in _SET_SPLIT.split(str(line)):
        if piece and piece.strip():
            sides = [q for q in re.split(r"(?<![<>!\\])=(?!=)", _strip(_TAIL.sub("", piece))) if q.strip()]
            if len(sides) >= 2 and not re.fullmatch(r"\s*[A-Za-z](?:_\{?\w+\}?)?\s*", sides[-2]):
                continue
            v = final_value(piece)
            if v is not None:
                vals.append(v)
    return vals


def check_answer(key: Key, candidates: list[str], narration: str = "", request: str = "") -> dict:
    """Compare the answer the video SHOWS against the key.

    `candidates` are answer-bearing maths lines, most authoritative first
    (the AnswerBox value, then the last working line). The first one that
    states a value decides; only if none does is the narration's final
    asserted value used (check_spoken's rule). Returns {status:
    correct|wrong|unparsed, source, detail}; a line that states no value is
    skipped, never scored wrong.
    """
    want = [float(v) for v in (key.value if key.kind == "set" else [key.value])] \
        if key.kind in ("number", "set") else []
    for src_i, text in enumerate(candidates):
        if not text or not str(text).strip():
            continue
        text = str(text)
        if key.kind == "number":
            v = final_value(text)
            if v is None:
                continue
            return {"status": "correct" if _close(v, want[0], key.rel_tol) else "wrong", "source": src_i,
                    "detail": f"states {v:g} want {want[0]:g}"}
        if key.kind == "set":
            nums = _set_values(text)
            if not nums:
                continue
            hit = all(any(_close(n, w, key.rel_tol) for n in nums) for w in want)
            return {"status": "correct" if hit else "wrong", "source": src_i,
                    "detail": f"states {nums[:6]} want {want}"}
        if key.kind in ("expr", "antiderivative"):
            for part in reversed(equation_parts(text)):
                if X not in part.free_symbols or part.has(sp.Integral, sp.Derivative):
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
    if narration and key.kind in ("number", "set"):
        s = check_spoken(key, narration, request)
        if s["status"] in ("correct", "wrong"):
            return {"status": s["status"], "source": "narration", "detail": s["detail"]}
    return {"status": "unparsed", "source": None, "detail": "no parseable answer"}


# --- the spoken answer ------------------------------------------------------------

_ONES = {"zero": 0, "oh": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
         "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
         "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
         "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
         "eighty": 80, "ninety": 90}
_SCALES = {"hundred": 100, "thousand": 1000, "million": 10**6}
_ORD = {"half": 2, "halves": 2, "third": 3, "thirds": 3, "quarter": 4, "quarters": 4, "fourth": 4,
        "fourths": 4, "fifth": 5, "fifths": 5, "sixth": 6, "sixths": 6, "seventh": 7, "sevenths": 7,
        "eighth": 8, "eighths": 8, "ninth": 9, "ninths": 9, "tenth": 10, "tenths": 10,
        "twelfth": 12, "twelfths": 12, "thirty-sixth": 36, "thirty-sixths": 36}


def _word_numbers(text: str) -> list[tuple[int, float]]:
    """Numbers written in words, with their character positions: 'thirty four
    point six', 'one sixth', 'twelve hundred and seventy five', 'minus five',
    'twenty root three'. 'and' joins only after hundred/thousand ('five and
    two' is two numbers). Deliberately simple."""
    found = list(re.finditer(r"[a-z]+(?:-[a-z]+)?", text.lower()))
    toks = [m.group(0) for m in found]
    vals, i = [], 0
    while i < len(toks):
        t = toks[i]
        neg = t in ("minus", "negative")
        j = i + 1 if neg else i
        total, cur, used = 0, 0, False
        while j < len(toks):
            w = toks[j]
            parts = w.split("-")
            if all(q in _ONES or q in _TENS for q in parts) and not (w in _ORD):
                cur += sum(_ONES.get(q, 0) + _TENS.get(q, 0) for q in parts)
                used = True
            elif w in _SCALES and used:
                if _SCALES[w] == 100:
                    cur *= 100
                else:
                    total += cur * _SCALES[w]
                    cur = 0
            elif w == "and" and used and toks[j - 1] in _SCALES and j + 1 < len(toks) and (
                    toks[j + 1] in _ONES or toks[j + 1] in _TENS or toks[j + 1].split("-")[0] in _TENS):
                pass
            else:
                break
            j += 1
        if not used:
            i += 1
            continue
        v = float(total + cur)
        # decimal part: 'point six four'
        if j < len(toks) and toks[j] == "point":
            k, digs = j + 1, ""
            while k < len(toks) and toks[k] in _ONES and _ONES[toks[k]] < 10:
                digs += str(_ONES[toks[k]])
                k += 1
            if digs:
                v = float(f"{int(v)}.{digs}")
                j = k
        def _num_word(k: int) -> float:
            return float(sum(_ONES.get(q, 0) + _TENS.get(q, 0) for q in toks[k].split("-")))

        def _is_num_word(k: int) -> bool:
            return k < len(toks) and all(q in _ONES or q in _TENS for q in toks[k].split("-"))

        # 'twenty root three', 'two square root of two'
        if j < len(toks) and toks[j] == "root" and _is_num_word(j + 1):
            v, j = v * math.sqrt(_num_word(j + 1)), j + 2
        elif toks[j:j + 3] == ["square", "root", "of"] and _is_num_word(j + 3):
            v, j = v * math.sqrt(_num_word(j + 3)), j + 4
        # fraction: 'one sixth', 'two thirds', 'one in six', 'one over six', 'one out of six'
        if j < len(toks) and toks[j] in _ORD:
            v, j = v / _ORD[toks[j]], j + 1
        elif j < len(toks) and (toks[j] in ("over", "in") and _is_num_word(j + 1)
                                or toks[j:j + 2] == ["out", "of"] and _is_num_word(j + 2)):
            k = j + (2 if toks[j] == "out" else 1)
            d = _num_word(k)
            if d:
                v, j = v / d, k + 1
        vals.append((found[i].start(), -v if neg else v))
        i = j if j > i else i + 1
    return vals


def _blank(s: str, a: int, b: int) -> str:
    return s[:a] + " " * (b - a) + s[b:]


def spoken_values(text: str) -> list[tuple[int, float]]:
    r"""Every value the narration states, with its position, however it is
    written: LaTeX segments ('$20\sqrt{3}$', '\frac{1}{6}'), digit fractions
    ('6/36', '1 in 6'), digits ('1,275'), and number words. Each span is
    read once (the 6 and 36 of '6/36' are not also values), and exponents
    ('m/s^2') are not values."""
    vals: list[tuple[int, float]] = []
    s = re.sub(r"\^\s*\{?-?\d+\}?", lambda m: " " * len(m.group(0)), text)
    for m in re.finditer(r"\$([^$]+)\$|\\[dt]?frac\s*\{[^{}]*\}\s*\{[^{}]*\}|\d+(?:\.\d+)?\s*\\sqrt\s*\{[^{}]*\}"
                         r"|\\sqrt\s*\{[^{}]*\}", s):
        seg = text[m.start():m.end()].strip("$")
        nums = _numbers_in(seg)
        vals += [(m.start(), v) for v in nums[-1:]]
        s = _blank(s, m.start(), m.end())
    for m in re.finditer(r"(-?\d+(?:\.\d+)?)\s*(?:/|\bover\b|\bout of\b|\bin\b)\s*(\d+(?:\.\d+)?)", s):
        if float(m.group(2)):
            vals.append((m.start(), float(m.group(1)) / float(m.group(2))))
        s = _blank(s, m.start(), m.end())
    for m in re.finditer(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?", s):
        vals.append((m.start(), float(m.group(0).replace(",", ""))))
        s = _blank(s, m.start(), m.end())
    vals += _word_numbers(s)
    return sorted(vals, key=lambda pv: pv[0])


def spoken_numbers(text: str) -> list[float]:
    return [v for _, v in spoken_values(text)]


# Words that introduce a stated result: 'the range IS 34.6', 'comes to ABOUT 35',
# 'it should BE 2.0', 'SO 1/6'. '=' and 'equals' are weak: they also restate
# givens ('for n equals four', 'g = 10').
_STRONG_CUES = {"is", "are", "was", "be", "gives", "get", "gets", "got", "comes", "about", "approximately",
                "roughly", "around", "nearly", "exactly", "so", "therefore", "thus", "makes", "answer", "result"}
_WEAK_CUES = {"equals", "equal", "=", "≈"}


def _cue(text: str, pos: int) -> str | None:
    toks = re.findall(r"[a-z']+|\d+(?:\.\d+)?|=|≈", text[max(0, pos - 40):pos].lower())[-2:]
    if any(t in _STRONG_CUES for t in toks):
        return "strong"
    if any(t in _WEAK_CUES for t in toks):
        return "weak"
    return None


def check_spoken(key: Key, text: str, request: str = "") -> dict:
    """Does the narration SAY the right answer as its result?

    Numbers copied from the request are restated givens, so they are dropped
    unless asserted as a result ('is', 'be', 'about', 'so' ...). The
    narration's result is the last value introduced by an answer cue
    (those, or a weak '='/'equals'), or the last value if none is cued;
    it is 'correct' if it matches the key, 'wrong' if it does not, and
    'not_found' if the narration states no value of its own. When the key
    itself appears in the request (binary search over a list containing the
    answer), speech cannot show the answer was worked out: 'n/a'. For a set
    (roots) every root must be stated. Expression answers are spoken in
    words and are not checked: 'n/a'.
    """
    if key.kind in ("expr", "antiderivative"):
        return {"status": "n/a", "detail": "expression answers are not checked in speech"}
    want = [float(v) for v in (key.value if key.kind == "set" else [key.value])]
    vals = spoken_values(text)
    if key.kind == "set":
        nums = [v for _, v in vals]
        hit = all(any(_close(n, w, key.rel_tol) for n in nums) for w in want)
        return {"status": "correct" if hit else "not_found", "detail": f"found {sorted(set(nums))[:10]} want {want}"}
    req = spoken_numbers(request) if request else []
    if any(_close(r, want[0], 1e-6) for r in req):
        return {"status": "n/a", "detail": "the key appears in the request"}
    # A number from the request is the narration restating a given, unless it
    # is asserted as a result ('it should be 2.0' when the block is 2 kg is a
    # wrong answer, and was a real one).
    own = []
    for p, v in vals:
        c = _cue(text, p)
        if c != "strong" and any(_close(v, r, 1e-6) for r in req):
            continue
        own.append((p, v, c))
    if not own:
        return {"status": "not_found", "detail": f"no value beyond the request's {req[:6]}"}
    cued = [x for x in own if x[2]]
    final = (cued or own)[-1][1]
    return {"status": "correct" if _close(final, want[0], key.rel_tol) else "wrong",
            "detail": f"result {final:g} ({'cued' if cued else 'last'}) want {want[0]:g}"}


def final_narration(beats: list[dict]) -> str:
    """What the student hears last: the last two narrations."""
    return " ".join(str(b.get("narration", "")) for b in beats[-2:])


def answer_candidates(beats: list[dict]) -> list[str]:
    """Answer-bearing maths lines from an assembled list of beats: the
    AnswerBox value, then the last working line. (The narration is checked
    separately: check_answer(..., narration=final_narration(beats)).)"""
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
