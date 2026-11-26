"""Animation components in the harness (run 3): the method option `anim=1`.

An animation (chalkdust/scenes/components/animation.py) plays over several
consecutive beats, one phase each, and every number it shows is computed by
code. A small model must not be trusted to keep four beats' setup identical or
to say numbers it was never shown, so with `anim=1`:

  - the planner sees the animations as ONE plan item each; `expand_plan`
    turns that item into one beat per phase, in order;
  - the first phase's beat fills the setup (function, x_range, ...) and its
    narration; every later phase gets that setup and its phase name from code
    and writes only narration and cues;
  - each later beat's prompt carries FACTS: the numbers the screen shows in
    that phase, computed with the component's own functions (`facts`);
  - cues may only name that phase's moments (`beat_schema`).

The calculator (`calc_value`) also takes derivative(f, a), limit(f, a) and
integral(f, a, b), so a calculus answer is computed rather than copied from
the model's arithmetic. With calc=1 its value decides the answer: generate.py
puts it in the prompt as the answer, writes it into the AnswerBox, and flags
an answer beat whose narration does not say it, and an animation whose own
headline number disagrees with it (the picture would be true for another
question).

`research/tests/test_animate.py` fails when a registered Animation has no card
here, or when a card's phases differ from the component's: that is the check
that the harness picks up new components.
"""

from __future__ import annotations

import ast
import copy
import math
from functools import lru_cache
from typing import Any

import sympy as sp

import chalkdust.scenes.components  # noqa: F401  (registers every component)
from chalkdust.scenes.components.animation import Animation
from chalkdust.scenes.components.base import get_component, registered_names
from research.catalog import NARRATION, _hoist, params_schema
from research.safe_math import UnsafeExpression, safe_sympify

# name -> (what it shows, when to use it, how to fill the setup, {phase: what it teaches})
CARDS: dict[str, tuple[str, str, str, dict[str, str]]] = {
    "SecantToTangent": (
        "ANIMATION, plays over 4 beats: a curve and a point P on it; a second point Q and the "
        "secant line through both with its slope; Q slides into P while the slope readout "
        "settles and the line becomes the tangent; then f'(a) is named with its value.",
        "The derivative at a point: what it means (slope of the tangent, limit of secant "
        "slopes) and its value.",
        "function: f in the one variable x, * for multiplication (e.g. 'x^2/4 + 1'). "
        "x_range: [min, max] holding a and a+h. a: the point. h: Q's first gap from a "
        "(e.g. 2; negative puts Q on the left).",
        {"pose": "show the curve and the point; ask how steep the curve is right there",
         "secant": "a second point and the line through both; its slope as rise over run",
         "shrink": "slide the second point in: the slope settles and the line becomes the tangent",
         "name": "name it: the derivative as the limit of the slope, and its value"}),
    "LimitAtHole": (
        "ANIMATION, plays over 3 beats: a curve with an open hole where f is undefined; two "
        "points slide toward the hole from both sides with live x and f(x) readouts that "
        "settle on one number; then the limit is named with its value.",
        "A limit at a point where f is undefined (0/0): what a limit means and its value.",
        "function: f in the one variable x, undefined at a (e.g. '(x^2 - 1)/(x - 1)'). "
        "x_range: [min, max] with a well inside. a: the point.",
        {"pose": "show the curve with a hole where f is undefined; ask what f does near it",
         "approach": "two points slide in from both sides; both readouts settle on one number",
         "name": "name it: the limit, written in symbols, and its value"}),
    "RiemannSum": (
        "ANIMATION, plays over 4 beats: the area under a curve from a to b; a few rectangles "
        "with their total and the gap to the true area; the rectangles split again and again "
        "while the sum closes in; then the sum notation and the definite integral with its value.",
        "A definite integral as area: rectangle (Riemann) sums and their limit.",
        "function: f in the one variable x, never negative on [a, b] (e.g. 'x^2/4 + 1'). "
        "x_range: [min, max] holding [a, b]. a, b: the interval. rule: left|right|mid (which "
        "point of each rectangle sets its height). n0: rectangles at first, 3 to 8.",
        {"pose": "shade the area under the curve from a to b; ask how big it is",
         "boxes": "a few rectangles, their total area, and the gap to the true area",
         "refine": "split the rectangles again and again: the sum closes in on the area",
         "name": "name it: the sum in sigma notation, then the definite integral and its value"}),
}

# Which calculator function an animation's headline number answers.
HEADLINE_CALC = {"SecantToTangent": "derivative", "LimitAtHole": "limit", "RiemannSum": "integral"}


def headline_call(component: str, setup: dict) -> str | None:
    """The calculator call for an animation's headline quantity at its setup,
    e.g. 'integral(x^2/4 + 1, 0, 3)'; None if it has no headline."""
    kind = HEADLINE_CALC.get(component)
    if kind is None or "function" not in setup:
        return None
    ends = [setup["a"], setup["b"]] if kind == "integral" else [setup["a"]]
    return f"{kind}({setup['function']}, {', '.join(f'{float(v):.12g}' for v in ends)})"

MAX_ANIMATIONS = 2          # per plan
CUE_MAX = 60                # chalkdust.scenes.components.animation.MAX_CUE_CHARS
# Narration for every beat under anim=1: long enough for a 3-5 minute video at
# 10-14 beats, short enough for the 25 s cap at the validator's 160 wpm estimate.
NARRATION_ANIM = {**NARRATION, "minLength": 240, "maxLength": 400}


def animation_names() -> list[str]:
    """Every registered animation component, in registry order."""
    return [n for n in registered_names() if issubclass(get_component(n), Animation)]


def phases(component: str) -> list[str]:
    return list(get_component(component).PHASES)


def is_animation(component: str) -> bool:
    return component in CARDS


def card_text(component: str) -> str:
    what, when, how, _ = CARDS[component]
    return f"{component}\n  shows: {what}\n  use for: {when}\n  setup: {how}"


def catalog_text() -> str:
    return "\n".join(card_text(c) for c in CARDS)


# --- plan -------------------------------------------------------------------------

def expand_plan(plan: dict) -> dict:
    """One beat per phase for each animation item, in phase order. Repeats of
    the same animation in a row are one item; animations past MAX_ANIMATIONS
    are dropped (logged in plan['anim_dropped'])."""
    out, seen, dropped = [], 0, []
    for b in plan.get("beats", []):
        comp = b.get("component")
        if not is_animation(comp):
            out.append(b)
            continue
        if out and out[-1].get("component") == comp and out[-1].get("anim_phase") is not None:
            continue
        if seen >= MAX_ANIMATIONS:
            dropped.append(comp)
            continue
        seen += 1
        ph = phases(comp)
        for k, name in enumerate(ph):
            out.append({**b, "idea": f"{b.get('idea', '')} -- phase {k + 1} of {len(ph)} "
                                     f"({name}): {CARDS[comp][3][name]}",
                        "anim_phase": name, "anim_first": k == 0, "anim_id": seen})
    plan = {**plan, "beats": out}
    if dropped:
        plan["anim_dropped"] = dropped
    return plan


# --- beat schema ---------------------------------------------------------------------

@lru_cache(maxsize=None)
def beat_schema(component: str, phase: str, with_setup: bool) -> dict:
    """narration + params, where params holds the setup fields only for the
    first phase, and cues keyed by this phase's moments only. phase itself is
    always set by code."""
    body, defs = _hoist(params_schema(component))
    props = copy.deepcopy(body.get("properties", {}))
    moments = [m.name for m in get_component(component).PHASES[phase]]
    cues = {"type": "object", "additionalProperties": False,
            "properties": {m: {"type": "string", "minLength": 3, "maxLength": CUE_MAX}
                           for m in moments}}
    keep = {k: v for k, v in props.items() if with_setup and k not in ("phase", "cues")}
    keep["cues"] = cues
    req = [k for k in body.get("required", []) if k in keep]
    params = {"type": "object", "additionalProperties": False, "properties": keep, "required": req}
    s = {"type": "object", "additionalProperties": False,
         "properties": {"narration": NARRATION_ANIM, "params": params},
         "required": ["narration", "params"]}
    if defs and with_setup:
        s["$defs"] = defs
    return s


def merge_params(component: str, phase: str, model_params: dict, setup: dict | None) -> dict:
    """The beat's real params: the carried setup (or the model's, for the
    first phase), the phase from code, and the model's cues."""
    p = dict(model_params) if isinstance(model_params, dict) else {}
    cues = p.pop("cues", {}) if isinstance(p.get("cues", {}), dict) else {}
    p.pop("phase", None)
    base = dict(setup) if setup is not None else p
    return {**base, "phase": phase, "cues": {k: v for k, v in cues.items() if isinstance(v, str)}}


def setup_of(params: dict) -> dict:
    return {k: v for k, v in params.items() if k not in ("phase", "cues")}


# --- what the screen shows -------------------------------------------------------------

def _f(v: float, d: int) -> str:
    return f"{v:,.{d}f}"


def _g(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else f"{v:.12g}"


def facts(component: str, setup: dict) -> tuple[dict[str, list[str]], float]:
    """{phase: numbers on screen, as sentences}, and the headline number
    (f'(a), the limit, the integral). Raises ValueError for a setup the
    component refuses, with its message."""
    params = get_component(component).Params.model_validate(
        {**setup, "phase": phases(component)[0]})
    if component == "SecantToTangent":
        from chalkdust.scenes.components.secant_tangent import slope_plan
        from chalkdust.scenes.plotting import Fn
        f, a, h = Fn(params.function), params.a, params.h
        x0, x1 = params.x_range
        plan = slope_plan(f, a, h, x1 - x0)
        d = plan.decimals
        fa, fq = f.at(a), f.at(a + h)
        sec = (fq - fa) / h
        return {
            "pose": [f"P is the point on the curve at x = {_g(a)}, where f(x) = {_g(round(fa, 6))}. "
                     "No slope is shown yet."],
            "secant": [f"Q is at x = {_g(a + h)}, where f(x) = {_g(round(fq, 6))}. The line PQ rises "
                       f"{_g(round(fq - fa, 6))} over a run of {_g(h)}; the screen reads "
                       f"slope = {_f(sec, d)}."],
            "shrink": [f"As Q slides in, the h readout shrinks from {_g(h)} toward 0 and the slope "
                       f"readout moves from {_f(sec, d)} and settles on {_f(plan.shown, d)}. The line "
                       "ends as the tangent at P."],
            "name": [f"The screen shows the limit definition of f'(a) and f'({_g(a)}) = "
                     f"{_f(plan.shown, d)}."],
        }, plan.slope
    if component == "LimitAtHole":
        from chalkdust.scenes.components.limit_hole import _analyse
        x0, x1 = params.x_range
        lim = _analyse(params.function, x0, x1, params.a)
        a = params.a
        if lim.exact_d is not None:
            shown = f"= {_f(round(lim.L, lim.exact_d), lim.exact_d)}"
        else:
            shown = f"is about {_f(lim.L, lim.dy)}"
        return {
            "pose": [f"The open hole at x = {_g(a)}: the screen says f({_g(a)}) is undefined, and asks "
                     f"what f does for x near {_g(a)}. No value is shown yet."],
            "approach": [f"Two points start at x = {_f(a - lim.h0, lim.dx)} and x = {_f(a + lim.h0, lim.dx)} "
                         f"and slide in to x = {_f(a - lim.h_end, lim.dx)} and x = {_f(a + lim.h_end, lim.dx)}. "
                         f"Both f(x) readouts end on {_f(lim.L, lim.dy)}."],
            "name": [f"The screen shows: the limit of f(x) as x approaches {_g(a)} {shown}. "
                     f"f({_g(a)}) is still undefined."],
        }, lim.L
    if component == "RiemannSum":
        # The component's own numbers (screen_numbers): the gap is the red area
        # drawn, which is |integral - sum| only when every sliver is on one side.
        from chalkdust.scenes.components.riemann_sum import screen_numbers
        a, b, rule, n0 = params.a, params.b, params.rule, params.n0
        nums = screen_numbers(params)
        d, levels, sums, gaps = nums.decimals, nums.levels, nums.sums, nums.gaps
        where = {"left": "left edge", "right": "right edge", "mid": "midpoint"}[rule]
        return {
            "pose": [f"The area under the curve from x = {_g(a)} to x = {_g(b)} is shaded; the screen "
                     "asks 'area = ?'. No number is shown yet."],
            "boxes": [f"{n0} rectangles, each {_g(round((b - a) / n0, 6))} wide and as tall as f at its "
                      f"{where}. The {rule} sum reads {_f(sums[0], d)}. The red slivers between the "
                      f"rectangles and the curve have total area {_f(gaps[0], d)} (the gap readout)."],
            "refine": [f"Each rectangle splits in two, {len(levels) - 1} times: n = "
                       f"{', '.join(map(str, levels))}. The sum reads "
                       f"{', '.join(_f(v, d) for v in sums)}; the red gap reads "
                       f"{', '.join(_f(v, d) for v in gaps)}."],
            "name": [f"The screen shows the sum over {levels[-1]} rectangles = {_f(sums[-1], d)}, then: "
                     f"the limit of the sums as n goes to infinity = the definite integral from {_g(a)} "
                     f"to {_g(b)} of f = {_f(nums.true, d)}."],
        }, nums.true
    raise KeyError(f"no facts for {component}")


# --- the calculator ----------------------------------------------------------------------

_X = sp.Symbol("x")
CALCULUS = {"derivative": 2, "limit": 2, "integral": 3}


def calc_kind(expr: str) -> str | None:
    """'derivative' | 'limit' | 'integral' if expr is one such call, else None."""
    try:
        node = ast.parse(str(expr).replace("^", "**"), mode="eval").body
    except SyntaxError:
        return None
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in CALCULUS:
        return node.func.id
    return None


def calc_value(expr: str) -> float | None:
    """A model-written answer expression, evaluated without eval: plain
    arithmetic (research.safe_math), or ONE call derivative(f, a) /
    limit(f, a) / integral(f, a, b) with f in x. None if it is neither or
    does not evaluate to one real number."""
    if not expr or not isinstance(expr, str):
        return None
    try:
        kind = calc_kind(expr)
        if kind is None:
            v = safe_sympify(expr)
        else:
            call = ast.parse(expr.replace("^", "**"), mode="eval").body
            if len(call.args) != CALCULUS[kind] or call.keywords:
                return None
            f = safe_sympify(ast.unparse(call.args[0]), {"x": _X})
            pts = [safe_sympify(ast.unparse(a)) for a in call.args[1:]]
            if any(p.free_symbols for p in pts):
                return None
            if kind == "derivative":
                v = sp.diff(f, _X).subs(_X, pts[0])
            elif kind == "limit":
                v = sp.limit(f, _X, pts[0], dir="+-")
            else:
                v = sp.Integral(f, (_X, pts[0], pts[1])).evalf(30)
        if v.free_symbols:
            return None
        c = complex(sp.N(v))
        if abs(c.imag) > 1e-9 or not math.isfinite(c.real):
            return None
        return float(c.real)
    except (UnsafeExpression, SyntaxError, ValueError, TypeError, ZeroDivisionError,
            NotImplementedError, AttributeError, RecursionError):
        return None


def answer_text(v: float) -> tuple[str, str]:
    """(LaTeX, plain) for the computed answer: up to 4 significant figures,
    with an approx sign when that rounds."""
    r = float(f"{v:.4g}")
    s = _g(r)
    approx = abs(r - v) > 1e-9 * max(1.0, abs(v))
    return (rf"\approx {s}" if approx else s), (f"about {s}" if approx else s)


def answerbox_value(model_value: Any, v: float) -> str:
    """The AnswerBox value with the computed number: keep the model's left
    side (e.g. "f'(1)") when it wrote "lhs = ...", else just the number."""
    tex, _ = answer_text(v)
    lhs = str(model_value).split("=")[0].strip() if isinstance(model_value, str) and "=" in model_value else ""
    if tex.startswith(r"\approx"):
        return f"{lhs} {tex}".strip()
    return f"{lhs} = {tex}" if lhs else tex


def says(narration: str, v: float) -> bool:
    """True if the narration speaks v (to the precision of 4 significant
    figures, or as an integer when v is one)."""
    from research.content import spoken_numbers
    tol = max(abs(v) * 1e-3, 1e-9) if not float(v).is_integer() else 1e-9
    return any(abs(s - v) <= tol or abs(s - float(f"{v:.4g}")) <= 1e-9 for s in spoken_numbers(narration))
