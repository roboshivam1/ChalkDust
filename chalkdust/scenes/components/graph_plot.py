"""GraphPlot: functions drawn on themed axes, with traced markers.

The spec carries functions as *expressions*, never code. An expression is a
restricted arithmetic string in the single variable `x` -- "x^2 - 2*x",
"sin(x)/x", "exp(-x^2/2)" -- parsed with Python's `ast` and walked against an
allowlist. Nothing is ever handed to eval, so a spec cannot execute anything:
the LLM does not write code, including inside a parameter (SCENE_SPEC.md §1).

Everything the viewer reads is generated from the parsed tree. The legend's
LaTeX is printed from the AST, so it is valid by construction and the spec has
no LaTeX surface to break (SCENE_SPEC.md §3: what the model cannot set, it
cannot break). Tick labels are theme text, except m x 10^n labels on huge or
tiny axes, which are LaTeX generated from the tick value (see tick_label).

Sampling, the visible y-window and discontinuity detection are pure numpy and
run at schema validation, so "undefined everywhere on x_range" or "marker sits
on an asymptote" fails as a param error -- rung 1, the cheapest place to catch
it (SCENE_SPEC.md §8) -- before anything builds.

Params beyond the §5 key params: `y_range` (optional). The automatic window
handles asymptotes by trimming heavy tails, but only the author knows that a
beat about tan(x) wants to show -5..5; without it the component would have to
guess, and a guess that crops the point of the beat is a broken render.
"""

from __future__ import annotations

import ast
import math as _math
from dataclasses import dataclass
from functools import lru_cache
from typing import Annotated, Callable

import numpy as np
from manim import (
    DL,
    DOWN,
    DR,
    LEFT,
    RIGHT,
    UL,
    UP,
    UR,
    Create,
    Dot,
    FadeIn,
    Line,
    MoveAlongPath,
    VGroup,
    VMobject,
)
from pydantic import Field, field_validator, model_validator

from chalkdust.core.models import Region
from chalkdust.scenes.base import ChalkdustScene
from chalkdust.scenes.components.base import (
    Component,
    ComponentParams,
    label,
    register,
    wrap,
)
from chalkdust.scenes.regions import (
    DEFAULT_PADDING,
    LayoutError,
    Rect,
    bbox,
    fit_to_region,
    region_rect,
)
from chalkdust.scenes.theme import Theme, caption_text, math

# --- expression language -----------------------------------------------------
#
# A deliberately small language. Every name here is something a maths explainer
# plots; anything else is a schema error that names the allowlist, so the
# repair loop's regenerate step (SCENE_SPEC.md §9) knows what it may use.

MAX_EXPR_LEN = 120  # bounds parse cost; nothing this long fits a legend anyway

# name -> (vectorised numpy function, LaTeX command or None if printed specially)
_FUNCS: dict[str, tuple[Callable[[np.ndarray], np.ndarray], str | None]] = {
    "sin": (np.sin, r"\sin"),
    "cos": (np.cos, r"\cos"),
    "tan": (np.tan, r"\tan"),
    "asin": (np.arcsin, r"\arcsin"),
    "acos": (np.arccos, r"\arccos"),
    "atan": (np.arctan, r"\arctan"),
    "sinh": (np.sinh, r"\sinh"),
    "cosh": (np.cosh, r"\cosh"),
    "tanh": (np.tanh, r"\tanh"),
    "exp": (np.exp, None),
    # `log` is the natural log, as in Python and most maths writing; it prints
    # as \ln so the viewer is never left guessing the base.
    "log": (np.log, r"\ln"),
    "ln": (np.log, r"\ln"),
    "log10": (np.log10, r"\log_{10}"),
    "log2": (np.log2, r"\log_{2}"),
    "sqrt": (np.sqrt, None),
    "abs": (np.abs, None),
}
_CONSTS: dict[str, tuple[float, str]] = {"pi": (np.pi, r"\pi"), "e": (np.e, "e")}
_BINOPS: dict[type, Callable] = {
    ast.Add: np.add,
    ast.Sub: np.subtract,
    ast.Mult: np.multiply,
    ast.Div: np.divide,
    ast.Pow: np.power,
}

_GRAMMAR_HINT = (
    "expressions use the variable x, numbers, + - * / ^, parentheses, the "
    f"constants {', '.join(_CONSTS)} and the functions {', '.join(_FUNCS)}; "
    "multiplication must be explicit (2*x, not 2x)"
)


@lru_cache(maxsize=256)
def _parse(expr: str) -> ast.expr:
    """Parse and allowlist-check an expression. Raises ValueError.

    `ast.parse` only builds a tree; it never executes. `^` is read as power
    because that is what an expression author means by it (Python's XOR is
    not in the allowlist, so nothing is lost).
    """
    try:
        tree = ast.parse(expr.replace("^", "**"), mode="eval")
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        raise ValueError(f"cannot parse expression {expr!r}: {_GRAMMAR_HINT}") from None
    _check(tree.body, expr)
    return tree.body


def _check(node: ast.AST, expr: str) -> None:
    def bad(what: str) -> ValueError:
        return ValueError(f"{what} in expression {expr!r}: {_GRAMMAR_HINT}")

    if isinstance(node, ast.Constant):
        # type() not isinstance(): bool is an int subclass, and True is not a number.
        if type(node.value) not in (int, float) or not _math.isfinite(node.value):
            raise bad(f"unsupported literal {node.value!r}")
    elif isinstance(node, ast.Name):
        if node.id != "x" and node.id not in _CONSTS:
            raise bad(f"unknown name {node.id!r}")
    elif isinstance(node, ast.BinOp):
        if type(node.op) not in _BINOPS:
            raise bad(f"unsupported operator {type(node.op).__name__}")
        _check(node.left, expr)
        _check(node.right, expr)
    elif isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, (ast.USub, ast.UAdd)):
            raise bad(f"unsupported operator {type(node.op).__name__}")
        _check(node.operand, expr)
    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS:
            raise bad("unknown function")
        if node.keywords or len(node.args) != 1 or isinstance(node.args[0], ast.Starred):
            raise bad(f"{node.func.id}() takes exactly one argument")
        _check(node.args[0], expr)
    else:
        raise bad(f"unsupported syntax ({type(node).__name__})")


def _eval(node: ast.expr, x: np.ndarray) -> np.ndarray:
    """Evaluate a checked tree over `x`. Constants are float64, never Python
    ints: 9^9^9 must overflow to inf instantly, not compute a huge integer."""
    if isinstance(node, ast.Constant):
        return np.float64(node.value)
    if isinstance(node, ast.Name):
        return x if node.id == "x" else np.float64(_CONSTS[node.id][0])
    if isinstance(node, ast.BinOp):
        return _BINOPS[type(node.op)](_eval(node.left, x), _eval(node.right, x))
    if isinstance(node, ast.UnaryOp):
        v = _eval(node.operand, x)
        return np.negative(v) if isinstance(node.op, ast.USub) else v
    assert isinstance(node, ast.Call)  # _check admits nothing else
    return _FUNCS[node.func.id][0](_eval(node.args[0], x))


# Values beyond this are treated as off-scale rather than plotted. Keeps every
# span finite: two samples near +-1e308 would otherwise overflow a subtraction.
_OFF_SCALE = 1e100


def evaluate(expr: str, x: np.ndarray) -> np.ndarray:
    """f(x) as float64. Out-of-domain points come back as nan, never raise."""
    x = np.asarray(x, dtype=float)
    with np.errstate(all="ignore"):
        y = np.broadcast_to(_eval(_parse(expr), x), x.shape).astype(float)
        y[np.abs(y) > _OFF_SCALE] = np.nan
    return y


# --- LaTeX printer -----------------------------------------------------------
# Precedence levels for parenthesisation. Division prints as \frac, which binds
# like an atom except as a power base, where (a/b)^2 still needs brackets.

_P_ADD, _P_MUL, _P_NEG, _P_POW, _P_ATOM = 1, 2, 3, 4, 5


def _tex_number(v: int | float) -> str:
    if float(v).is_integer() and abs(v) < 1e15:
        return str(int(v))
    s = format(float(v), ".6g")
    if "e" in s:
        mant, exp = s.split("e")
        return rf"{mant} \times 10^{{{int(exp)}}}"
    return s


def _paren(s: str) -> str:
    return rf"\left({s}\right)"


def _tex(node: ast.expr) -> tuple[str, int]:
    if isinstance(node, ast.Constant):
        return _tex_number(node.value), _P_ATOM
    if isinstance(node, ast.Name):
        return ("x" if node.id == "x" else _CONSTS[node.id][1]), _P_ATOM
    if isinstance(node, ast.UnaryOp):
        s, p = _tex(node.operand)
        if isinstance(node.op, ast.UAdd):
            return s, p
        return "-" + (_paren(s) if p <= _P_ADD else s), _P_NEG
    if isinstance(node, ast.Call):
        name = node.func.id
        arg, _ = _tex(node.args[0])
        if name == "sqrt":
            return rf"\sqrt{{{arg}}}", _P_ATOM
        if name == "abs":
            return rf"\left|{arg}\right|", _P_ATOM
        if name == "exp":
            return rf"e^{{{arg}}}", _P_POW
        # Plain brackets unless the argument is tall: \left( is an "inner"
        # atom, and TeX puts operator spacing before it -- "sin (x)".
        tall = r"\frac" in arg
        return _FUNCS[name][1] + (_paren(arg) if tall else f"({arg})"), _P_ATOM
    assert isinstance(node, ast.BinOp)
    (ls, lp), (rs, rp) = _tex(node.left), _tex(node.right)
    op = type(node.op)
    if op is ast.Div:
        return rf"\frac{{{ls}}}{{{rs}}}", _P_POW
    if op is ast.Pow:
        base_needs = lp < _P_ATOM or isinstance(node.left, (ast.Call, ast.BinOp))
        return rf"{_paren(ls) if base_needs else ls}^{{{rs}}}", _P_POW
    if op in (ast.Add, ast.Sub):
        # a - (b + c), and a + (-b) rather than the unreadable "a + -b".
        if rp == _P_NEG or (op is ast.Sub and rp <= _P_ADD):
            rs = _paren(rs)
        return f"{ls} {'+' if op is ast.Add else '-'} {rs}", _P_ADD
    # Multiplication: "2x", "3\sin(x)" by juxtaposition when the left factor is
    # a number and the right one starts with a letter; a \cdot otherwise.
    if lp < _P_MUL:
        ls = _paren(ls)
    if rp <= _P_NEG:
        rs = _paren(rs)
    left = node.left.operand if isinstance(node.left, ast.UnaryOp) else node.left
    juxtapose = isinstance(left, ast.Constant) and not rs[0].isdigit()
    return (f"{ls} {rs}" if juxtapose else rf"{ls} \cdot {rs}"), _P_MUL


def legend_tex(expr: str, name: str | None) -> str:
    """The legend line for one function, e.g. "f(x) = x^{2} - 1"."""
    lhs = f"{name}(x)" if name else "y"
    return f"{lhs} = {_tex(_parse(expr))[0]}"


# --- sampling and the visible window ------------------------------------------

N_SAMPLES = 801
# A step between neighbouring samples larger than this fraction of the window
# height is a *candidate* discontinuity; bisection decides (see _is_jump).
JUMP_FRACTION = 0.25
# If the full value range is this many times the 10-90 percentile spread, the
# extremes are asymptote spikes, not shape: trim to the 5-95 band instead.
HEAVY_TAIL = 6.0
WINDOW_PAD = 0.08  # breathing room above/below the curves, fraction of span

RANGE_LIMIT = 1e12      # largest |value| an explicit range may name
MIN_RELATIVE_SPAN = 1e-3  # narrower than this and tick labels cannot differ


@dataclass(frozen=True)
class _Plan:
    """Everything build() needs that does not depend on the scene."""

    x0: float
    x1: float
    y0: float
    y1: float
    # Per function: drawable runs, each an (n, 2) array in data coordinates,
    # already clipped to the window and split at discontinuities.
    runs: tuple[tuple[np.ndarray, ...], ...]
    marker_ys: tuple[float, ...]


def _is_jump(expr: str, xa: float, ya: float, xb: float, yb: float,
             threshold: float) -> bool:
    """True if f is discontinuous between two samples.

    Bisect towards the half carrying the larger change. Across a continuous
    stretch the change shrinks with the interval; across an asymptote or a
    jump it never does. Steep-but-continuous curves therefore stay connected
    while tan(x) is split at every pole.
    """
    for _ in range(48):
        xm = 0.5 * (xa + xb)
        if xm in (xa, xb):
            break
        ym = float(evaluate(expr, np.array([xm]))[0])
        if not np.isfinite(ym):
            return True
        if abs(ym - ya) >= abs(yb - ym):
            xb, yb = xm, ym
        else:
            xa, ya = xm, ym
    return abs(yb - ya) > threshold / 2


def _crossing(xa: float, ya: float, xb: float, yb: float, edge: float) -> tuple[float, float]:
    t = (edge - ya) / (yb - ya)
    return xa + t * (xb - xa), edge


def _runs(expr: str, xs: np.ndarray, ys: np.ndarray, y0: float, y1: float) -> list[np.ndarray]:
    """Split one sampled function into drawable runs inside [y0, y1].

    A run ends where f is undefined, where it jumps, or where it leaves the
    window -- in which case the run is extended exactly to the window edge so
    curves meet the frame instead of stopping a sample short of it.
    """
    threshold = JUMP_FRACTION * (y1 - y0)
    runs: list[np.ndarray] = []
    cur: list[tuple[float, float]] = []

    def flush() -> None:
        if len(cur) >= 2:
            runs.append(np.array(cur))
        cur.clear()

    def inside(y: float) -> bool:
        return y0 <= y <= y1

    def edge(y: float) -> float:
        return y1 if y > y1 else y0

    for i, (x, y) in enumerate(zip(xs.tolist(), ys.tolist())):
        if i == 0 or not (np.isfinite(y) and np.isfinite(ys[i - 1])):
            flush()
            if np.isfinite(y) and inside(y):
                cur.append((x, y))
            continue
        xa, ya = float(xs[i - 1]), float(ys[i - 1])
        if abs(y - ya) > threshold and _is_jump(expr, xa, ya, x, y, threshold):
            flush()
            if inside(y):
                cur.append((x, y))
            continue
        a_in, b_in = inside(ya), inside(y)
        if a_in and b_in:
            cur.append((x, y))
        elif a_in:
            cur.append(_crossing(xa, ya, x, y, edge(y)))
            flush()
        elif b_in:
            flush()
            cur.extend([_crossing(xa, ya, x, y, edge(ya)), (x, y)])
        elif edge(ya) != edge(y):
            # Passed clean through the window between two samples.
            flush()
            runs.append(np.array([_crossing(xa, ya, x, y, edge(ya)),
                                  _crossing(xa, ya, x, y, edge(y))]))
    flush()
    return runs


def _auto_window(values: np.ndarray) -> tuple[float, float]:
    """Pick a y-window that shows the shape, not the asymptote spikes."""
    lo, hi = float(values.min()), float(values.max())
    q05, q10, q90, q95 = np.percentile(values, [5, 10, 90, 95])
    if hi - lo > HEAVY_TAIL * max(q90 - q10, 1e-300):
        lo, hi = float(q05), float(q95)

    magnitude = max(abs(lo), abs(hi))
    if hi - lo <= MIN_RELATIVE_SPAN * magnitude or hi - lo == 0:
        # Flat at this scale. Draw a level line mid-window rather than
        # magnifying float noise into a shape.
        mid = 0.5 * (lo + hi)
        half = max(0.5 * abs(mid), 1.0)
        return mid - half, mid + half

    # Pull the x-axis into view when it is nearly there anyway.
    if 0 < lo < 0.25 * hi:
        lo = 0.0
    elif 0.25 * lo < hi < 0:
        hi = 0.0
    pad = WINDOW_PAD * (hi - lo)
    return (lo if lo == 0 else lo - pad), (hi if hi == 0 else hi + pad)


@lru_cache(maxsize=64)
def _plan(exprs: tuple[str, ...], x_range: tuple[float, float],
          y_range: tuple[float, float] | None,
          markers: tuple[tuple[float, int], ...]) -> _Plan:
    """Sample, choose the window, split into runs. Raises ValueError with a
    message aimed at the spec author (or the repair loop)."""
    x0, x1 = x_range
    xs = np.linspace(x0, x1, N_SAMPLES)
    samples = [evaluate(e, xs) for e in exprs]
    for i, (e, ys) in enumerate(zip(exprs, samples)):
        if not np.isfinite(ys).any():
            raise ValueError(f"functions[{i}] {e!r} is undefined everywhere on x_range {list(x_range)}")

    marker_ys = []
    for j, (mx, fi) in enumerate(markers):
        my = float(evaluate(exprs[fi], np.array([mx]))[0])
        if not np.isfinite(my):
            raise ValueError(f"markers[{j}] at x={mx}: functions[{fi}] {exprs[fi]!r} is undefined there")
        marker_ys.append(my)

    if y_range is not None:
        y0, y1 = y_range
        for j, my in enumerate(marker_ys):
            if not y0 <= my <= y1:
                raise ValueError(f"markers[{j}] sits at y={my:.4g}, outside y_range {list(y_range)}")
    else:
        finite = np.concatenate([ys[np.isfinite(ys)] for ys in samples])
        y0, y1 = _auto_window(finite)
        if marker_ys:
            # A marker is the point of the beat; never crop it.
            y0, y1 = min(y0, *marker_ys), max(y1, *marker_ys)

    runs = []
    for i, (e, ys) in enumerate(zip(exprs, samples)):
        r = _runs(e, xs, ys, y0, y1)
        if not r:
            raise ValueError(f"functions[{i}] {e!r} never enters the visible y-window [{y0:.4g}, {y1:.4g}]")
        runs.append(tuple(r))
    return _Plan(x0, x1, y0, y1, tuple(runs), tuple(marker_ys))


# --- ticks ---------------------------------------------------------------------

X_TICK_TARGET = 8
Y_TICK_TARGET = 5


def _nice_step(span: float, target: int) -> float:
    """A 1-2-5 step giving roughly `target` intervals. Rounds to the nearest
    nice step rather than up: rounding up can halve the tick count, which on
    a short y-axis leaves one lonely label. Crowding is handled by thinning."""
    raw = span / target
    mag = 10.0 ** _math.floor(_math.log10(raw))
    for m in (1, 2, 5, 10):
        if m * mag >= raw * 0.75:
            return m * mag
    return 10 * mag  # unreachable; keeps the type checker honest


def _ticks(lo: float, hi: float, step: float) -> list[tuple[int, float]]:
    """(k, k*step) for every multiple of step inside [lo, hi]."""
    k0 = _math.ceil(lo / step - 1e-9)
    k1 = _math.floor(hi / step + 1e-9)
    return [(k, k * step) for k in range(k0, k1 + 1)]


def tick_label(v: float, step: float, magnitude: float) -> tuple[str, bool]:
    """(label, is_latex). Huge or tiny axes switch to m x 10^n so a label never
    grows wider than the space between ticks. Those go through LaTeX: plain
    text would need a superscript minus, which fallback fonts lack (it renders
    as a missing-glyph box)."""
    sci = magnitude >= 1e5 or magnitude < 1e-3
    if abs(v) < step * 1e-6:
        return "0", sci
    if sci:
        v = round(v / step) * step
        n = _math.floor(_math.log10(abs(v)) + 1e-9)
        digits = min(max(n - _math.floor(_math.log10(step) + 1e-9), 0), 4)
        mant = f"{v / 10.0 ** n:.{digits}f}"
        if "." in mant:
            mant = mant.rstrip("0").rstrip(".")
        return rf"{mant} \times 10^{{{n}}}", True
    s = f"{v:.{max(0, -_math.floor(_math.log10(step) + 1e-9))}f}"
    return ("−" + s[1:] if s.startswith("-") else s), False


def axis_ticks(lo: float, hi: float, target: int) -> tuple[float, list[tuple[int, float, str, bool]]]:
    """The step, and (k, value, label, is_latex) for every tick on one axis."""
    step = _nice_step(hi - lo, target)
    magnitude = max(abs(lo), abs(hi))
    return step, [(k, v, *tick_label(v, step, magnitude)) for k, v in _ticks(lo, hi, step)]


# --- params ----------------------------------------------------------------------

FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]


def _check_range(v: tuple[float, float] | None, what: str) -> tuple[float, float] | None:
    if v is None:
        return v
    lo, hi = v
    if not lo < hi:
        raise ValueError(f"{what} must be [low, high] with low < high, got {list(v)}")
    magnitude = max(abs(lo), abs(hi))
    if magnitude > RANGE_LIMIT:
        raise ValueError(f"{what} {list(v)} exceeds +-{RANGE_LIMIT:g}")
    if hi - lo < MIN_RELATIVE_SPAN * magnitude:
        raise ValueError(
            f"{what} {list(v)} is too narrow for its magnitude to label distinct "
            f"ticks; shift the variable instead (plot f(x + c) near 0)")
    return v


class GraphFunction(ComponentParams):
    expr: str = Field(min_length=1, max_length=MAX_EXPR_LEN)
    # Optional single-letter name: the legend reads "f(x) = ..." instead of
    # "y = ...", so narration can say "f" and the viewer can find it.
    name: str | None = Field(default=None, pattern=r"^[A-Za-z]$")

    @field_validator("expr")
    @classmethod
    def _parses(cls, v: str) -> str:
        _parse(v)
        return v


class GraphMarker(ComponentParams):
    x: FiniteFloat
    # Index into `functions`; the marker sits on that curve at x.
    function: int = Field(default=0, ge=0)
    label: str | None = Field(default=None, min_length=1, max_length=80)


class GraphPlotParams(ComponentParams):
    # Capped at 3: one per emphasis colour in the palette (accent, accent_alt,
    # success). A fourth curve would have to reuse a colour, and two curves the
    # viewer cannot tell apart are worse than one fewer.
    functions: list[GraphFunction] = Field(min_length=1, max_length=3)
    x_range: tuple[FiniteFloat, FiniteFloat]
    # Optional explicit window; see the module docstring for why it exists.
    y_range: tuple[FiniteFloat, FiniteFloat] | None = None
    # Each marker is traced to in turn; more than four and the beat is a table.
    markers: list[GraphMarker] = Field(default_factory=list, max_length=4)

    @field_validator("x_range")
    @classmethod
    def _x_ok(cls, v):
        return _check_range(v, "x_range")

    @field_validator("y_range")
    @classmethod
    def _y_ok(cls, v):
        return _check_range(v, "y_range")

    @model_validator(mode="after")
    def _plannable(self) -> GraphPlotParams:
        for j, m in enumerate(self.markers):
            if m.function >= len(self.functions):
                raise ValueError(f"markers[{j}].function={m.function} but only "
                                 f"{len(self.functions)} function(s) given")
            if not self.x_range[0] <= m.x <= self.x_range[1]:
                raise ValueError(f"markers[{j}].x={m.x} is outside x_range {list(self.x_range)}")
        plan_for(self)  # raises ValueError -> pydantic ValidationError
        return self


def plan_for(p: GraphPlotParams) -> _Plan:
    return _plan(tuple(f.expr for f in p.functions), tuple(p.x_range),
                 tuple(p.y_range) if p.y_range else None,
                 tuple((m.x, m.function) for m in p.markers))


# --- the component ---------------------------------------------------------------

# Layout, in Manim units.
LEGEND_MAX_FRACTION = 0.42  # of stage width; past this the plot gets cramped
LEGEND_GAP = 0.45           # between plot block and legend column
SWATCH_LEN = 0.45
TICK_LEN = 0.12
LABEL_GAP = 0.1             # tick to tick label
LABEL_SEP = 0.2             # minimum clear space between neighbouring tick labels
MARKER_BUFF = 0.12
INK_CLEARANCE = 0.05       # clear space a label wants from any drawn line
INK_STEP = 0.02            # max spacing of the points that stand in for curve ink
MIN_PLOT_SIZE = 1.5         # smaller than this is not a graph, it is a doodle
AXIS_STROKE = 2.0
CURVE_STROKE = 4.0
DOT_RADIUS = 0.08

# Timing weights (relative) and per-step legibility minimums (seconds).
W_AXES, W_CURVE, W_TRACE, W_LABEL, W_HOLD = 1.0, 2.5, 1.5, 0.75, 2.0
MIN_AXES, MIN_CURVE, MIN_TRACE, MIN_LABEL, MIN_HOLD = 0.4, 0.8, 0.5, 0.3, 0.5


def _ink(paths: list[np.ndarray]) -> np.ndarray:
    """Points along polylines, at most INK_STEP apart.

    Sample points alone under-report: a steep run near an asymptote can cross
    a whole label between two samples. Each path is densified on its own, so
    nothing is ever bridged across a pole.
    """
    out = []
    for pts in paths:
        seg = np.diff(pts, axis=0)
        n = np.maximum(np.ceil(np.linalg.norm(seg, axis=1) / INK_STEP), 1).astype(int)
        idx = np.repeat(np.arange(len(seg)), n)
        frac = (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)) / np.repeat(n, n)
        out += [pts[idx] + seg[idx] * frac[:, None], pts[-1:]]
    return np.vstack(out)


def _inked(ink: np.ndarray, box: Rect, pad: float = INK_CLEARANCE) -> bool:
    """True if any ink point falls inside `box` grown by `pad`."""
    return bool(((ink[:, 0] > box.left - pad) & (ink[:, 0] < box.right + pad)
                 & (ink[:, 1] > box.bottom - pad) & (ink[:, 1] < box.top + pad)).any())


def _curve_colours(theme: Theme) -> tuple[str, str, str]:
    p = theme.palette
    return p.accent, p.accent_alt, p.success


@register
class GraphPlot(Component):
    name = "GraphPlot"
    Params = GraphPlotParams

    def regions(self) -> set[Region]:
        # Plot plus legend column span the whole stage.
        return {Region.STAGE}

    # --- semantic-rung hooks (SCENE_SPEC.md §8 rung 2) ---------------------

    def latex_strings(self) -> list[str]:
        plan = plan_for(self.params)
        ticks = (axis_ticks(plan.x0, plan.x1, X_TICK_TARGET)[1]
                 + axis_ticks(plan.y0, plan.y1, Y_TICK_TARGET)[1])
        return ([legend_tex(f.expr, f.name) for f in self.params.functions]
                + [s for *_, s, tex in ticks if tex])

    def min_seconds(self) -> float:
        p: GraphPlotParams = self.params
        return (MIN_AXES + MIN_CURVE * len(p.functions) + MIN_HOLD
                + sum(MIN_TRACE + (MIN_LABEL if m.label else 0) for m in p.markers))

    # --- build -------------------------------------------------------------

    def build(self, scene: ChalkdustScene) -> None:
        p: GraphPlotParams = self.params
        theme = scene.theme
        plan = plan_for(p)
        colours = _curve_colours(theme)
        stage = region_rect(Region.STAGE).inset(DEFAULT_PADDING)

        # Legend first: its width decides how much is left for the plot.
        entries = []
        for i, tex in enumerate(legend_tex(f.expr, f.name) for f in p.functions):
            swatch = Line(LEFT * SWATCH_LEN / 2, RIGHT * SWATCH_LEN / 2,
                          color=colours[i], stroke_width=CURVE_STROKE)
            entries.append(label(VGroup(swatch, math(tex, theme, size=theme.type.body))
                                 .arrange(RIGHT, buff=0.2), f"legend[{i}]"))
        legend = label(VGroup(*entries).arrange(DOWN, aligned_edge=LEFT, buff=0.3), "legend")
        col_w = min(legend.width, LEGEND_MAX_FRACTION * stage.width)
        # Raises LayoutError(overflow) if a long expression would need type
        # below the legibility floor -- the beat should split, not shrink.
        fit_to_region(legend, Rect(stage.right - col_w / 2, stage.y, col_w, stage.height),
                      padding=0, align=UP)

        axes, plot, to_point, curve_ink = self._axes(plan, theme, stage,
                                                     stage.right - col_w - LEGEND_GAP)
        axes = label(axes, "axes")

        curves = []
        for i, runs in enumerate(plan.runs):
            curve = VMobject(stroke_color=colours[i], stroke_width=CURVE_STROKE)
            for run in runs:
                pts = to_point(run[:, 0], run[:, 1])
                curve.start_new_path(pts[0])
                curve.add_points_as_corners(pts[1:])
            curves.append(label(curve, f"curve[{i}]"))

        # Where marker labels would rather not sit: on a curve or an axis line.
        x_axis, y_axis = axes.submobjects[:2]
        ink = np.vstack([curve_ink] + [
            np.linspace(line.get_start(), line.get_end(), 200) for line in (x_axis, y_axis)])
        markers = self._markers(plan, theme, plot, to_point, ink, obstacles=[
            bbox(m) for m in axes.submobjects if getattr(m, "_chalk_font_size", None)])

        # --- animate: axes, each curve with its legend entry, each marker
        # traced in turn, then hold. Weights become seconds via budget(), so the
        # beat lasts exactly as long as its narration (D-002).
        weights = [W_AXES] + [W_CURVE] * len(curves)
        for _, mlabel, _ in markers:
            weights += [W_TRACE] + ([W_LABEL] if mlabel is not None else [])
        times = iter(scene.budget(*weights, W_HOLD))

        legend.set_opacity(0)
        scene.add(legend)
        scene.exclusive(legend, axes)

        scene.play(FadeIn(axes), run_time=next(times))
        for curve, entry in zip(curves, entries):
            scene.play(Create(curve), entry.animate.set_opacity(1), run_time=next(times))
        scene.settle("curves drawn")

        for dot, mlabel, path in markers:
            scene.add(dot)
            if path is None:
                scene.play(FadeIn(dot), run_time=next(times))
            else:
                scene.play(MoveAlongPath(dot, path), run_time=next(times))
            if mlabel is not None:
                scene.play(FadeIn(mlabel), run_time=next(times))
        scene.settle("graph traced")
        scene.wait(next(times))

    def _axes(self, plan: _Plan, theme: Theme, stage: Rect, right: float):
        """Axes, ticks and tick labels sized to the space left of the legend.

        Built at final size rather than built large and fitted down: tick
        labels are caption text, only a hair above the legibility floor, so
        they cannot survive any real downscale.

        The plot rect depends only on label *sizes*, so the curves' ink is
        known before any label is kept, and a label a curve runs through is
        dropped exactly like one the other axis runs through: muted caption
        text under a curve stroke reads wrong ("-2" with its minus hidden is
        "2"). Also returns that ink, for marker-label placement.
        """
        def tick_mob(s: str, tex: bool):
            if tex:
                return math(s, theme, size=theme.type.caption, color=theme.palette.muted)
            return caption_text(s, theme)

        xstep, xt = axis_ticks(plan.x0, plan.x1, X_TICK_TARGET)
        ystep, yt = axis_ticks(plan.y0, plan.y1, Y_TICK_TARGET)
        xticks = [(k, v, tick_mob(s, tex)) for k, v, s, tex in xt]
        yticks = [(k, v, tick_mob(s, tex)) for k, v, s, tex in yt]

        max_xw = max((t.width for *_, t in xticks), default=0.0)
        max_xh = max((t.height for *_, t in xticks), default=0.0)
        max_yw = max((t.width for *_, t in yticks), default=0.0)
        max_yh = max((t.height for *_, t in yticks), default=0.0)

        # Reserve margins on every side a label could poke past the plot rect.
        left = stage.left + max(max_yw + TICK_LEN / 2 + LABEL_GAP, max_xw / 2)
        right = right - max_xw / 2
        bottom = stage.bottom + max_xh + TICK_LEN / 2 + LABEL_GAP
        top = stage.top - max_yh / 2
        if right - left < MIN_PLOT_SIZE or top - bottom < MIN_PLOT_SIZE:
            raise LayoutError(
                f"GraphPlot: only {right - left:.2f} x {top - bottom:.2f} left for the "
                "plot after legend and tick labels. Shorten the expressions or split "
                "the beat.", kind="overflow")
        plot = Rect((left + right) / 2, (bottom + top) / 2, right - left, top - bottom)
        sx = plot.width / (plan.x1 - plan.x0)
        sy = plot.height / (plan.y1 - plan.y0)

        def to_point(xd, yd) -> np.ndarray:
            """Data coordinates -> scene points, relative to the plot rect."""
            xd, yd = np.atleast_1d(xd), np.atleast_1d(yd)
            return np.column_stack([plot.left + (xd - plan.x0) * sx,
                                    plot.bottom + (yd - plan.y0) * sy,
                                    np.zeros_like(xd, dtype=float)])

        # Axes cross at the origin when it is in view, otherwise sit on the
        # edge nearest to it -- the textbook convention (and Manim's).
        ax_y = min(max(0.0, plan.y0), plan.y1)
        ax_x = min(max(0.0, plan.x0), plan.x1)
        stroke = dict(color=theme.palette.muted, stroke_width=AXIS_STROKE)
        # Curve ink in scene units: a tick label it runs through is dropped.
        ink = _ink([to_point(r[:, 0], r[:, 1]) for runs in plan.runs for r in runs])

        x_axis = Line(to_point(plan.x0, ax_y)[0], to_point(plan.x1, ax_y)[0], **stroke)
        y_axis = Line(to_point(ax_x, plan.y0)[0], to_point(ax_x, plan.y1)[0], **stroke)
        parts = [x_axis, y_axis]

        # Thin the labels until neighbours have clear space between them.
        def stride(pitch: float, need: float) -> int:
            return next((s for s in (1, 2, 5, 10, 20, 50) if s * pitch >= need), 100)

        xs = stride(xstep * sx, max_xw + LABEL_SEP)
        ys = stride(ystep * sy, max_yh + LABEL_SEP)

        kept_y = []
        for k, v, text in yticks:
            at = to_point(ax_x, v)[0]
            parts.append(Line(at + LEFT * TICK_LEN / 2, at + RIGHT * TICK_LEN / 2, **stroke))
            text.next_to(at, LEFT, buff=TICK_LEN / 2 + LABEL_GAP)
            # A label the x-axis runs through is unreadable; the origin gets none.
            box = bbox(text)
            if k % ys == 0 and not box.intersects(bbox(x_axis)) and not _inked(ink, box):
                kept_y.append(text)
        kept_x = []
        for k, v, text in xticks:
            at = to_point(v, ax_y)[0]
            parts.append(Line(at + DOWN * TICK_LEN / 2, at + UP * TICK_LEN / 2, **stroke))
            text.next_to(at, DOWN, buff=TICK_LEN / 2 + LABEL_GAP)
            box = bbox(text)
            if (k % xs == 0 and not box.intersects(bbox(y_axis)) and not _inked(ink, box)
                    and not any(box.intersects(bbox(t)) for t in kept_y)):
                kept_x.append(text)
        return VGroup(*parts, *kept_y, *kept_x), plot, to_point, ink

    def _markers(self, plan: _Plan, theme: Theme, plot: Rect, to_point, ink: np.ndarray,
                 obstacles: list[Rect]):
        """Dots, labels and trace paths. Labels go on the side of the dot the
        curve is not on (below a minimum, above a maximum, off the outside of a
        slope), and must stay inside the plot clear of everything placed."""
        p: GraphPlotParams = self.params
        placed: list[tuple[Dot, object, VMobject | None]] = []
        # Every dot is reserved up front, so an early label cannot cover a
        # later marker.
        points = [to_point(m.x, my)[0] for m, my in zip(p.markers, plan.marker_ys)]
        taken = list(obstacles) + [Rect(a[0], a[1], 2 * DOT_RADIUS, 2 * DOT_RADIUS) for a in points]
        last_x: dict[tuple[int, int], float] = {}
        h = (plan.x1 - plan.x0) / (N_SAMPLES - 1)

        for j, (m, my, at) in enumerate(zip(p.markers, plan.marker_ys, points)):
            dot = label(Dot(at, radius=DOT_RADIUS, color=theme.palette.fg), f"marker[{j}]")

            # Trace along the run the marker sits on, from the previous marker
            # on that run if there was one, else from where the run starts.
            path = None
            runs = plan.runs[m.function]
            ri = next((i for i, r in enumerate(runs) if r[0, 0] <= m.x <= r[-1, 0]), None)
            if ri is not None:
                run = runs[ri]
                start = last_x.get((m.function, ri), run[0, 0])
                start = start if start < m.x else run[0, 0]
                pts = run[(run[:, 0] >= start) & (run[:, 0] < m.x)]
                pts = np.vstack([pts, [[m.x, my]]])
                if len(pts) >= 2:
                    path = VMobject().set_points_as_corners(to_point(pts[:, 0], pts[:, 1]))
                    dot.move_to(path.get_start())
                last_x[(m.function, ri)] = m.x

            text = None
            if m.label:
                text = label(caption_text(wrap(m.label, 22), theme, theme.palette.fg),
                             f"marker[{j}].label")
                text = self._place_label(text, at, m, to_point, plot, ink, taken, h)
                taken.append(bbox(text))
            placed.append((dot, text, path))
        return placed

    def _place_label(self, text, at, m: GraphMarker, to_point, plot: Rect,
                     ink: np.ndarray, taken: list[Rect], h: float):
        expr = self.params.functions[m.function].expr
        y = evaluate(expr, np.array([m.x - h, m.x, m.x + h]))
        p0, p1, p2 = to_point(np.array([m.x - h, m.x, m.x + h]), np.nan_to_num(y))
        slope = (p2[1] - p0[1]) / max(p2[0] - p0[0], 1e-9)
        if abs(slope) < 0.35:
            convex = (p0[1] + p2[1] - 2 * p1[1]) > 0
            first = [DOWN, DR, DL] if convex else [UP, UR, UL]
        else:
            first = [UL, DR] if slope > 0 else [UR, DL]
        order = first + [d for d in (UR, UL, DR, DL, UP, DOWN, RIGHT, LEFT)
                         if not any(d is f for f in first)]
        # Hard constraints: inside the plot, clear of every label and dot.
        # Soft: clear of curve and axis ink -- preferred, but a label touching
        # a line is still readable, so the first hard-valid side is the fallback.
        fallback = None
        for d in order:
            text.next_to(at, d, buff=MARKER_BUFF)
            box = bbox(text)
            if not plot.contains(box) or any(box.intersects(t) for t in taken):
                continue
            if not _inked(ink, box):
                return text
            if fallback is None:
                fallback = d
        if fallback is not None:
            return text.next_to(at, fallback, buff=MARKER_BUFF)
        raise LayoutError(
            f"GraphPlot: no room to label the marker at x={m.x} without covering "
            "another label or leaving the plot. Fewer or shorter marker labels, or "
            "split the beat.", kind="overflow")

    # --- fixtures --------------------------------------------------------------

    @classmethod
    def examples(cls):
        return [
            {"functions": [{"expr": "x^2 - 2*x - 3", "name": "f"}],
             "x_range": [-2, 4],
             "markers": [{"x": 1, "label": "vertex"}, {"x": 3, "label": "root"}]},
            {"functions": [{"expr": "sin(x)"}, {"expr": "cos(x)"}],
             "x_range": [-6.5, 6.5]},
            # Asymptotes, with the author choosing the window (y_range).
            {"functions": [{"expr": "tan(x)"}], "x_range": [-4.5, 4.5],
             "y_range": [-5, 5], "markers": [{"x": 0.785398, "label": "tan = 1"}]},
        ]

    @classmethod
    def stress(cls):
        return [
            # (a) ~3x a realistic beat: every function slot, long expressions,
            # every marker slot with a sentence for a label.
            {"functions": [
                {"expr": "x^5 - 4*x^3 + 2*x^2 - 3*x + 1 + sin(3*x)*exp(-x^2/4)", "name": "f"},
                {"expr": "(x^4 - 3*x^2 + 1)/(x^2 + 1) + sqrt(abs(x))*cos(2*x)", "name": "g"},
                {"expr": "log(abs(x) + 1)*tanh(x) - atan(x^3)/pi + 2^(-x^2)", "name": "h"}],
             "x_range": [-3, 3],
             "markers": [{"x": -2, "label": "the leftmost point we care about here"},
                         {"x": -1, "function": 1, "label": "where the second curve dips"},
                         {"x": 1, "function": 2, "label": "a third label crowding in"},
                         {"x": 2, "label": "and one more for good measure"}]},
            # (b) an unwrappable 60-character token as a marker label.
            {"functions": [{"expr": "x^2"}], "x_range": [-3, 3],
             "markers": [{"x": 1, "label": "a" * 60}]},
            # (c) minimal and degenerate inputs.
            {"functions": [{"expr": "x"}], "x_range": [0, 1]},
            {"functions": [{"expr": "0"}], "x_range": [-1, 1]},
            # Asymptotes and holes without a y_range: the window must trim the
            # spikes and the curve must break at every pole, never bridge it.
            {"functions": [{"expr": "1/x"}], "x_range": [-5, 5]},
            {"functions": [{"expr": "tan(x)"}], "x_range": [-20, 20]},
            {"functions": [{"expr": "sqrt(x)"}, {"expr": "log(x)"}], "x_range": [-4, 4]},
            {"functions": [{"expr": "x*sin(1/x)"}], "x_range": [-1, 1]},
            # Huge and tiny ranges: tick labels must stay compact.
            {"functions": [{"expr": "x^3"}], "x_range": [-1e6, 1e6]},
            {"functions": [{"expr": "exp(x)"}], "x_range": [0, 700]},
            {"functions": [{"expr": "x^2"}], "x_range": [0, 1e-6]},
        ]
