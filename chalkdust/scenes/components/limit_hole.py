"""LimitAtHole: a limit at a point where the function is undefined.

The picture: a curve f with an open circle at x = a, because f has no value
there ("f(a) is undefined, so what happens near a?"). Two points, one on each
side of a, slide along the curve towards the hole; the gap to a shrinks
geometrically, and live readouts of x and f(x) for both (computed from f at
the points' actual x) settle on one number. Only then the name: lim f(x) = L,
with L computed by code, and L marked on the y-axis at the hole's height. The
limit is about the approach, never about the value at a.

Phases (each one beat; animation.py):
  pose      curve -> hole -> question
  approach  points -> slide -> agree
  name      limit -> value

Setup: `function` (an expression in x), `x_range`, `a`. Everything on screen
is computed from these (_analyse): the one-sided limits, from f at a +- h
along five geometric sequences of h offset by the golden ratio (so none lines
up with a period of f), with Richardson extrapolation and only the samples
whose rounding error is small (_eval_err); the points' start gap; where the
slide stops (HOLD_DEPTH times closer in than the gap from which both
readouts show L at every gap); the readouts' decimals, from the y-scale.
Refused as param errors, before anything is drawn: a jump (the one-sided
limits differ), a side that diverges, drifts or does not settle, f that
cancels too finely at a to compute, f defined at a (no hole: another
component; "defined" means every part of f has a value at a), f undefined on
more than the one point a near a, a too close to the edge to approach from
both sides, and f too small to read at the readouts' most decimals.
"""

from __future__ import annotations

import ast
import math as _math
from dataclasses import dataclass
from functools import lru_cache
from typing import Annotated, Any, Literal

import numpy as np
from manim import (
    LEFT,
    RIGHT,
    UP,
    Create,
    DashedLine,
    FadeIn,
    FadeOut,
    GrowFromCenter,
    Indicate,
    Line,
    ReplacementTransform,
    ValueTracker,
    VGroup,
    VMobject,
    Write,
    always_redraw,
    linear,
    there_and_back,
)
from pydantic import Field, model_validator

from chalkdust.scenes.base import ChalkdustScene, Moment, Timeline
from chalkdust.scenes.components.animation import Animation, AnimationParams
from chalkdust.scenes.components.base import label, register
from chalkdust.scenes.components.graph_plot import (
    _OFF_SCALE,
    _check_range,
    _eval,
    _parse,
    _runs,
    axis_ticks,
)
from chalkdust.scenes.plotting import (
    POINT_Z,
    X_TICKS,
    Expr,
    Fn,
    Plot,
    column_lines,
    fmt,
    live,
    readout,
    role,
    y_window,
)
from chalkdust.scenes.regions import LayoutError, Rect, scale_with_tags
from chalkdust.scenes.theme import Theme, math

FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]

MAX_SPAN = 1e4
# The points start this share of the room on a's nearer side away from a, at
# most MAX_START of the x-range, rounded down to a nice number ...
START_SHARE_OF_ROOM = 0.75
MAX_START = 0.4
# ... and never closer than this share of the x-range, or the approach cannot
# be seen. A hole within about 11% of the edge is refused for this reason.
MIN_START = 0.08
START_MANTISSAS = (1, 1.5, 2, 2.5, 3, 4, 5, 6, 8)
# The slide shrinks the gap at least this many times before the readouts must
# hold L, so the points visibly close in even where the readouts show L from
# the start (a flat window) ...
MIN_SHRINK = 50
# ... and then goes this many times closer still, with both readouts showing L
# all the way (checked at N_HOLD gaps): the last stretch of the slide is the
# readouts visibly holding L, not landing on it at one gap.
HOLD_DEPTH = 10
N_HOLD = 600
# One-sided limits: f at a +- HS * width * c * 2^-k for k = 0..K, so h ends
# near 1e-7 of HS * width (_one_sided says why no deeper), along one sequence
# per offset c. The offsets are spread over an octave by the golden ratio, so
# no sample lines up with a period of f: sin(pi/x) is exactly 0 at every
# x = 2^-k / 10, and a single sequence of those would read it as a limit 0.
GOLDEN = (5 ** 0.5 - 1) / 2
OFFSETS = tuple(2.0 ** -((j * GOLDEN) % 1.0) for j in range(1, 6))
HS = 0.05
K = 23
# A side's values running away from its plateau, or a tail with no plateau,
# that grows at least GROWTH times per halving of the gap heads to infinity
# (1/x: 2 times). A slower steady drift cannot be told from a limit approached
# very slowly (|x|^0.01 ~ 1 + 0.01 log|x|), so it is refused as either.
GROWTH = 1.3
DRIFT = 1.03
# Plateau windows of Richardson estimates start this deep (h <= HS*width/64),
# so a plateau at large h cannot hide a divergence closer in -- unless the
# samples that deep are already lost to rounding (a narrow x_range around a
# cancellation), then as deep as they allow, never above PLATEAU_MIN; the
# deepest computable value must still agree with the plateau (_one_sequence).
PLATEAU_FROM = 6
PLATEAU_MIN = 2
PLATEAU_LEN = 4
# A sample is used only if its rounding-error bound (_eval_err) is under this
# share of the convergence tolerance.
RELIABLE = 0.1
# Unit roundoff of float64.
UNIT = np.finfo(float).eps / 2
# Convergence tolerance, relative to the window's height: 1e-4 of the height
# is far below a pixel at 1080p.
TOL = 1e-4
# A limit is printed as an exact decimal ("= 0.25") when it is this close to
# one with no more digits than the readouts, relative to the larger of |L| and
# the picture's scale (never to 1: a tiny f would round to "= 0"); else
# "\approx". Not tighter: a limit computed through a cancellation is good to
# about 2e-8 ((1 - cos x)/x^2 gives 0.50000000924), and far below what the
# readouts resolve (a few significant digits of the window).
EXACT_REL = 1e-7
# The two one-sided limits agree when this close, relative to the same.
AGREE_REL = 1e-9
# Readout decimals for f: READOUT_DIGITS minus the window height's magnitude.
READOUT_DIGITS = 2
MAX_DECIMALS = 8
DOT_RADIUS = 0.065
HOLE_RADIUS = 0.09
ROW_GAP = 0.45
TICK_HALF = 0.2
MIN_GUIDE = 1e-3
# The emphasis pulse: grow to at most PULSE and back, keeping PULSE_GAP clear
# of the neighbours (_pulse).
PULSE = 1.2
PULSE_GAP = 0.06
# Clear space between neighbouring x tick labels (scene units; 0.1 is about
# 13 px at 1080p). Low enough that only a real collision trips it: the
# examples' tightest gap is 0.19, and a threshold near it would make the tick
# count, and so the snapshot structure, depend on the fonts installed.
MIN_TICK_GAP = 0.1


class LimitHoleParams(AnimationParams):
    phase: Literal["pose", "approach", "name"]
    function: Expr = Field(description="f as an expression in x, undefined at a, e.g. 'sin(x)/x'")
    x_range: tuple[FiniteFloat, FiniteFloat] = Field(description="[min, max] of x shown")
    a: FiniteFloat = Field(description="x of the hole: where f is undefined and the limit is taken")

    @model_validator(mode="after")
    def _true_picture(self) -> LimitHoleParams:
        _check_range(self.x_range, "x_range")
        x0, x1 = self.x_range
        if x1 - x0 > MAX_SPAN:
            raise ValueError(f"x_range is {x1 - x0:g} wide; keep it under {MAX_SPAN:g}")
        if not x0 < self.a < x1:
            raise ValueError(f"a = {self.a:g} must lie strictly inside x_range {list(self.x_range)}")
        _analyse(self.function, float(x0), float(x1), float(self.a))
        return self


# --- the mathematics, from the setup alone ------------------------------------


@dataclass(frozen=True)
class _Limit:
    L: float
    exact_d: int | None  # decimals of L as an exact decimal ("="); None: rounded ("\approx")
    h0: float         # gap the points start at
    h_end: float      # gap the slide stops at: both readouts show L
    y0: float
    y1: float
    dx: int           # decimals of the x readouts
    dy: int           # decimals of the f readouts and of L


def _side_name(s: int) -> str:
    return "left" if s < 0 else "right"


# |g'(u)| for each function the expression grammar admits (graph_plot._FUNCS),
# given u and g(u): how an error in u carries through g.
_SLOPES = {
    "sin": lambda u, g: np.abs(np.cos(u)),
    "cos": lambda u, g: np.abs(np.sin(u)),
    "tan": lambda u, g: 1 + g * g,
    "asin": lambda u, g: 1 / np.sqrt(np.abs(1 - u * u)),
    "acos": lambda u, g: 1 / np.sqrt(np.abs(1 - u * u)),
    "atan": lambda u, g: 1 / (1 + u * u),
    "sinh": lambda u, g: np.cosh(u),
    "cosh": lambda u, g: np.abs(np.sinh(u)),
    "tanh": lambda u, g: 1 - g * g,
    "exp": lambda u, g: np.abs(g),
    "log": lambda u, g: 1 / np.abs(u),
    "ln": lambda u, g: 1 / np.abs(u),
    "log10": lambda u, g: 1 / (np.abs(u) * np.log(10)),
    "log2": lambda u, g: 1 / (np.abs(u) * np.log(2)),
    "sqrt": lambda u, g: 0.5 / np.abs(g),
    "abs": lambda u, g: np.ones_like(u),
}


def _carry(slope: np.ndarray, e: np.ndarray) -> np.ndarray:
    """slope * e, with an exact input (e = 0) staying exact even where the
    slope is infinite. A slope that is nan is 0 * inf -- 0^r's slope in r,
    0 * log 0 -- and the result does not move: 0."""
    slope = np.where(np.isnan(slope), 0.0, np.abs(slope))
    return np.where(e == 0, 0.0, slope * e)


def _eval_err(node: ast.expr, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(value, bound on its rounding error) of a parsed expression at the
    exact points x, to first order: each operation rounds its result (UNIT
    relative) and carries its inputs' errors by its derivative. A difference
    of nearly equal numbers keeps their absolute error, so 1 - cos h, which
    is h^2/2, is good to about 1e-16 / (h^2/2) relative, and exactly 0, all
    error, below h ~ 1e-8. Says which samples f can be computed at."""
    if isinstance(node, ast.Constant):
        # An integer literal (1e8, 1000) is exactly its float; another
        # decimal (0.1) is rounded to one.
        v = np.float64(node.value)
        exact = float(v).is_integer() and abs(v) < 2.0 ** 53
        return v + 0 * x, (0.0 if exact else UNIT * abs(v)) + 0 * x
    if isinstance(node, ast.Name):
        if node.id == "x":
            return x, np.zeros_like(x)
        v = np.float64(_eval(node, x))
        return v + 0 * x, UNIT * abs(v) + 0 * x
    if isinstance(node, ast.UnaryOp):
        v, e = _eval_err(node.operand, x)
        return (-v if isinstance(node.op, ast.USub) else v), e
    if isinstance(node, ast.BinOp):
        lv, le = _eval_err(node.left, x)
        rv, re_ = _eval_err(node.right, x)
        res = np.asarray(_eval(node, x), dtype=float)
        if isinstance(node.op, (ast.Add, ast.Sub)):
            err = le + re_
        elif isinstance(node.op, ast.Mult):
            err = _carry(rv, le) + _carry(lv, re_)
        elif isinstance(node.op, ast.Div):
            err = _carry(1 / rv, le) + _carry(res / rv, re_)
        else:  # Pow: d/dl = r l^(r-1) = r res / l, d/dr = res log|l|
            err = _carry(rv * res / lv, le) + _carry(res * np.log(np.abs(lv)), re_)
        return res, err + UNIT * np.abs(res)
    assert isinstance(node, ast.Call)
    uv, ue = _eval_err(node.args[0], x)
    g = np.asarray(_eval(node, x), dtype=float)
    slope = _SLOPES[node.func.id](uv, g)
    return g, _carry(slope, ue) + UNIT * np.abs(g)


def _rounding(expr: str, xs: np.ndarray) -> np.ndarray:
    """Bound on the rounding error of f at each of xs (inf where unknown)."""
    with np.errstate(all="ignore"):
        _, err = _eval_err(_parse(expr), np.asarray(xs, dtype=float))
    err = np.broadcast_to(err, np.shape(xs)).astype(float)
    return np.where(np.isfinite(err), err, np.inf)


def _one_sided(f: Fn, a: float, s: int, hs: float, tol: float) -> tuple[float, float]:
    """(limit, accuracy) of f(x) as x -> a from side s (-1 left, +1 right):
    one estimate per sequence of gaps (OFFSETS), and they must all agree. An
    oscillation that one sequence happens to sample at a single phase (its
    zeros) shows a different value, or none, along the others."""
    ests = [_one_sequence(f, a, s, hs * c, tol) for c in OFFSETS]
    vals = np.array([e for e, _ in ests])
    accuracy = max(max(sp for _, sp in ests), float(np.ptp(vals)))
    if np.ptp(vals) > max(tol, 10 * max(sp for _, sp in ests)):
        raise ValueError(
            f"f(x) does not settle on one value as x -> {a:g} from the {_side_name(s)} (it "
            f"reads {vals.min():.4g} to {vals.max():.4g} at gaps a few % apart): there is no "
            "limit to show; pick a point where f approaches a number from both sides")
    return float(np.mean(vals)), accuracy


def _runaway(a: float, side: str, xs: np.ndarray, v: np.ndarray, steps: np.ndarray) -> None:
    """Refuse a side whose values keep moving one way, gap after halved gap
    (`steps` are the last differences). Growing GROWTH times per halving is
    a pole; any slower steady drift is told the way it is, since no finite
    sample separates log|x| (no limit) from |x|^0.01 (a limit, approached
    too slowly to show)."""
    mag = np.abs(steps)
    if np.all(mag[1:] >= GROWTH * mag[:-1]):
        way = "+infinity" if steps[-1] > 0 else "-infinity"
        raise ValueError(
            f"f(x) heads to {way} as x -> {a:g} from the {side} "
            f"(f = {v[-1]:.3g} at x = {xs[-1]:.3g}): there is no limit to show; "
            "pick a point where f approaches a number from both sides")
    raise ValueError(
        f"f(x) is still drifting as x -> {a:g} from the {side} (by {abs(steps[-1]):.2g} per "
        f"halving of the gap, at x = {xs[-1]:.3g}): it has no limit there, or approaches one "
        "too slowly to show; pick a point where f settles quickly from both sides")


def _one_sequence(f: Fn, a: float, s: int, hs: float, tol: float) -> tuple[float, float]:
    """(limit, accuracy) of f(x) as x -> a from side s along one sequence.

    f at a + s*hs*2^-k for k = 0..K. Richardson's R_k = 2 v_{k+1} - v_k
    removes the term linear in h; the limit is the first (shallowest) run of
    PLATEAU_LEN consecutive R's within `tol` of each other, at depth
    PLATEAU_FROM or more -- deep enough that a plateau at large h cannot hide
    a divergence closer in -- that the sequence was already approaching.

    Only samples computed to within RELIABLE * tol are used (_rounding):
    closer in, the cancellation f makes at a has eaten the digits -- 1 - cos h
    is exactly 0 below h ~ 1e-8, and (e^h - 1 - h)/h^2 is rounding error
    over h^2, which grows like a pole. Shallowest and approached, as a
    second guard: a run of rounding zeros is never headed for. Raises
    ValueError for a side that is undefined, cannot be computed, diverges,
    drifts or never settles."""
    side = _side_name(s)
    hk = hs * 2.0 ** -np.arange(K + 1)
    xs = a + s * hk
    v = f.np(xs)
    bad = ~np.isfinite(v)
    if bad.any():
        x = float(xs[np.argmax(bad)])
        raise ValueError(
            f"f is undefined at x = {x:.6g}, just {side} of a = {a:g}: the hole must be the "
            "only point near a where f is undefined; pick a function defined on both sides of a")
    err = _rounding(f.expr, xs)
    ok = err <= RELIABLE * tol
    n = len(ok) if ok.all() else int(np.argmin(ok))

    def cancels() -> ValueError:
        return ValueError(
            f"f cancels too finely near a = {a:g} to compute in floating point: its values "
            f"within {float(hk[n]):.2g} {side} of a have lost their digits to rounding; write f "
            "so it does not subtract nearly equal numbers, or pick another function")

    if n < PLATEAU_MIN + PLATEAU_LEN + 1:
        raise cancels()

    def drifting(steps: np.ndarray) -> None:
        """Refuse a side that keeps moving: a pole as a pole; a slower drift
        in samples cut short by rounding (n < K + 1) as the cancellation it
        may be -- closer in, where the digits are gone, it may yet settle."""
        mag = np.abs(steps)
        if n < K + 1 and not np.all(mag[1:] >= GROWTH * mag[:-1]):
            raise cancels()
        _runaway(a, side, xs, v, steps)

    xs, v, err = xs[:n], v[:n], err[:n]
    r = 2 * v[1:] - v[:-1]
    spreads = [float(np.ptp(r[j:j + PLATEAU_LEN])) for j in range(len(r) - PLATEAU_LEN + 1)]

    def approached(j: int) -> bool:
        """The value just before the run is already near it, as a converging
        sequence's is; a run of rounding zeros is entered from far away."""
        lim = float(np.mean(r[j:j + PLATEAU_LEN]))
        start = float(np.max(np.abs(v[:3] - lim)))
        return abs(v[j - 1] - lim) <= max(0.5 * start, tol)

    j = max(PLATEAU_MIN, min(PLATEAU_FROM, len(spreads) - 1))
    while j < len(spreads):
        if spreads[j] > tol:
            j += 1
            continue
        if not approached(j):
            # Skip the whole run, or its later windows would be compared
            # with values inside it.
            while j < len(spreads) and spreads[j] <= tol:
                j += 1
            continue
        # Refine: go deeper while the runs keep tightening; the first
        # loosening is rounding error taking over, so stop there.
        while j + 1 < len(spreads) and spreads[j + 1] < spreads[j]:
            j += 1
        limit = float(np.mean(r[j:j + PLATEAU_LEN]))
        # A divergence too small to break the plateau (x + 1e-9/x, or the
        # slower x + 1e-4 log|x|) still pulls the deepest values away from
        # it: all one way, further every step, and by more than tol. Rounding
        # noise near a cancellation changes sign.
        dev = v[-8:] - limit
        gap = np.abs(dev)
        if ((np.all(dev > 0) or np.all(dev < 0)) and gap[-1] > tol
                and np.all(gap[1:] >= DRIFT * gap[:-1]) and gap[-1] >= 1.5 * gap[0]):
            drifting(np.diff(v[-8:]))
        # Closer in than the plateau, f must stay with it: a value further
        # from it than where the plateau began is f going somewhere else
        # (1 + exp(-|x|/1e-4) is flat at 1, then rises to 2).
        if abs(v[-1] - limit) > max(2 * tol, 2 * abs(v[j] - limit)):
            raise ValueError(
                f"f(x) moves away from {limit:.4g} again closer to a = {a:g} from the {side} "
                f"(f = {v[-1]:.4g} at x = {xs[-1]:.3g}): no limit can be shown; pick a point "
                "where f settles from both sides")
        # A limit approached too slowly to show: the last Richardson steps
        # (the term linear in h is already gone from them), all one way and
        # well clear of rounding, shrink by rho per halving, so what they
        # still add up to past the last one is about step * rho/(1 - rho).
        # If that would show, the plateau is not the limit: sin(x)/x +
        # 0.007|x|^0.001 creeps to 1 through 1.0069 at every computable gap.
        steps = np.diff(r[-8:])
        noise = (2 * err[1:] + err[:-1])[-8:]
        if (len(steps) >= 4 and (np.all(steps > 0) or np.all(steps < 0))
                and np.all(np.abs(steps) > 10 * (noise[1:] + noise[:-1]))):
            rho = float(np.mean(np.abs(steps[1:]) / np.abs(steps[:-1])))
            if rho >= 1 or abs(steps[-1]) * rho / (1 - rho) > tol:
                drifting(steps)
        return limit, spreads[j]
    tail = np.diff(v[-12:])
    if len(tail) >= 4 and (np.all(tail > 0) or np.all(tail < 0)):
        drifting(tail)
    raise ValueError(
        f"f(x) does not settle on one value as x -> {a:g} from the {side} (or cancels too "
        "finely at a to compute in floating point): there is no limit to show; pick a point "
        "where f approaches a number from both sides")


def _nice_floor(v: float) -> float:
    mag = 10.0 ** _math.floor(_math.log10(v))
    for m in reversed(START_MANTISSAS):
        if m * mag <= v * (1 + 1e-12):
            return m * mag
    return START_MANTISSAS[-1] * mag / 10


def _starts_below(h: float):
    """Nice gaps (START_MANTISSAS x 10^n), descending from h."""
    v = _nice_floor(h)
    while True:
        yield v
        v = _nice_floor(v * (1 - 1e-9))


def _grid_below(h: float):
    """1-2-5 gaps below h, descending: the slide stops on one, so the x
    readouts end on exact decimals."""
    mag = 10.0 ** _math.floor(_math.log10(h))
    while True:
        for m in (5, 2, 1):
            if m * mag < h * (1 - 1e-12):
                yield m * mag
        mag /= 10


def _decimals_for(h: float) -> int:
    """Decimals that show a +- h as different from a: a step of at most h/2."""
    return max(1, _math.ceil(-_math.log10(h / 2) - 1e-9))


# Where a start gap's path is sampled, as shares of the gap. Not deeper: the
# one-sided limits already found f defined closer in, and the slide's own
# path down to where it stops is checked once that is known (_analyse);
# deeper samples would read the rounding error of f's cancellation at a
# ((cos x - 1 + x^2/2)/x^4) as the curve leaving the picture.
PATH_TS = np.linspace(1.0, 1e-3, 801)


def _path_problem(f: Fn, a: float, h: float, y0: float, y1: float,
                  ts: np.ndarray = PATH_TS) -> str | None:
    """Why a point cannot slide from a +- h towards the hole, or None."""
    margin = 0.02 * (y1 - y0)
    for s in (-1, 1):
        xs = a + s * h * ts
        ys = f.np(xs)
        bad = ~np.isfinite(ys)
        if bad.any():
            return (f"f is undefined at x = {float(xs[np.argmax(bad)]):.6g}, close {_side_name(s)} "
                    "of a: the hole must be the only point near a where f is undefined")
        if ys.min() < y0 + margin or ys.max() > y1 - margin:
            return (f"f leaves the picture within {h:.3g} {_side_name(s)} of a: it is too steep "
                    "near a to follow; narrow x_range around a")
        # Against the picture's own window (the path is inside it), so a jump
        # counts exactly when the drawn curve would break there.
        order = np.argsort(xs)
        if len(_runs(f.expr, xs[order], ys[order], y0, y1)) != 1:
            return (f"f breaks within {h:.3g} {_side_name(s)} of a: the hole must be the only "
                    "break near a")
    return None


def _undefined_at(expr: str, a: float) -> bool:
    """f has no value at a as written: some part of it is undefined there.
    numpy can still return a number for the whole -- (1 + x)^(1/x) at 0 is
    1^inf = 1, x*atan(1/x) is 0 * pi/2 -- so every subexpression is
    evaluated at a, and one that is not a finite number makes a a hole."""
    xa = np.array([a], dtype=float)
    with np.errstate(all="ignore"):
        for node in ast.walk(_parse(expr)):
            if isinstance(node, (ast.BinOp, ast.UnaryOp, ast.Call)):
                v = np.asarray(_eval(node, xa), dtype=float)
                if not np.all(np.isfinite(v)) or np.any(np.abs(v) > _OFF_SCALE):
                    return True
    return False


@lru_cache(maxsize=64)
def _analyse(expr: str, x0: float, x1: float, a: float) -> _Limit:
    """Everything the picture shows, computed from the setup; raises
    ValueError (a param error, rung 1) for a setup that would draw a false
    picture, with the fix in the message."""
    f = Fn(expr)
    width = x1 - x0
    room = min(a - x0, x1 - a)
    edge = (f"a = {a:g} is too close to the edge of x_range [{x0:g}, {x1:g}] to approach it "
            "from both sides; widen x_range on that side")
    if START_SHARE_OF_ROOM * room < MIN_START * width:
        raise ValueError(edge)
    ys0, ys1 = y_window([f], x0, x1, include_zero=False)
    samples = f.np(np.linspace(x0, x1, 600))
    samples = samples[np.isfinite(samples)]
    own = max(float(np.ptp(samples)), float(np.max(np.abs(samples))))
    yscale = _picture_scale(ys1 - ys0, own)
    tol = TOL * yscale
    hs = HS * width
    left, s_left = _one_sided(f, a, -1, hs, tol)
    right, s_right = _one_sided(f, a, 1, hs, tol)
    agree = max(10 * (s_left + s_right), AGREE_REL * max(abs(left), abs(right), yscale))
    if abs(left - right) > agree:
        raise ValueError(
            f"the one-sided limits differ ({left:.4g} from the left, {right:.4g} from the right): "
            f"f jumps at a = {a:g}, so the limit does not exist; pick a point where both sides "
            "approach the same value")
    L = 0.5 * (left + right)
    fa = f.at(a)
    if _math.isfinite(fa) and not _undefined_at(expr, a):
        if abs(fa - L) <= max(tol, agree):
            raise ValueError(
                f"f is defined at a = {a:g} (f(a) = {fa:.4g}) and continuous there: there is no "
                "hole to show. Use GraphPlot with a marker for a point on a curve, or pick an f "
                "that is undefined at a, such as sin(x)/x at 0")
        raise ValueError(
            f"f is defined at a = {a:g} (f(a) = {fa:.4g}) but approaches {L:.4g}: this animation "
            "shows a hole where f has no value; pick an f that is undefined at a")
    y0, y1 = y_window([f], x0, x1, extra=[L], include_zero=False)
    scale = _picture_scale(y1 - y0, max(own, abs(L)))
    dy = int(max(0, READOUT_DIGITS - _math.floor(_math.log10(scale))))
    if dy > MAX_DECIMALS:
        raise ValueError(
            f"f's values on x_range are only about {scale:.2g} in size: too small to read at "
            f"{MAX_DECIMALS} decimals; multiply f by a power of ten")
    # L on a rounding boundary (0.125 at two decimals): values from the two
    # sides round apart for ever, so show one more decimal.
    while dy < MAX_DECIMALS and abs((abs(L) * 10.0 ** dy) % 1 - 0.5) < 0.01:
        dy += 1

    # Where the points start: the largest nice gap whose paths to the hole
    # are defined, unbroken and inside the window.
    h0, why = None, None
    for h in _starts_below(min(START_SHARE_OF_ROOM * room, MAX_START * width)):
        if h < MIN_START * width * (1 - 1e-9):
            break
        why = _path_problem(f, a, h, y0, y1)
        if why is None:
            h0 = h
            break
    if h0 is None:
        raise ValueError(why or edge)

    # Where the slide stops: HOLD_DEPTH times closer in than the first 1-2-5
    # gap (at least MIN_SHRINK times under h0) from which both readouts show
    # L at every gap down to the stop -- checked at N_HOLD gaps spread by the
    # golden ratio, so they line up neither with decimals nor with a period
    # of f. The first such gap, not a deeper one: closer in, rounding in f's
    # cancellation at a can make a readout lie.
    target = fmt(L, dy)
    half = 0.5 * 10.0 ** -dy
    spread = np.concatenate([[0.0, 1.0], (np.arange(1, N_HOLD) * GOLDEN) % 1.0])

    def holds_l(hi: float, lo: float) -> bool:
        """Both readouts show L at every gap in [lo, hi], and so does f's
        true value: the reading stays L across its rounding bound."""
        hs_ = hi * (lo / hi) ** spread
        for s in (-1, 1):
            xs_ = a + s * hs_
            for v, e in zip(f.np(xs_), _rounding(f.expr, xs_)):
                if not (abs(v - L) + e < half and fmt(v, dy) == target
                        and fmt(v - e, dy) == target and fmt(v + e, dy) == target):
                    return False
        return True

    floor_h = max(1e-12 * abs(a), 2 * 10.0 ** -MAX_DECIMALS)
    h_end = None
    for h in _grid_below(h0 / MIN_SHRINK * (1 + 1e-9)):
        if h / HOLD_DEPTH < floor_h * (1 - 1e-9):
            break
        if holds_l(h, h / HOLD_DEPTH):
            h_end = h / HOLD_DEPTH
            break
    if h_end is None:
        raise ValueError(
            f"f changes too fast near a = {a:g} for its readouts to settle on the limit at "
            f"{dy} decimals; narrow x_range around a, or rescale f")
    # The whole slide, h0 down to h_end, stays defined and in the window.
    why = _path_problem(f, a, h0, y0, y1, ts=(h_end / h0) ** np.linspace(0, 1, 241))
    if why is not None:
        raise ValueError(why)
    dx = _decimals_for(h_end)
    exact_d = next((d for d in range(dy + 1)
                    if abs(L - round(L, d)) <= EXACT_REL * max(abs(L), scale)), None)
    return _Limit(L=L, exact_d=exact_d, h0=h0, h_end=h_end, y0=y0, y1=y1, dx=dx, dy=dy)


def _picture_scale(height: float, own: float) -> float:
    """The size the picture's numbers are judged against: the window's
    height, or f's own size when that is smaller. y_window pads a flat f by
    +-1, and a tolerance or a decimal count taken from that padding would
    call 1e-12 * abs(x)/x a limit, printed "= 0". An f that is 0 everywhere
    has no size of its own, so the window's height is used then."""
    return min(height, own) if own > 0 else height


def _drawn(m) -> bool:
    """True if m draws anything (a DashedLine keeps its points in its dashes)."""
    return any(len(x.points) for x in m.get_family())


def _pulse(m, lo: float, hi: float):
    """Emphasis: m grows to at most PULSE and back, as Indicate does, but
    about a point chosen so that at its peak it stays between x = lo and
    x = hi. Indicate grows about the centre, which pushed a wide readout
    (4,000,000) across the column's edge."""
    left, right = m.get_left()[0], m.get_right()[0]
    room_l, room_r = max(0.0, left - lo), max(0.0, hi - right)
    room = room_l + room_r
    s = min(PULSE, 1 + room / m.width)
    px = left + m.width * room_l / room if room > 0 else m.get_center()[0]
    about = np.array([px, m.get_center()[1], 0.0])
    return m.animate(rate_func=there_and_back).scale(s, about_point=about)


def _num(v: float) -> str:
    """A setup value as written: integers without decimals."""
    return str(int(v)) if float(v).is_integer() else f"{v:g}"


def _grouped(s: str) -> str:
    """Thousands commas as the readouts' DecimalNumber prints them, in LaTeX."""
    sign = "-" if s.startswith("-") else ""
    body = s.lstrip("-")
    whole, _, frac = body.partition(".")
    whole = f"{int(whole):,}".replace(",", "{,}")
    return sign + whole + ("." + frac if frac else "")


def _value_tex(lim: _Limit) -> str:
    """L as printed in the limit line: an exact decimal when it is one (to
    EXACT_REL), else rounded to the readouts' decimals with \\approx."""
    if lim.exact_d is not None:
        return "= " + _grouped(fmt(round(lim.L, lim.exact_d), lim.exact_d))
    return r"\approx " + _grouped(fmt(lim.L, lim.dy))


def _undefined_tex(a: float) -> str:
    return rf"f({_num(a)})\ \text{{is undefined}}"


def _question_tex(a: float) -> str:
    return rf"f(x)\ \text{{for }} x \text{{ near }} {_num(a)}\,?"


def _limit_tex(a: float, value: str) -> str:
    return rf"\lim_{{x \to {_num(a)}}} f(x) {value}"


# --- the picture ---------------------------------------------------------------


def _fitted_plot(theme: Theme, box: Rect, x0: float, x1: float, y0: float, y1: float) -> Plot:
    """A Plot with as many x ticks as have room. Plot labels every tick it
    is asked for, and on a narrow range the labels are long (-0.0015) and
    collide; so ask for fewer until neighbours stand MIN_TICK_GAP apart.
    Plot.axes holds the two axis lines, then a (tick, label) pair per x tick."""
    for n in range(X_TICKS, 1, -1):
        plot = Plot(theme, box, x0, x1, y0, y1, x_ticks=n)
        count = len(axis_ticks(plot.x0, plot.x1, n)[1])
        labels = [plot.axes[3 + 2 * i] for i in range(count)]
        if all(b.get_left()[0] - a.get_right()[0] >= MIN_TICK_GAP
               for a, b in zip(labels, labels[1:])):
            return plot
    raise LayoutError(f"x tick labels on [{x0:g}, {x1:g}] collide however few; widen x_range",
                      kind="overflow")


class _Picture:
    """Every mobject of the animation, built from the setup alone, so each
    phase's start is exactly the previous phase's end (animation.py)."""

    def __init__(self, p: LimitHoleParams, theme: Theme) -> None:
        self.p, self.theme = p, theme
        self.f = Fn(p.function)
        self.a = float(p.a)
        x0, x1 = map(float, p.x_range)
        self.lim = lim = _analyse(p.function, x0, x1, self.a)
        self.L, self.h0, self.h_end = lim.L, lim.h0, lim.h_end
        self.plot = _fitted_plot(theme, LimitAtHole.plot_box(), x0, x1, lim.y0, lim.y1)
        self.col = LimitAtHole.column_box()
        self.c_fn, self.c_text = role(theme, "function"), role(theme, "text")
        self.c_alert, self.c_guide = role(theme, "alert"), role(theme, "guide")
        self.ax_x = min(max(0.0, self.plot.x0), self.plot.x1)
        self._layout_rows()

    # the picture -------------------------------------------------------------------

    def axes(self):
        return label(self.plot.axes, "axes")

    def curve(self):
        """f on each side of the hole, each piece ending where the points'
        slide ends (a -+ h_end, where both readouts show L): never sampled at
        a itself, where numpy can return a value f does not have ((1 + x)^(1/x)
        gives 1 there; its limit is e), and reaching the hole however steep f
        is there (x log|x|)."""
        d = self.h_end
        return label(VGroup(*self.plot.curve(self.f, self.c_fn, b=self.a - d),
                            *self.plot.curve(self.f, self.c_fn, a=self.a + d)), "curve f")

    def hole(self):
        """The open circle at (a, L), above the points: f has no value there,
        however close the points come."""
        c = self.plot.hole(self.a, self.L, self.c_alert, radius=HOLE_RADIUS)
        c.set_z_index(POINT_Z + 1)
        return label(c, "hole")

    def dot(self, s: int, h: float):
        x = self.a + s * h
        return label(self.plot.dot(x, self.f.at(x), self.c_text, radius=DOT_RADIUS),
                     f"point {_side_name(s)}")

    def _dash(self, p0, p1, color: str):
        if np.linalg.norm(p1 - p0) < MIN_GUIDE:
            return VMobject()
        return DashedLine(p0, p1, dash_length=0.08, color=color, stroke_width=2)

    def vguide(self, s: int, h: float):
        """Dashed drop from the point to the x-axis."""
        x = self.a + s * h
        base = min(max(0.0, self.plot.y0), self.plot.y1)
        return self._dash(self.plot.p(x, base), self.plot.p(x, self.f.at(x)), self.c_guide)

    def hguide(self, s: int, h: float):
        """Dashed line from the point across to the y-axis."""
        x = self.a + s * h
        y = self.f.at(x)
        return self._dash(self.plot.p(self.ax_x, y), self.plot.p(x, y), self.c_guide)

    def l_guide(self):
        """L carried from the hole across to the y-axis, in f's colour."""
        return label(self._dash(self.plot.p(self.a, self.L), self.plot.p(self.ax_x, self.L),
                                self.c_fn), "limit guide")

    def l_tick(self):
        """L marked on the y-axis: a short bar across it, wider than the hole,
        so it shows even when the hole sits on the axis (a = 0). Where the
        axis runs along the plot's edge, the bar stays inside the plot, clear
        of the tick labels beyond that edge."""
        at = self.plot.p(self.ax_x, self.L)
        if self.ax_x == self.plot.x0:
            left, right = 0.0, 2 * TICK_HALF
        elif self.ax_x == self.plot.x1:
            left, right = 2 * TICK_HALF, 0.0
        else:
            left = right = TICK_HALF
        bar = Line(at + LEFT * left, at + RIGHT * right, color=self.c_fn, stroke_width=6)
        return label(bar, "L on the y-axis")

    # the maths column -------------------------------------------------------------

    def legend(self):
        return label(math(self.f.tex("f"), self.theme, size=self.theme.type.body,
                          color=self.c_fn, what="legend"), "legend")

    def undefined_line(self):
        return label(math(_undefined_tex(self.a), self.theme,
                          size=self.theme.type.body, color=self.c_alert, what="undefined"),
                     "f(a) undefined")

    def question(self):
        return label(math(_question_tex(self.a), self.theme,
                          size=self.theme.type.body, color=self.c_text, what="question"),
                     "question")

    def _x_readout(self, x: float):
        # The phantom gives "x =" the height of "f(x) =", so readout() centres
        # both numbers at the same height and the digits share a baseline.
        return readout(self.theme, r"\vphantom{f(x)}x =", x, self.c_text, decimals=self.lim.dx)

    def _f_readout(self, x: float):
        return readout(self.theme, "f(x) =", self.f.at(x), self.c_fn, decimals=self.lim.dy)

    def _layout_rows(self) -> None:
        """Fix the row layout once, from the widest numbers the slide will
        show: the f readout sits at one offset and every row is scaled by one
        factor, so a row is the same in every phase whatever its numbers."""
        a, h0, h1, lim = self.a, self.h0, self.h_end, self.lim
        hs = h0 * (h1 / h0) ** np.linspace(0, 1, 120)
        xs = np.concatenate([a - hs, a + hs])
        fs = self.f.np(xs)

        def widest(vals, d):
            strs = [f"{v:,.{d}f}" for v in vals]
            n = max(len(s) for s in strs)
            pick = [v for v, s in zip(vals, strs) if len(s) == n]
            neg = [v for v in pick if v < 0]
            return (neg or pick)[0]

        wx = self._x_readout(widest(xs, lim.dx)).width
        wf = readout(self.theme, "f(x) =", widest(fs, lim.dy), self.c_fn, decimals=lim.dy).width
        self.f_offset = wx + ROW_GAP
        full = self.f_offset + wf
        self.row_scale = min(1.0, 0.995 * self.col.width / full)

    def row(self, s: int, h: float):
        """`x = ...   f(x) = ...` for the point on side s at gap h."""
        x = self.a + s * h
        xr, fr = self._x_readout(x), self._f_readout(x)
        fr.shift(xr[0].get_left() + RIGHT * self.f_offset - fr[0].get_left())
        fr.shift(UP * (xr[0][0][-1].get_y() - fr[0][0][-1].get_y()))  # the "=" signs level
        r = VGroup(xr, fr)
        if self.row_scale < 1.0:
            scale_with_tags(r, self.row_scale)
        return label(r, f"readouts {_side_name(s)}")

    def limit_line(self):
        """lim_{x -> a} f(x) = L, with L (computed) in f's colour."""
        value = _value_tex(self.lim)
        m = math(_limit_tex(self.a, value), self.theme,
                 size=self.theme.type.body, color=self.c_text, what="limit")
        n = len(math(value.split(" ", 1)[1], self.theme, size=self.theme.type.body)[0])
        m[0][-n:].set_color(self.c_fn)
        m.value_part = m[0][-n:]  # type: ignore[attr-defined]
        m.head_part = m[0][:-n]   # type: ignore[attr-defined]
        return label(m, "limit")

    def column(self, *lines):
        """The maths column, top-left aligned: the same lines always land in
        the same place, whatever follows them."""
        return column_lines(self.theme, list(lines), self.col)


@register
class LimitAtHole(Animation):
    name = "LimitAtHole"
    Params = LimitHoleParams

    PHASES = {
        "pose": (Moment("curve", 2, motion=2.0), Moment("hole", 1.5, motion=1.0),
                 Moment("question", 1.5, motion=1.0)),
        "approach": (Moment("points", 1.5, motion=1.2), Moment("slide", 4),
                     Moment("agree", 1.5, motion=1.0)),
        "name": (Moment("limit", 2, motion=1.8), Moment("value", 1.5, motion=1.2)),
    }

    def latex_strings(self) -> list[str]:
        p = self.params
        lim = _analyse(p.function, float(p.x_range[0]), float(p.x_range[1]), float(p.a))
        return [Fn(p.function).tex("f"), _undefined_tex(p.a), _question_tex(p.a),
                _limit_tex(p.a, _value_tex(lim))]

    # state at the END of each phase, which is the START of the next -------------

    def _state(self, pic: _Picture, k: int) -> dict[str, Any]:
        """Name -> mobject on screen when phase k has finished (k = -1: none),
        in drawing order: axes, curve, guides, points, hole, then the column."""
        if k < 0:
            return {}
        s: dict[str, Any] = {"axes": pic.axes(), "curve": pic.curve()}
        if k == 0:
            s["hole"] = pic.hole()
            s["col"] = pic.column(pic.legend(), pic.undefined_line(), pic.question())
            return s
        h = pic.h_end
        if k == 1:
            s.update(vL=pic.vguide(-1, h), vR=pic.vguide(1, h), hL=pic.hguide(-1, h),
                     hR=pic.hguide(1, h), dL=pic.dot(-1, h), dR=pic.dot(1, h))
            s["hole"] = pic.hole()
            s["col"] = pic.column(pic.legend(), pic.undefined_line(), pic.row(-1, h),
                                  pic.row(1, h))
            return s
        s.update(lg=pic.l_guide(), tick=pic.l_tick(), hole=pic.hole())
        s["col"] = pic.column(pic.legend(), pic.undefined_line(), pic.row(-1, h),
                              pic.row(1, h), pic.limit_line())
        return s

    def enter(self, scene: ChalkdustScene) -> None:
        self._pic = _Picture(self.params, scene.theme)
        self._on = self._state(self._pic, self.phase_index() - 1)
        if self._on:
            scene.add(*self._on.values())

    def perform(self, scene: ChalkdustScene, tl: Timeline) -> None:
        getattr(self, f"_{self.params.phase}")(scene, tl, self._pic)

    @staticmethod
    def _settle_into(scene: ChalkdustScene, end: dict[str, Any]) -> None:
        """Swap whatever the phase animated for its end state's own mobjects,
        which sit exactly where those pieces ended: the next phase starts from
        _state(k), so this phase must end on it."""
        scene.remove(*scene.mobjects)
        scene.add(*end.values())

    # phases ------------------------------------------------------------------------

    def _pose(self, scene, tl, pic: _Picture) -> None:
        end = self._state(pic, 0)
        legend, undefined, question = end["col"]
        tl.sequence("curve", [([FadeIn(end["axes"])], 1),
                              ([Create(end["curve"]), FadeIn(legend)], 3)])
        tl.play("hole", GrowFromCenter(end["hole"]), Write(undefined))
        tl.sequence("question", [([Write(question)], 1),
                                 ([Indicate(end["hole"], color=pic.c_alert)], 1)])
        self._settle_into(scene, end)

    def _approach(self, scene, tl, pic: _Picture) -> None:
        on = self._on
        a, h0, h1 = pic.a, pic.h0, pic.h_end
        question = on["col"][2]
        rows = (pic.row(-1, h0), pic.row(1, h0))
        start = pic.column(pic.legend(), pic.undefined_line(), *rows)
        rows = (start[2], start[3])
        first = {"vL": pic.vguide(-1, h0), "vR": pic.vguide(1, h0), "hL": pic.hguide(-1, h0),
                 "hR": pic.hguide(1, h0)}
        dots = (pic.dot(-1, h0), pic.dot(1, h0))
        tl.play("points", *[Create(g) for g in first.values() if _drawn(g)],
                *[GrowFromCenter(d) for d in dots], FadeOut(question),
                *[FadeIn(r) for r in rows])

        t = ValueTracker(0.0)

        def h_now() -> float:
            return h0 * (h1 / h0) ** t.get_value()

        moving = {
            "vL": always_redraw(lambda: pic.vguide(-1, h_now())),
            "vR": always_redraw(lambda: pic.vguide(1, h_now())),
            "hL": always_redraw(lambda: pic.hguide(-1, h_now())),
            "hR": always_redraw(lambda: pic.hguide(1, h_now())),
            "dL": always_redraw(lambda: pic.dot(-1, h_now())),
            "dR": always_redraw(lambda: pic.dot(1, h_now())),
        }
        # The readouts are the very rows the points moment put up; only their
        # numbers follow the points now (plotting.live), at the points' x.
        (xl, fl), (xr, fr) = rows
        live(xl, lambda: a - h_now())
        live(fl, lambda: pic.f.at(a - h_now()))
        live(xr, lambda: a + h_now())
        live(fr, lambda: pic.f.at(a + h_now()))
        scene.remove(*first.values(), *dots)
        scene.add(*moving.values())
        tl.play("slide", t.animate.set_value(1.0), rate_func=linear)
        for m in [*moving.values(), xl[1], fl[1], xr[1], fr[1]]:
            m.clear_updaters()
        tl.play("agree", *[_pulse(r[1], r[0].get_right()[0] + PULSE_GAP, pic.col.right)
                           for r in (fl, fr)])
        self._settle_into(scene, self._state(pic, 1))

    def _name(self, scene, tl, pic: _Picture) -> None:
        on, end = self._on, self._state(pic, 2)
        rows = (on["col"][2], on["col"][3])
        limit = end["col"][4]
        # Both readouts' numbers flow into the one value: each its own copy
        # and target, since a transform animates its target and two would
        # fight over one. The readouts' own numbers hide while their copies
        # still cover them, then fade back: every frame before and at the
        # start of the move draws each number once, exactly as the previous
        # phase ended. The copies go on screen now, with the hiding: a cued
        # first moment holds before it plays (Timeline's lead), and a number
        # hidden with no copy up yet is a blank readout for all that hold.
        nums = [r[1][1] for r in rows]
        movers = [n.copy() for n in nums]
        for n in nums:
            n.set_opacity(0)
        scene.add(*movers)
        spare = limit.value_part.copy()
        tl.play("limit", FadeIn(limit.head_part),
                ReplacementTransform(movers[0], limit.value_part),
                ReplacementTransform(movers[1], spare),
                *[n.animate.set_opacity(1) for n in nums])
        scene.remove(spare)
        merge = [FadeOut(on[k]) for k in ("vL", "vR", "dL", "dR") if _drawn(on[k])]
        guides = [on[k] for k in ("hL", "hR") if _drawn(on[k])]
        # The two guides across to the y-axis become one, at L, in f's colour.
        if _drawn(end["lg"]) and guides:
            merge.append(ReplacementTransform(VGroup(*guides), end["lg"]))
        else:  # the hole sits on (or at) the y-axis: nothing to carry across
            merge += [FadeOut(g) for g in guides]
            if _drawn(end["lg"]):
                merge.append(Create(end["lg"]))
        tl.play("value", *merge, GrowFromCenter(end["tick"]),
                _pulse(limit.value_part, limit.head_part.get_right()[0] + PULSE_GAP,
                       pic.col.right))
        self._settle_into(scene, end)

    # fixtures -------------------------------------------------------------------------

    @classmethod
    def examples(cls) -> list[dict[str, Any]]:
        setup = {"function": "(x^2 - 1)/(x - 1)", "x_range": [-1, 3], "a": 1}
        out = [{"phase": ph, **setup} for ph in cls.PHASES]
        out.append({"phase": "approach", "function": "sin(x)/x", "x_range": [-10, 10], "a": 0})
        out.append({"phase": "name", "function": "(sqrt(x) - 2)/(x - 4)", "x_range": [0, 9],
                    "a": 4})
        return out

    @classmethod
    def stress(cls) -> list[dict[str, Any]]:
        return [
            # Steep near a: slope 18 at the hole, and the window runs to 400.
            {"phase": "approach", "function": "(exp(6*x) - 1)/x", "x_range": [-1, 1], "a": 0},
            # A huge y-range: L = 4,000,000, readouts with thousands commas.
            {"phase": "name", "function": "1000000*(x^2 - 4)/(x - 2)", "x_range": [0, 4], "a": 2},
            # The hole 12% from the left edge; and one next to an asymptote.
            {"phase": "approach", "function": "(x^2 - 1)/(x - 1)", "x_range": [0.7, 3.2], "a": 1},
            {"phase": "name", "function": "(x^2 - 1)/(x^2 - x)", "x_range": [-1, 3], "a": 1},
            # A tiny x-range: f is flat at this scale, and the readouts say so;
            # the hole on the y-axis, so L has no guide to carry across.
            {"phase": "name", "function": "sin(x)/x", "x_range": [-0.002, 0.002], "a": 0},
        ]
