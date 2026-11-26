"""RiemannSum: the definite integral as the limit of rectangle sums.

The picture: a curve f and the region under it between a and b ("how big is
this area?"). Rectangles stand on [a, b], each as tall as f at its sample
point (left end, right end or midpoint), and their total area is a live
number. Wherever a rectangle and the curve disagree (the rectangle's top above
the curve, or the curve above the top) is red, in one colour either way: that
is the gap, and its number is the total red area drawn. Then every rectangle
splits in two, in place, again and again, and the red shrinks with the gap.
Only then the notation: the last sum as a sum over its n rectangles, and its
limit, the integral, with its value.

Phases (each one beat; animation.py):
  pose    curve -> interval -> question
  boxes   rects -> sum -> gap
  refine  split -> more -> settle
  name    sigma -> integral

Setup: `function` (an expression in x), `x_range`, `a`, `b`, `rule` (left,
right or mid) and `n0` (rectangles at first; refine doubles them DOUBLINGS
times). Every number on screen is computed from these (screen_numbers):
  - each rectangle's height is f at its sample point;
  - each sum is the sum of the heights drawn times their width;
  - the gap is the area of the red polygons drawn, from the very vertices they
    are drawn with (_shapes);
  - the integral is adaptive quadrature, vouched for here (_integral): its
    error estimate must be under the last digit shown, and it must agree with
    a Simpson sum on a grid finer than the curve's sampling.
A spec cannot type any of them.

Area, not signed area. f < 0 anywhere on [a, b] is refused. The picture asks
"how big is this area?" and the last phase answers it with the integral; where
f < 0 the integral subtracts, so that answer would be false unless the
words changed with the setup. Signed area is its own idea and would be a
second idea in one beat (the brief: one idea per beat). The refusal names the
fix: an interval where f >= 0, or f shifted up.

Also refused, as param errors before anything is drawn:
  - f undefined or broken anywhere on [a, b];
  - a pole or a feature narrower than the curve's sampling (the curve drawn
    would leave f by more than CURVE_TOL_PX);
  - an integral the quadrature cannot vouch for;
  - an empty or reversed interval, or one outside the window;
  - an interval too narrow to read;
  - rectangles that would end thinner than MIN_RECT_PX at 1080p;
  - an area too flat to see against the rest of f;
  - numbers that would read 0 at the most decimals shown.
"""

from __future__ import annotations

import math as _math
from functools import lru_cache
from typing import Annotated, Any, Literal, NamedTuple

import numpy as np
from manim import (
    DOWN,
    LEFT,
    AnimationGroup,
    Create,
    FadeIn,
    FadeOut,
    GrowFromEdge,
    LaggedStart,
    Line,
    Polygon,
    ReplacementTransform,
    ValueTracker,
    VGroup,
    VMobject,
    Write,
    always_redraw,
    config,
    linear,
    smooth,
    squish_rate_func,
    there_and_back,
)
from pydantic import Field, model_validator

from chalkdust.scenes.base import ChalkdustScene, Moment, Timeline
from chalkdust.scenes.components.animation import Animation, AnimationParams
from chalkdust.scenes.components.base import label, register
from chalkdust.scenes.plotting import (Expr, Fn, Plot, column_lines, live, readout, role, seat,
                                      y_window)
from chalkdust.scenes.theme import Theme, math

FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]

# How many times the refine phase doubles n: 4 -> 8 -> 16 -> 32 -> 64.
DOUBLINGS = 4
# The thinnest rectangle the last doubling may draw, in pixels at 1080p; under
# about 2 px a rectangle is its own outline and the picture is a smear.
MIN_RECT_PX = 2.0
# Share of plot_box()'s width the plot rect keeps after its tick labels. The
# real rect is only known once the labels are typeset; wide labels (2.5 x 10^7)
# leave about 0.77, so 0.75 errs on the side of refusing.
PLOT_WIDTH_AFTER_TICKS = 0.75
MAX_N0 = 64
MAX_SPAN = 1e4
# [a, b] must be at least this share of the x-range, and the tallest point of
# the area at least this share of the y-window, or the area is a sliver.
MIN_AB_SHARE = 0.1
MIN_HEIGHT_SHARE = 0.1
# f below -NEG_TOL * max(1, max f) counts as negative (float noise at a zero
# of f, x^2 at 0, does not).
NEG_TOL = 1e-9
GRID = 2001
# The most decimals a readout shows (_decimals).
MAX_DECIMALS = 6

# Samples of the curve across x_range (Plot.curve's n), about one per pixel
# of the plot at 1080p. Every other edge that follows f -- the area's, and the
# tops of the rectangles' fills and of the red -- is sampled at least this
# densely, and every one of them is checked against f (_survey).
CURVE_SAMPLES = 960
# f is checked on a grid FINE times denser than the curve's sampling.
FINE = 8
FINE_MAX = 40001
# The curve drawn through its samples must stay this close to f, in pixels at
# 1080p measured across the line: under half the curve's stroke (5, about
# 6.75 px), so the true f lies inside the line drawn. Further off is a pole or
# a feature narrower than the sampling, and the picture would not be f.
CURVE_TOL_PX = 3.0
# The quadrature's error estimate may be at most this share of the last digit
# a readout shows...
QUAD_TOL = 0.01
# ...and it must agree with a Simpson sum on the fine grid to this share of
# the area's bounding box (max f times b - a), or one of them missed part of f.
AGREE = 1e-6

# Share of one doubling spent drawing the cut down each rectangle's middle;
# the rest is the two halves finding their own heights.
CUT = 0.3
RECT_OPACITY = 0.45
SLIVER_OPACITY = 0.6
SLIVER_PULSE_OPACITY = 0.9
REGION_OPACITY = 0.35
# Rectangle outline: a share of the rectangle's width (stroke_width 1 is 0.01
# scene units in Manim's Cairo camera), within these bounds, so the outline
# never swallows a thin rectangle's fill.
STROKE_PER_UNIT = 12.0
STROKE_MIN, STROKE_MAX = 0.6, 2.5
# Emphasis pulse on a column line: at most this scale, grown from its left
# edge and never wider than the column (_Picture.emphasis).
EMPHASIS_SCALE = 1.12
# Between the limit line and the "= integral" line under it.
BLOCK_GAP = 0.12

# Draw order, fixed by z so a redrawn piece and the end state stack the same
# way whatever order they were added in (continuity, animation.py). The fills
# stop at the curve and the red starts there, so no red is ever drawn over
# green; outlines and cuts go over both.
Z_REGION, Z_GUIDE, Z_FILL, Z_SLIVER, Z_OUTLINE, Z_CUT, Z_CURVE = (
    0.1, 0.2, 0.3, 0.4, 0.45, 0.46, 1.0)

SUM_TEX = {"left": r"\text{left sum} =", "right": r"\text{right sum} =",
           "mid": r"\text{midpoint sum} ="}
GAP_TEX = r"\text{gap} ="
N_TEX = "n ="
QUESTION_TEX = r"\text{area} = \,?"
# The limit and the integral it equals, under the display-style sigma line:
# text-style operators keep the column inside its height with every legend.
LIMIT_TEX = r"\lim_{n \to \infty} \textstyle\sum_{i=1}^{n} f(x_i)\,\Delta x"


def _num(v: float) -> str:
    """A setup value as written: integers without decimals."""
    return str(int(v)) if float(v).is_integer() else f"{v:g}"


def _sigma_tex(n: int) -> str:
    """The last sum, over the n rectangles it has: a finite sum, not the limit."""
    return rf"\sum_{{i=1}}^{{{n}}} f(x_i)\,\Delta x ="


def _integral_tex(a: float, b: float) -> str:
    return rf"= \textstyle\int_{{{_num(a)}}}^{{{_num(b)}}} f(x)\,dx ="


def _edges(a: float, b: float, n: int) -> np.ndarray:
    return np.linspace(a, b, n + 1)


def _samples(a: float, b: float, n: int, rule: str) -> np.ndarray:
    """Where each of the n rectangles meets the curve."""
    e = _edges(a, b, n)
    if rule == "left":
        return e[:-1]
    if rule == "right":
        return e[1:]
    return (e[:-1] + e[1:]) / 2


def _cells(f: Fn, a: float, b: float, n: int, rule: str) -> np.ndarray:
    """The n rectangles as rows (x_left, x_right, height = f at the sample)."""
    e = _edges(a, b, n)
    return np.column_stack([e[:-1], e[1:], f.np(_samples(a, b, n, rule))])


def riemann_sum(f: Fn, a: float, b: float, n: int, rule: str) -> float:
    """Σ f(x_i) Δx with n equal rectangles. The same arithmetic as
    _Picture.area_of() on the drawn cells, so the static and the live readout
    agree to the bit."""
    return float(f.np(_samples(a, b, n, rule)).sum() * (b - a) / n)


def _values_on(f: Fn, a: float, b: float, n0: int, rule: str) -> tuple[np.ndarray, np.ndarray]:
    """f on a dense grid of [a, b] and at every sample point of every level."""
    xs = [np.linspace(a, b, GRID)]
    xs += [_samples(a, b, n0 * 2 ** k, rule) for k in range(DOUBLINGS + 1)]
    x = np.concatenate(xs)
    return x, f.np(x)


def _rect_px(x0: float, x1: float, a: float, b: float, n: int) -> float:
    """Width of one of n rectangles on [a, b], in pixels at 1080p."""
    plot_w = Animation.plot_box().width * PLOT_WIDTH_AFTER_TICKS
    return plot_w * (b - a) / (x1 - x0) / n * (1080 / config.frame_height)


def _decimals(m: float) -> int:
    """Decimals for readouts whose values reach m: two for everyday sizes,
    none for thousands and up, more for small areas (so 0.004 is not 0.00)."""
    if not m > 0:
        return 2
    e = _math.floor(_math.log10(m))
    if e >= 3:
        return 0
    if e >= 0:
        return 2
    return min(MAX_DECIMALS, 2 - e)


def _curve_dx(x0: float, x1: float) -> float:
    """Spacing of the curve's samples (CURVE_SAMPLES across x_range)."""
    return (x1 - x0) / (CURVE_SAMPLES - 1)


def _fine_grid(x0: float, x1: float, a: float, b: float) -> np.ndarray:
    n = int(min(FINE_MAX, _math.ceil((b - a) * FINE / _curve_dx(x0, x1)) + 1))
    return np.linspace(a, b, max(n, 3))


# the red: between each rectangle's top and the curve -------------------------------

def _tops(cells: np.ndarray, dx: float) -> np.ndarray:
    """x across each cell, one row per cell: an odd count (a midpoint sample
    is a vertex), spaced at most dx, so these edges follow f as closely as
    the curve does."""
    w = float(np.max(cells[:, 1] - cells[:, 0]))
    m = max(3, 2 * _math.ceil(w / (2 * dx)) + 1)
    t = np.linspace(0.0, 1.0, m)
    return cells[:, :1] + (cells[:, 1:2] - cells[:, :1]) * t


# Gauss-Legendre, 3 points on [-1, 1]: exact for cubics on each piece.
_GL_T = np.array([-_math.sqrt(0.6), 0.0, _math.sqrt(0.6)])
_GL_W = np.array([5.0, 8.0, 5.0]) / 9.0


def _red_terms(f: Fn, cells: np.ndarray, dx: float):
    """f across each cell, where f crosses each rectangle's top, and the red
    area: ∫ |f - height| over each cell, the gross error.

    The red is drawn with straight edges between the samples (the vertices
    _shapes uses), which stay within CURVE_TOL_PX of f, under the curve's
    stroke. The number is the area those edges approximate, to the digit:
    between each pair of samples, split where f crosses the top (or at the
    middle, if it does not), each one-signed piece by Gauss-Legendre."""
    x = _tops(cells, dx)
    y = f.np(x)
    h = cells[:, 2:3]
    d = y - h
    d0, d1 = d[:, :-1], d[:, 1:]
    s = np.abs(d0) + np.abs(d1)
    cross = d0 * d1 < 0
    xa, xb = x[:, :-1], x[:, 1:]
    xm = np.where(cross, xa + (xb - xa) * (np.abs(d0) / np.where(cross, s, 1.0)), (xa + xb) / 2)
    area = np.zeros_like(xa)
    for lo, hi in ((xa, xm), (xm, xb)):
        mid, half = (lo + hi) / 2, (hi - lo) / 2
        for t, w in zip(_GL_T, _GL_W):
            area += w * half * np.abs(f.np(mid + half * t) - h)
    return x, y, d0, s, cross, area


def _red_area(f: Fn, cells: np.ndarray, dx: float) -> float:
    return float(_red_terms(f, cells, dx)[5].sum())


class _Shapes(NamedTuple):
    """One frame of rectangles, in data coordinates: per cell the green part
    (the rectangle up to the curve, never above it) and the red part
    (between the top and the curve), and the total red area."""

    fills: list[np.ndarray]
    reds: list[np.ndarray]
    gap: float


def _shapes(f: Fn, cells: np.ndarray, dx: float) -> _Shapes:
    """The polygons drawn and the gap readout's number, from one set of
    vertices: the number is the area of the red as drawn."""
    x, y, d0, s, cross, area = _red_terms(f, cells, dx)
    fills, reds = [], []
    for i, (xl, xr, h) in enumerate(cells):
        xi, yi = x[i], y[i]
        c = np.nonzero(cross[i])[0]
        if c.size:  # put the crossings in as vertices, so each piece is one-signed
            xc = xi[c] + (xi[c + 1] - xi[c]) * (np.abs(d0[i, c]) / s[i, c])
            xi, yi = np.insert(xi, c + 1, xc), np.insert(yi, c + 1, h)
        fills.append(np.column_stack([np.r_[xl, xi, xr], np.r_[0.0, np.minimum(yi, h), 0.0]]))
        reds.append(np.column_stack([np.r_[xi, xr, xl], np.r_[yi, h, h]]))
    return _Shapes(fills, reds, float(area.sum()))


def riemann_gap(f: Fn, a: float, b: float, n: int, rule: str,
                x_range: tuple[float, float]) -> float:
    """The gap the screen shows with n rectangles: the red area drawn."""
    return _red_area(f, _cells(f, a, b, n, rule), _curve_dx(*x_range))


# the checks ---------------------------------------------------------------------------

def _drawn_nodes(x_range: tuple[float, float], a: float, b: float,
                 levels: tuple[int, ...]) -> list[np.ndarray]:
    """Every set of x at which an edge following f is drawn on [a, b]: the
    curve's samples, the area's, and the tops of each level's cells (a split
    uses the next level's edges)."""
    x0, x1 = x_range
    dx = _curve_dx(x0, x1)
    xc = np.linspace(x0, x1, CURVE_SAMPLES)
    sets = [np.concatenate([[a], xc[(xc > a) & (xc < b)], [b]]), _area_xs(a, b, dx)]
    for n in levels:
        e = _edges(a, b, n)
        sets.append(np.unique(_tops(np.column_stack([e[:-1], e[1:], e[1:]]), dx)))
    return sets


def _area_xs(a: float, b: float, dx: float) -> np.ndarray:
    """Where the area's edge samples f (Plot.area's n): at most dx apart."""
    return np.linspace(a, b, _math.ceil((b - a) / dx) + 1)


def _off_curve_px(f: Fn, x_range: tuple[float, float], nodes: np.ndarray,
                  xf: np.ndarray, yf: np.ndarray,
                  window: tuple[float, float]) -> tuple[float, float]:
    """How far, in px at 1080p, f on the fine grid lies from the straight
    pieces drawn between f at `nodes`, measured across the line (a steep
    stretch is not off just because it is steep); and where. The scale is the
    whole plot box's, which the plot rect never exceeds, so this errs high."""
    x0, x1 = x_range
    y0, y1 = window
    yn = f.np(nodes)
    if not np.isfinite(yn).all():
        return float("inf"), float(nodes[~np.isfinite(yn)][0])
    k = 1080 / config.frame_height
    box = Animation.plot_box()
    sx, sy = box.width * k / (x1 - x0), box.height * k / (y1 - y0)
    i = np.clip(np.searchsorted(nodes, xf, side="right") - 1, 0, len(nodes) - 2)
    ax, ay = nodes[i] * sx, yn[i] * sy
    ux, uy = (nodes[i + 1] - nodes[i]) * sx, (yn[i + 1] - yn[i]) * sy
    px, py = xf * sx - ax, yf * sy - ay
    t = np.clip((px * ux + py * uy) / np.maximum(ux * ux + uy * uy, 1e-300), 0.0, 1.0)
    dist = np.hypot(px - t * ux, py - t * uy)
    j = int(np.argmax(dist))
    return float(dist[j]), float(xf[j])


def _integral(f: Fn, a: float, b: float, xf: np.ndarray, yf: np.ndarray,
              scale: float, decimals: int) -> float:
    """∫_a^b f, vouched for to the last digit shown, or ValueError.

    Adaptive quadrature (scipy, a Manim dependency), told where f peaks so
    a narrow bump is not stepped over, and asked for a hundredth of the last
    digit shown. Refused when it reports trouble, when its own error estimate
    is not that small, or when a Simpson sum on the fine grid disagrees."""
    from scipy.integrate import quad, simpson

    tol = QUAD_TOL * 10.0 ** -decimals
    top = float(xf[int(np.argmax(yf))])
    res = quad(lambda t: f.at(t), a, b, points=[top] if a < top < b else None,
               limit=500, epsabs=tol, epsrel=1e-10, full_output=1)
    value, err = float(res[0]), float(res[1])
    trouble = res[3] if len(res) > 3 else None
    if trouble is not None or not np.isfinite(value) or err > tol:
        why = str(trouble).split("\n")[0].strip() if trouble else f"error estimate {err:.2g}"
        raise ValueError(f"the integral of f over [a, b] cannot be computed to the "
                         f"{decimals} decimals shown ({why}); pick an interval where f is smooth")
    grid = float(simpson(yf, x=xf))
    if abs(grid - value) > AGREE * scale + tol:
        raise ValueError(f"the integral of f over [a, b] is unclear: quadrature gives "
                         f"{value:.6g}, a sum on a fine grid {grid:.6g}; f has a feature too "
                         "narrow to sample. Pick an interval away from it")
    return value


def _flat_fix(function: str, f: Fn, x_range: tuple[float, float], a: float, b: float,
              lo: float, hi: float) -> str:
    """A change that would make the area tall enough, checked, for the
    refusal to name."""
    def tall(fn: Fn, x0: float, x1: float, k: float) -> bool:
        y0, y1 = y_window([fn], x0, x1, extra=[lo * k, hi * k])
        return hi * k >= MIN_HEIGHT_SHARE * (y1 - y0)

    if (a, b) != tuple(x_range) and tall(f, a, b, 1.0):
        return f"narrow x_range to [{a:g}, {b:g}]"
    for p in range(1, 13):
        if tall(Fn(f"10^{p}*({function})"), *x_range, 10.0 ** p):
            return f"scale f up by a power of ten, e.g. '10^{p}*({function})'"
    return "pick an f that varies on [a, b], or another [a, b]"


class _Numbers(NamedTuple):
    """Every number the screen shows for a setup, and the window it is drawn
    in. Shared by the checks, the picture and the harness (screen_numbers)."""

    levels: tuple[int, ...]
    sums: tuple[float, ...]
    gaps: tuple[float, ...]
    true: float
    lo: float
    hi: float
    window: tuple[float, float]
    widest: float
    decimals: int


@lru_cache(maxsize=64)
def _survey(function: str, x_range: tuple[float, float], a: float, b: float, n0: int,
            rule: str) -> _Numbers:
    """The setup's numbers, or ValueError naming what is wrong and a fix."""
    f = Fn(function)
    x0, x1 = x_range
    xs, ys = _values_on(f, a, b, n0, rule)
    xf = _fine_grid(x0, x1, a, b)
    yf = f.np(xf)
    x_all, y_all = np.concatenate([xs, xf]), np.concatenate([ys, yf])
    bad = ~np.isfinite(y_all)
    if bad.any():
        raise ValueError(f"{function!r} is undefined at x = {float(x_all[bad].min()):g}, inside "
                         "[a, b]; pick an interval where f is defined")
    if not f.continuous_on(a, b):
        raise ValueError(f"{function!r} jumps between a = {a:g} and b = {b:g}; "
                         "pick an interval where f is continuous")
    lo, hi = float(y_all.min()), float(y_all.max())
    if lo < -NEG_TOL * max(1.0, abs(hi)):
        i = int(np.argmin(y_all))
        raise ValueError(f"{function!r} is negative on [a, b] (f({x_all[i]:g}) = "
                         f"{y_all[i]:.3g}); this animation shows the area between f and the "
                         "x-axis, so f must be >= 0 there: choose [a, b] where f >= 0, "
                         "or add a constant to f")
    window = y_window([f], x0, x1, extra=[lo, hi])
    y0, y1 = window
    if hi < MIN_HEIGHT_SHARE * (y1 - y0):
        raise ValueError(f"the area under {function!r} on [a, b] would be under "
                         f"{MIN_HEIGHT_SHARE:.0%} of the plot's height (f reaches {hi:.3g} "
                         f"there, the window is {y1 - y0:.3g} tall); "
                         f"{_flat_fix(function, f, x_range, a, b, lo, hi)}")
    levels = tuple(n0 * 2 ** k for k in range(DOUBLINGS + 1))
    off, at = max(_off_curve_px(f, x_range, nodes, xf, yf, window)
                  for nodes in _drawn_nodes(x_range, a, b, levels))
    if off > CURVE_TOL_PX:
        raise ValueError(f"{function!r} changes too fast near x = {at:.6g} to draw: the curve "
                         f"would pass {off:.1f} px from f at 1080p (at most {CURVE_TOL_PX:g}). "
                         "For a pole, keep [a, b] and x_range clear of it; for a narrow spike, narrow "
                         "x_range around it or widen the spike")
    scale = hi * (b - a)
    true = _integral(f, a, b, xf, yf, scale, _decimals(scale))
    sums = tuple(riemann_sum(f, a, b, n, rule) for n in levels)
    dx = _curve_dx(x0, x1)
    gaps = tuple(_red_area(f, _cells(f, a, b, n, rule), dx) for n in levels)
    # The largest value any readout can show (a sum is at most max f times
    # the width): readouts are laid out at it, so a live number never
    # outgrows the column.
    widest = max(scale, abs(true), *map(abs, sums))
    d = _decimals(widest)
    # Too small to read: even at the most decimals the answer or the first sum
    # would show as 0. (Below the cap a 0 is a true reading at the picture's
    # scale -- a sum whose rectangles miss a bump -- and scaling f would not
    # change it.)
    if d == MAX_DECIMALS and (round(true, d) == 0 or round(sums[0], d) == 0):
        p = max(1, _math.ceil(-_math.log10(widest)))
        raise ValueError(f"the numbers would read 0 even at {d} decimals (integral "
                         f"{true:.3g}, first sum {sums[0]:.3g}); scale f up by a power of "
                         f"ten, e.g. '10^{p}*({function})'")
    return _Numbers(levels, sums, gaps, true, lo, hi, window, widest, d)


def screen_numbers(p: RiemannSumParams) -> _Numbers:
    """The numbers a valid setup puts on screen, without a theme: the sums,
    the gaps (red area) and the integral at every level, and the decimals the
    readouts use. For the harness's facts."""
    return _survey(p.function, (float(p.x_range[0]), float(p.x_range[1])), float(p.a),
                   float(p.b), int(p.n0), p.rule)


class RiemannSumParams(AnimationParams):
    phase: Literal["pose", "boxes", "refine", "name"]
    function: Expr = Field(description="f as an expression in x, e.g. 'x^2/4 + 1'; "
                                       "f >= 0 on [a, b]")
    x_range: tuple[FiniteFloat, FiniteFloat] = Field(description="[min, max] of x shown")
    a: FiniteFloat = Field(description="left end of the interval under the curve")
    b: FiniteFloat = Field(description="right end of the interval (b > a)")
    rule: Literal["left", "right", "mid"] = Field(
        description="where each rectangle meets the curve: its left end, right end or midpoint")
    n0: int = Field(ge=1, le=MAX_N0, description=f"number of rectangles at first; the refine "
                                                 f"phase doubles it {DOUBLINGS} times")

    @model_validator(mode="after")
    def _true_picture(self) -> RiemannSumParams:
        x0, x1 = self.x_range
        a, b = self.a, self.b
        if not x1 > x0:
            raise ValueError(f"x_range {list(self.x_range)} must be [min, max] with min < max")
        if x1 - x0 > MAX_SPAN:
            raise ValueError(f"x_range is {x1 - x0:g} wide; keep it under {MAX_SPAN:g}")
        if not b > a:
            raise ValueError(f"[a, b] = [{a:g}, {b:g}] is empty or reversed; "
                             "b must be greater than a")
        if not (x0 <= a and b <= x1):
            raise ValueError(f"[a, b] = [{a:g}, {b:g}] must lie inside x_range "
                             f"{list(self.x_range)}")
        if b - a < MIN_AB_SHARE * (x1 - x0):
            raise ValueError(f"[a, b] is {b - a:g} wide, under {MIN_AB_SHARE:.0%} of the "
                             "x-range; narrow x_range around [a, b]")
        n_last = self.n0 * 2 ** DOUBLINGS
        px = _rect_px(x0, x1, a, b, n_last)
        if px < MIN_RECT_PX:
            fit = int(self.n0 * px / MIN_RECT_PX)
            fix = f"use n0 <= {fit}" if fit >= 1 else "widen [a, b]"
            raise ValueError(f"after {DOUBLINGS} doublings there are {n_last} rectangles, each "
                             f"about {px:.1f} px wide at 1080p (at least {MIN_RECT_PX:g} px "
                             f"needed); {fix}, or narrow x_range around [a, b]")
        screen_numbers(self)
        return self


def _baseline_tex(tex: str) -> str:
    """The label with a digit after it, which plotting.readout typesets to
    find the line's baseline: listed in latex_strings with the labels."""
    return tex + r"\,1"


def _z(mob, z: float):
    mob.set_z_index(z)
    return mob


def _pad(group: VGroup, size: int) -> VGroup:
    """Pad with empty pieces to a fixed count. A redrawn group then keeps one
    structure for the whole refine, so Mobject.become never has to invent
    submobjects to align two different counts."""
    while len(group) < size:
        group.add(VMobject())
    return group


class _Drawn(NamedTuple):
    """What the refine shows at one moment: the cells (x_left, x_right,
    height), how far the cut down each cell's middle has grown (0 for none),
    the n the readout shows, and the rectangles' stroke width."""

    cells: np.ndarray
    cut: float
    n: int
    stroke: float


class _Picture:
    """Every mobject of the animation, built from the setup alone, so each
    phase's start is exactly the previous phase's end (animation.py)."""

    def __init__(self, p: RiemannSumParams, theme: Theme) -> None:
        self.p, self.theme = p, theme
        self.f = Fn(p.function)
        self.a, self.b, self.rule = p.a, p.b, p.rule
        num = screen_numbers(p)
        self.levels = list(num.levels)
        self.sums, self.gaps, self.true = list(num.sums), list(num.gaps), num.true
        self.hi, self.widest, self.decimals = num.hi, num.widest, num.decimals
        x0, x1 = p.x_range
        self.dx = _curve_dx(x0, x1)
        # The window holds every height any rectangle reaches: f anywhere on
        # [a, b] (y_window's percentile trim must not clip a steep f).
        self.plot = Plot(theme, RiemannSum.plot_box(), x0, x1, *num.window)
        self.col = RiemannSum.column_box()
        self.c_fn, self.c_area = role(theme, "function"), role(theme, "area")
        self.c_alert, self.c_text = role(theme, "alert"), role(theme, "text")

    # cells: (x_left, x_right, height) per rectangle -----------------------------

    def cells(self, k: int) -> np.ndarray:
        """The rectangles of level k (n = n0 * 2^k), each as tall as f at its
        sample point."""
        return _cells(self.f, self.a, self.b, self.levels[k], self.rule)

    def split_cells(self, k: int, u: float) -> np.ndarray:
        """Level k's rectangles cut in two. Each half is as tall as f at a
        point sliding from its parent's sample point (u = 0: the parent's
        height, so the cut changes nothing) to its own (u = 1: level k+1).
        Every frame is still f at some point, so the live sum stays a sum
        of drawn heights."""
        n = self.levels[k]
        own = _samples(self.a, self.b, 2 * n, self.rule)
        if u >= 1.0:
            x = own
        else:
            parent = np.repeat(_samples(self.a, self.b, n, self.rule), 2)
            x = parent + (own - parent) * u
        e = _edges(self.a, self.b, 2 * n)
        return np.column_stack([e[:-1], e[1:], self.f.np(x)])

    def area_of(self, cells: np.ndarray) -> float:
        """Total area of equal-width cells: Σ height × Δx (riemann_sum's
        arithmetic)."""
        return float(cells[:, 2].sum() * (self.b - self.a) / len(cells))

    def shapes(self, cells: np.ndarray) -> _Shapes:
        return _shapes(self.f, cells, self.dx)

    def stroke(self, n: int) -> float:
        w = (self.b - self.a) / n * self.plot.sx
        return float(np.clip(w * STROKE_PER_UNIT, STROKE_MIN, STROKE_MAX))

    def drawn(self, s: float) -> _Drawn:
        """The refine at progress s in [0, DOUBLINGS]: in doubling k = floor(s),
        first the cut grows down each rectangle's middle, then the halves move
        to their own heights. Integer s is exactly level s."""
        s = min(max(float(s), 0.0), float(DOUBLINGS))
        k = min(int(s), DOUBLINGS - 1)
        t = s - k
        n0, n1 = self.levels[k], self.levels[k + 1]
        if t < CUT:
            return _Drawn(self.cells(k), t / CUT, n0, self.stroke(n0))
        u = smooth((t - CUT) / (1 - CUT))
        if u >= 1.0 - 1e-12:
            return _Drawn(self.cells(k + 1), 0.0, n1, self.stroke(n1))
        sw = self.stroke(n0) + (self.stroke(n1) - self.stroke(n0)) * u
        return _Drawn(self.split_cells(k, u), 0.0, n1, sw)

    # the picture -------------------------------------------------------------------

    def axes(self):
        return label(self.plot.axes, "axes")

    def curve(self):
        return label(_z(self.plot.curve(self.f, self.c_fn, n=CURVE_SAMPLES), Z_CURVE), "curve f")

    def region(self):
        n = len(_area_xs(self.a, self.b, self.dx))
        return label(_z(self.plot.area(self.f, self.a, self.b, self.c_area,
                                       opacity=REGION_OPACITY, n=n), Z_REGION), "area")

    def guides(self):
        g = VGroup(self.plot.vguide(self.a, self.f.at(self.a)),
                   self.plot.vguide(self.b, self.f.at(self.b)))
        return label(_z(g, Z_GUIDE), "a and b")

    def _polygon(self, v: np.ndarray, **style) -> Polygon:
        return Polygon(*self.plot.points(v[:, 0], v[:, 1]), **style)

    def boxes(self, cells: np.ndarray) -> VGroup:
        """Whole green rectangles, before the red is pointed out (boxes'
        first moments); they become fills() at the gap. Built on the fills'
        own vertices with every top vertex at the rectangle's height, so the
        change is each top vertex dropping straight down to the curve (where
        the rectangle overshoots) or staying put: no vertex wanders."""
        g = VGroup()
        for (_, _, h), v in zip(cells, self.shapes(cells).fills):
            whole = v.copy()
            whole[1:-1, 1] = h
            g.add(self._polygon(whole, stroke_width=0, fill_color=self.c_area,
                                fill_opacity=RECT_OPACITY))
        return label(_z(g, Z_FILL), "rectangles")

    def fills(self, shapes: _Shapes, pool: int = 0) -> VGroup:
        """Each rectangle's green, up to the curve and no higher: where the
        rectangle overshoots, that part is red instead, never red on green."""
        g = VGroup(*[self._polygon(v, stroke_width=0, fill_color=self.c_area,
                                   fill_opacity=RECT_OPACITY) for v in shapes.fills])
        return label(_z(_pad(g, pool), Z_FILL), "rectangles")

    def slivers(self, shapes: _Shapes, pool: int = 0) -> VGroup:
        """Between each rectangle's top and the curve, above it or below: the
        gap, in one colour. Its area is shapes.gap."""
        g = VGroup(*[self._polygon(v, stroke_width=0, fill_color=self.c_alert,
                                   fill_opacity=SLIVER_OPACITY) for v in shapes.reds])
        return label(_z(_pad(g, pool), Z_SLIVER), "gap slivers")

    def outlines(self, cells: np.ndarray, stroke: float, pool: int = 0) -> VGroup:
        """The rectangles themselves, as outlines over the green and the red."""
        g = VGroup(*[self._polygon(np.array([[xl, 0.0], [xl, h], [xr, h], [xr, 0.0]]),
                                   stroke_color=self.c_area, stroke_width=stroke,
                                   fill_opacity=0.0)
                     for xl, xr, h in cells])
        return label(_z(_pad(g, pool), Z_OUTLINE), "rectangle outlines")

    def cuts(self, cells: np.ndarray, frac: float, stroke: float, pool: int = 0) -> VGroup:
        """The cut down each rectangle's middle, grown `frac` of the way up."""
        g = VGroup()
        if frac > 0:
            for xl, xr, h in cells:
                top = h * frac
                if top * self.plot.sy < 1e-4:
                    g.add(VMobject())
                    continue
                xm = (xl + xr) / 2
                g.add(Line(self.plot.p(xm, 0.0), self.plot.p(xm, top), color=self.c_area,
                           stroke_width=stroke))
        return label(_z(_pad(g, pool), Z_CUT), "cuts")

    def level(self, k: int) -> tuple[VGroup, VGroup, VGroup]:
        cells = self.cells(k)
        sh = self.shapes(cells)
        return self.fills(sh), self.slivers(sh), self.outlines(cells, self.stroke(self.levels[k]))

    # the maths column ---------------------------------------------------------------

    def legend(self):
        return label(math(self.f.tex("f"), self.theme, size=self.theme.type.body,
                          color=self.c_fn, what="legend"), "legend")

    def interval(self):
        tex = rf"[a, b] = [{_num(self.a)}, {_num(self.b)}]"
        return label(math(tex, self.theme, size=self.theme.type.body, color=self.c_text,
                          what="interval"), "[a, b]")

    def question(self):
        return label(math(QUESTION_TEX, self.theme, size=self.theme.type.body,
                          color=self.c_area, what="question"), "question")

    def _readout(self, tex: str, value: float, widest: float, color: str, decimals: int,
                 name: str):
        """A readout laid out at its widest value, then set to `value` once
        placed (column()), so its size never depends on the value shown."""
        r = readout(self.theme, tex, widest, color, decimals=decimals)
        r._chalk_value = value  # type: ignore[attr-defined]
        return label(r, name)

    def n_line(self, n: int):
        return self._readout(N_TEX, n, self.levels[-1], self.c_text, 0, "n")

    def sum_line(self, value: float):
        return self._readout(SUM_TEX[self.rule], value, self.widest, self.c_area,
                             self.decimals, "sum")

    def gap_line(self, value: float):
        return self._readout(GAP_TEX, value, self.widest, self.c_alert, self.decimals, "gap")

    def sigma_line(self):
        """The last sum, named as the sum over the n rectangles drawn."""
        return self._readout(_sigma_tex(self.levels[-1]), self.sums[-1], self.widest,
                             self.c_area, self.decimals, "sigma")

    def integral_block(self):
        """The limit of those sums, and under it the integral it equals,
        with its value: two lines, one column item."""
        lim = label(math(LIMIT_TEX, self.theme, size=self.theme.type.body, color=self.c_area,
                         what="limit"), "limit")
        value = self._readout(_integral_tex(self.a, self.b), self.true, self.widest,
                              self.c_area, self.decimals, "integral")
        g = VGroup(lim, value).arrange(DOWN, aligned_edge=LEFT, buff=BLOCK_GAP)
        return label(g, "limit = integral")

    def emphasis(self, line):
        """A there-and-back pulse on a column line. Not Indicate: it scales
        by 1.2 about the centre, which pushes a wide line across the column
        gap into the plot. This grows from the line's left edge, only as far
        as the column allows, by a few hundredths vertically (the lines are
        0.35 apart), and ends exactly where it began."""
        k = max(1.0, min(EMPHASIS_SCALE, self.col.width / line.width))
        return line.animate(rate_func=there_and_back).scale(k, about_edge=LEFT)

    @staticmethod
    def pulse(slivers: VGroup):
        """The red brightens and returns: this is what the gap measures."""
        return slivers.animate(rate_func=there_and_back).set_fill(opacity=SLIVER_PULSE_OPACITY)

    def column(self, *lines):
        """The maths column, top-left aligned; positions depend only on which
        lines are there, so the same lines always land in the same place."""
        g = column_lines(self.theme, list(lines), self.col)
        for line in lines:
            for m in line.get_family():
                v = getattr(m, "_chalk_value", None)
                if v is not None:
                    m[1].set_value(v)
                    seat(m)
        return g


@register
class RiemannSum(Animation):
    name = "RiemannSum"
    Params = RiemannSumParams

    PHASES = {
        "pose": (Moment("curve", 2, motion=2.0), Moment("interval", 1.5, motion=1.2),
                 Moment("question", 1.5, motion=1.2)),
        "boxes": (Moment("rects", 2, motion=1.8), Moment("sum", 1.5, motion=1.0),
                  Moment("gap", 2, motion=1.8)),
        "refine": (Moment("split", 3), Moment("more", 4), Moment("settle", 1.5, motion=1.0)),
        "name": (Moment("sigma", 2, motion=1.5), Moment("integral", 2, motion=1.8)),
    }

    def latex_strings(self) -> list[str]:
        p = self.params
        f = Fn(p.function)
        readouts = [N_TEX, SUM_TEX[p.rule], GAP_TEX, _sigma_tex(p.n0 * 2 ** DOUBLINGS),
                    _integral_tex(p.a, p.b)]
        return [f.tex("f"), rf"[a, b] = [{_num(p.a)}, {_num(p.b)}]", QUESTION_TEX, LIMIT_TEX,
                *readouts, *map(_baseline_tex, readouts)]

    # state at the END of each phase, which is the START of the next -------------

    def _state(self, pic: _Picture, k: int) -> dict[str, Any]:
        """Name -> mobject on screen when phase k has finished (k = -1: none),
        in drawing order (z fixes the order of the picture's pieces)."""
        if k < 0:
            return {}
        s: dict[str, Any] = {"axes": pic.axes()}
        if k in (0, 3):
            s["region"] = pic.region()
        s["guides"] = pic.guides()
        if k in (1, 2):
            lvl = 0 if k == 1 else DOUBLINGS
            s["fills"], s["slivers"], s["outlines"] = pic.level(lvl)
        s["curve"] = pic.curve()
        head = (pic.legend(), pic.interval())
        if k == 0:
            s["col"] = pic.column(*head, pic.question())
        elif k in (1, 2):
            lvl = 0 if k == 1 else DOUBLINGS
            s["col"] = pic.column(*head, pic.n_line(pic.levels[lvl]), pic.sum_line(pic.sums[lvl]),
                                  pic.gap_line(pic.gaps[lvl]))
        else:
            s["col"] = pic.column(*head, pic.n_line(pic.levels[-1]), pic.sigma_line(),
                                  pic.integral_block())
        return s

    def enter(self, scene: ChalkdustScene) -> None:
        self._pic = _Picture(self.params, scene.theme)
        self._on = self._state(self._pic, self.phase_index() - 1)
        if self._on:
            scene.add(*self._on.values())

    def perform(self, scene: ChalkdustScene, tl: Timeline) -> None:
        getattr(self, f"_{self.params.phase}")(scene, tl, self._pic)

    @staticmethod
    def _settle_into(scene: ChalkdustScene, end: dict[str, Any], *loose) -> None:
        """Swap the loose pieces a phase animated for its end state's own
        mobjects, which sit exactly where those pieces ended: the next phase
        starts from _state(k), so this phase must end on it."""
        scene.remove(*loose)
        scene.add(*end.values())

    # phases ------------------------------------------------------------------------

    def _pose(self, scene, tl, pic: _Picture) -> None:
        end = self._state(pic, 0)
        legend, interval, question = end["col"]
        tl.sequence("curve", [([FadeIn(end["axes"])], 1),
                              ([Create(end["curve"]), FadeIn(legend)], 3)])
        tl.play("interval", Create(end["guides"]), FadeIn(end["region"]), FadeIn(interval))
        pulse = end["region"].animate(rate_func=there_and_back).set_fill(opacity=0.7)
        tl.sequence("question", [([Write(question)], 1), ([pulse], 1)])
        self._settle_into(scene, end, *scene.mobjects)

    def _boxes(self, scene, tl, pic: _Picture) -> None:
        on, end = self._on, self._state(pic, 1)
        question = on["col"][2]
        _, _, n_line, sum_line, gap_line = end["col"]
        # Whole rectangles first: the sum is their area. At the gap, the part
        # of each above the curve turns from green to red.
        whole = pic.boxes(pic.cells(0))
        grow = LaggedStart(*[AnimationGroup(GrowFromEdge(w, DOWN), GrowFromEdge(o, DOWN))
                             for w, o in zip(whole, end["outlines"])], lag_ratio=0.15)
        tl.play("rects", grow, FadeOut(on["region"]), FadeOut(question), FadeIn(n_line))
        tl.play("sum", FadeIn(sum_line))
        # The green leaves the overshoot first; only then does the red come
        # in, on bare background, so red is never drawn over green.
        first, then = squish_rate_func(smooth, 0.0, 0.5), squish_rate_func(smooth, 0.5, 1.0)
        tl.sequence("gap", [([ReplacementTransform(whole, end["fills"], rate_func=first),
                              FadeIn(end["slivers"], rate_func=then),
                              FadeIn(gap_line, rate_func=then)], 1),
                            ([pic.emphasis(gap_line)], 1)])
        self._settle_into(scene, end, *scene.mobjects)

    def _refine(self, scene, tl, pic: _Picture) -> None:
        on = self._on
        _, _, n_line, sum_line, gap_line = on["col"]
        t = ValueTracker(0.0)
        memo: dict[str, Any] = {}

        def now() -> _Drawn:
            v = t.get_value()
            if memo.get("s") != v:
                d = pic.drawn(v)
                memo.update(s=v, d=d, sh=pic.shapes(d.cells))
            return memo["d"]

        def shapes() -> _Shapes:
            now()
            return memo["sh"]

        pool, cut_pool = pic.levels[-1], pic.levels[-2]
        fills = always_redraw(lambda: pic.fills(shapes(), pool))
        slivers = always_redraw(lambda: pic.slivers(shapes(), pool))
        outlines = always_redraw(lambda: pic.outlines(now().cells, now().stroke, pool))
        cuts = always_redraw(lambda: pic.cuts(now().cells, now().cut, now().stroke, cut_pool))
        # The column's readouts are the very objects the entry put up; only
        # their numbers follow the picture now (live), so frame 0 is
        # unchanged, every frame's sum is the area of the rectangles drawn and
        # every frame's gap is the area of the red drawn.
        live(n_line, lambda: now().n)
        live(sum_line, lambda: pic.area_of(now().cells))
        live(gap_line, lambda: shapes().gap)
        scene.remove(on["fills"], on["slivers"], on["outlines"])
        scene.add(fills, slivers, outlines, cuts)
        tl.play("split", t.animate.set_value(1.0), rate_func=linear)
        tl.play("more", t.animate.set_value(float(DOUBLINGS)), rate_func=linear)
        for m in (fills, slivers, outlines, cuts, n_line[1], sum_line[1], gap_line[1]):
            m.clear_updaters()
        # At s = DOUBLINGS the redrawn pieces are level DOUBLINGS exactly; swap
        # in the end state's own (identical) mobjects before the last moment.
        end = self._state(pic, 2)
        scene.remove(fills, slivers, outlines, cuts, on["col"])
        scene.add(end["fills"], end["slivers"], end["outlines"], end["col"])
        # The gap is what has settled; the sum may still be visibly short of
        # the area, so it is not pointed at.
        gap_end = end["col"][4]
        tl.play("settle", pic.emphasis(gap_end), pic.pulse(end["slivers"]))
        self._settle_into(scene, end, *scene.mobjects)

    def _name(self, scene, tl, pic: _Picture) -> None:
        on, end = self._on, self._state(pic, 3)
        _, _, _, sum_line, gap_line = on["col"]
        _, _, _, sigma, block = end["col"]
        # Where the gap line sits once the sigma line (taller) replaces the sum.
        mid = pic.column(pic.legend(), pic.interval(), pic.n_line(pic.levels[-1]),
                         pic.sigma_line(), pic.gap_line(0.0))
        tl.play("sigma", ReplacementTransform(sum_line, sigma),
                gap_line.animate.move_to(mid[4], aligned_edge=LEFT))
        # The red and its number leave together, and only then does the
        # limit arrive where the gap was: never one over the other.
        tl.sequence("integral", [([FadeOut(on["fills"]), FadeOut(on["slivers"]),
                                   FadeOut(on["outlines"]), FadeOut(gap_line),
                                   FadeIn(end["region"])], 1),
                                 ([FadeIn(block)], 1.5)])
        self._settle_into(scene, end, *scene.mobjects)

    # fixtures -------------------------------------------------------------------------

    @classmethod
    def examples(cls) -> list[dict[str, Any]]:
        setup = {"function": "x^2/4 + 1", "x_range": [-1, 4], "a": 0, "b": 3,
                 "rule": "left", "n0": 4}
        out = [{"phase": ph, **setup} for ph in cls.PHASES]
        # Midpoint rule on a curve that bends both ways: red above and below.
        out.append({"phase": "boxes", "function": "sin(2*x) + 1.5", "x_range": [0, 5.5],
                    "a": 0.3, "b": 5, "rule": "mid", "n0": 4})
        # Right ends on a falling curve: every rectangle under the curve.
        out.append({"phase": "refine", "function": "8/(x + 1)", "x_range": [0, 5],
                    "a": 0.5, "b": 4.5, "rule": "right", "n0": 3})
        return out

    @classmethod
    def stress(cls) -> list[dict[str, Any]]:
        return [
            # Steep: e^x climbs 50-fold across [a, b]; the window must hold it.
            {"phase": "refine", "function": "exp(x)", "x_range": [0, 5], "a": 1, "b": 5,
             "rule": "right", "n0": 4},
            # Oscillating: many bumps, 128 rectangles at the end.
            {"phase": "refine", "function": "sin(6*x) + 1.2", "x_range": [0, 6.3],
             "a": 0.3, "b": 6, "rule": "mid", "n0": 8},
            # Large values: long readouts and tick labels.
            {"phase": "name", "function": "1000000*x^3", "x_range": [-1, 3], "a": 0.5,
             "b": 3, "rule": "left", "n0": 4},
            {"phase": "boxes", "function": "1000000*x^3", "x_range": [-1, 3], "a": 0.5,
             "b": 3, "rule": "left", "n0": 4},
            # Narrow interval in a wide window, and a long legend.
            {"phase": "refine", "function": "x^2 + 1", "x_range": [-5, 5], "a": 1, "b": 2.2,
             "rule": "right", "n0": 2},
            {"phase": "name", "function": "x^4/100 - x^3/10 + x^2/2 + 3*sqrt(x + 1) + 1",
             "x_range": [0, 6], "a": 0.5, "b": 5.5, "rule": "mid", "n0": 4},
        ]
