"""RiemannSum: the definite integral as the limit of rectangle sums.

The picture: a curve f and the region under it between a and b ("how big is
this area?"). Rectangles stand on [a, b], each as tall as f at its sample
point (left end, right end or midpoint), and their total area is a live
number; the slivers where the rectangles miss the curve are the gap. Then
every rectangle splits in two, in place, again and again: the slivers shrink,
the sum settles and the gap goes to zero. Only then the notation: the sum as
Σ f(x_i) Δx, and the integral it closes in on, with its value.

Phases (each one beat; animation.py):
  pose    curve -> interval -> question
  boxes   rects -> sum -> gap
  refine  split -> more -> settle
  name    sigma -> integral

Setup: `function` (an expression in x), `x_range`, `a`, `b`, `rule` (left,
right or mid) and `n0` (rectangles at first; refine doubles them DOUBLINGS
times). Every number on screen is computed from these (plotting.Fn): each
rectangle's height is f at its sample point, each sum is the sum of the
heights drawn times their width, and the integral is a quadrature
(Fn.integral). A spec cannot type any of them.

Area, not signed area. f < 0 anywhere on [a, b] is refused. The picture asks
"how big is this area?" and the last phase answers it with the integral; where
f < 0 the integral subtracts, so that answer would be false unless the
words changed with the setup. Signed area is its own idea and would be a
second idea in one beat (the brief: one idea per beat). The refusal names the
fix: an interval where f >= 0, or f shifted up.

Also refused, as param errors before anything is drawn: f undefined or broken
anywhere on [a, b], an empty or reversed interval or one outside the window,
an interval too narrow to read, rectangles that would end thinner than
MIN_RECT_PX at 1080p, and an area too flat to see against the rest of f.
"""

from __future__ import annotations

import math as _math
from typing import Annotated, Any, Literal, NamedTuple

import numpy as np
from manim import (
    DOWN,
    LEFT,
    UP,
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
    there_and_back,
)
from pydantic import Field, model_validator

from chalkdust.scenes.base import ChalkdustScene, Moment, Timeline
from chalkdust.scenes.components.animation import Animation, AnimationParams
from chalkdust.scenes.components.base import label, register
from chalkdust.scenes.plotting import Expr, Fn, Plot, column_lines, readout, role, y_window
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

# Share of one doubling spent drawing the cut down each rectangle's middle;
# the rest is the two halves finding their own heights.
CUT = 0.3
RECT_OPACITY = 0.45
SLIVER_OPACITY = 0.6
REGION_OPACITY = 0.35
# Rectangle outline: a share of the rectangle's width (stroke_width 1 is 0.01
# scene units in Manim's Cairo camera), within these bounds, so the outline
# never swallows a thin rectangle's fill.
STROKE_PER_UNIT = 12.0
STROKE_MIN, STROKE_MAX = 0.6, 2.5
SLIVER_SAMPLES = 160
# Emphasis pulse on a column line: at most this scale, grown from its left
# edge and never wider than the column (_Picture.emphasis).
EMPHASIS_SCALE = 1.12

# Draw order, fixed by z so a redrawn piece and the end state stack the same
# way whatever order they were added in (continuity, animation.py).
Z_REGION, Z_GUIDE, Z_RECT, Z_CUT, Z_SLIVER, Z_CURVE = 0.1, 0.2, 0.3, 0.35, 0.4, 1.0

SUM_TEX = {"left": r"\text{left sum} =", "right": r"\text{right sum} =",
           "mid": r"\text{midpoint sum} ="}
GAP_TEX = r"\text{gap} ="
N_TEX = "n ="
QUESTION_TEX = r"\text{area} = \,?"
SIGMA_TEX = r"\sum_{i=1}^{n} f(x_i)\,\Delta x ="


def _num(v: float) -> str:
    """A setup value as written: integers without decimals."""
    return str(int(v)) if float(v).is_integer() else f"{v:g}"


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


def riemann_sum(f: Fn, a: float, b: float, n: int, rule: str) -> float:
    """Σ f(x_i) Δx with n equal rectangles. The same arithmetic as _area() on
    the drawn cells, so the static and the live readout agree to the bit."""
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
    return min(6, 2 - e)


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
        f = Fn(self.function)
        xs, ys = _values_on(f, a, b, self.n0, self.rule)
        bad = ~np.isfinite(ys)
        if bad.any():
            raise ValueError(f"{self.function!r} is undefined at x = {xs[bad][0]:g}, inside "
                             "[a, b]; pick an interval where f is defined")
        if not f.continuous_on(a, b):
            raise ValueError(f"{self.function!r} jumps between a = {a:g} and b = {b:g}; "
                             "pick an interval where f is continuous")
        lo, hi = float(ys.min()), float(ys.max())
        if lo < -NEG_TOL * max(1.0, abs(hi)):
            i = int(np.argmin(ys))
            raise ValueError(f"{self.function!r} is negative on [a, b] (f({xs[i]:g}) = "
                             f"{ys[i]:.3g}); this animation shows the area between f and the "
                             "x-axis, so f must be >= 0 there: choose [a, b] where f >= 0, "
                             "or add a constant to f")
        y0, y1 = y_window([f], x0, x1, extra=[lo, hi])
        if hi < MIN_HEIGHT_SHARE * (y1 - y0):
            raise ValueError(f"the area under {self.function!r} on [a, b] would be under "
                             f"{MIN_HEIGHT_SHARE:.0%} of the plot's height (f reaches {hi:.3g} "
                             f"there, the window is {y1 - y0:.3g} tall); narrow x_range "
                             "towards [a, b]")
        return self


def _baseline_tex(tex: str) -> str:
    """The label with a digit after it: where that digit sits is the line's
    baseline (_on_baseline)."""
    return tex + r"\,1"


def _on_baseline(r: VGroup, tex: str, theme: Theme, color: str) -> VGroup:
    """Sit a readout's number on its label's baseline, and keep it there
    (_snap, after every set_value).

    plotting.readout centres the number on the label's bounding box, which
    drops it below the line when the label has descenders and no ascenders
    ("gap =") and lifts it when the label is all x-height ("n ="); and
    DecimalNumber.set_value keeps the box centre, which moves the digits when
    a thousands comma (it hangs below the line) comes or goes. Typesetting the
    label with a digit after it gives the baseline relative to the label's
    first glyph, as a share of the label's height so it survives scaling.
    Shared-kit candidate: the same fix belongs in readout() and live()."""
    head = r[0]
    ref = math(_baseline_tex(tex), theme, size=theme.type.body, color=color, what="readout")
    ref_glyphs = ref.family_members_with_points()
    r._chalk_base = (ref_glyphs[-1].get_bottom()[1]  # type: ignore[attr-defined]
                     - ref_glyphs[0].get_bottom()[1]) / head.height
    return _snap(r)


def _snap(r: VGroup) -> VGroup:
    """Put the bottom of the number's first digit (a digit sits on the
    baseline; the box bottom may be a comma's tail) on the label's baseline."""
    head, num = r
    base = head.family_members_with_points()[0].get_bottom()[1] + r._chalk_base * head.height
    num.shift((base - num[0].get_bottom()[1]) * UP)
    return r


def _live(r: VGroup, value) -> VGroup:
    """plotting.live, with the number re-seated on the baseline each frame."""
    def follow(m) -> None:
        m.set_value(value())
        _snap(r)

    r[1].add_updater(follow)
    return r


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
        self.f = f = Fn(p.function)
        self.a, self.b, self.rule = p.a, p.b, p.rule
        self.levels = [p.n0 * 2 ** k for k in range(DOUBLINGS + 1)]
        x0, x1 = p.x_range
        # Every height any rectangle reaches is f somewhere on [a, b]; the
        # window must hold all of them (y_window's percentile trim must not
        # clip a steep f under the rectangles).
        _, ys = _values_on(f, p.a, p.b, p.n0, p.rule)
        self.hi = float(ys.max())
        y0, y1 = y_window([f], x0, x1, extra=[float(ys.min()), self.hi])
        self.plot = Plot(theme, RiemannSum.plot_box(), x0, x1, y0, y1)
        self.col = RiemannSum.column_box()
        self.c_fn, self.c_area = role(theme, "function"), role(theme, "area")
        self.c_alert, self.c_text = role(theme, "alert"), role(theme, "text")
        self.true = f.integral(p.a, p.b)
        self.sums = [riemann_sum(f, p.a, p.b, n, p.rule) for n in self.levels]
        # The largest value any readout can show (a sum is at most max f
        # times the width): readouts are laid out at it, so a live number
        # never outgrows the column.
        self.widest = max(self.hi * (p.b - p.a), abs(self.true), *map(abs, self.sums))
        self.decimals = _decimals(self.widest)

    # cells: (x_left, x_right, height) per rectangle -----------------------------

    def cells(self, k: int) -> np.ndarray:
        """The rectangles of level k (n = n0 * 2^k), each as tall as f at its
        sample point."""
        n = self.levels[k]
        e = _edges(self.a, self.b, n)
        return np.column_stack([e[:-1], e[1:], self.f.np(_samples(self.a, self.b, n, self.rule))])

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
        return label(_z(self.plot.curve(self.f, self.c_fn), Z_CURVE), "curve f")

    def region(self):
        return label(_z(self.plot.area(self.f, self.a, self.b, self.c_area,
                                       opacity=REGION_OPACITY), Z_REGION), "area")

    def guides(self):
        g = VGroup(self.plot.vguide(self.a, self.f.at(self.a)),
                   self.plot.vguide(self.b, self.f.at(self.b)))
        return label(_z(g, Z_GUIDE), "a and b")

    def rects(self, cells: np.ndarray, stroke: float, pool: int = 0) -> VGroup:
        p = self.plot.p
        g = VGroup(*[Polygon(p(xl, 0.0), p(xl, h), p(xr, h), p(xr, 0.0),
                             stroke_color=self.c_area, stroke_width=stroke,
                             fill_color=self.c_area, fill_opacity=RECT_OPACITY)
                     for xl, xr, h in cells])
        return label(_z(_pad(g, pool), Z_RECT), "rectangles")

    def slivers(self, cells: np.ndarray, pool: int = 0) -> VGroup:
        """Between each rectangle's top and the curve: what the rectangle
        misses (under the curve) or adds (above it). The gap, in alert."""
        g = VGroup()
        m = max(4, _math.ceil(SLIVER_SAMPLES / len(cells)))
        for xl, xr, h in cells:
            xs = np.linspace(xl, xr, m)
            ys = self.f.np(xs)
            if float(np.max(np.abs(ys - h))) * self.plot.sy < 1e-4:
                g.add(VMobject())  # the top is the curve here: nothing to show
                continue
            pts = [*self.plot.points(xs, ys), self.plot.p(xr, h), self.plot.p(xl, h)]
            g.add(Polygon(*pts, stroke_width=0, fill_color=self.c_alert,
                          fill_opacity=SLIVER_OPACITY))
        return label(_z(_pad(g, pool), Z_SLIVER), "gap slivers")

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

    def level(self, k: int) -> tuple[VGroup, VGroup]:
        cells = self.cells(k)
        return self.rects(cells, self.stroke(self.levels[k])), self.slivers(cells)

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
        _on_baseline(r, tex, self.theme, color)
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
        return self._readout(SIGMA_TEX, self.sums[-1], self.widest, self.c_area,
                             self.decimals, "sigma")

    def integral_tex(self) -> str:
        return rf"\int_{{{_num(self.a)}}}^{{{_num(self.b)}}} f(x)\,dx ="

    def integral_line(self):
        return self._readout(self.integral_tex(), self.true, self.widest, self.c_area,
                             self.decimals, "integral")

    def emphasis(self, line):
        """A there-and-back pulse on a column line. Not Indicate: it scales
        by 1.2 about the centre, which pushes a wide line across the column
        gap into the plot. This grows from the line's left edge, only as far
        as the column allows, by a few hundredths vertically (the lines are
        0.35 apart), and ends exactly where it began."""
        k = max(1.0, min(EMPHASIS_SCALE, self.col.width / line.width))
        return line.animate(rate_func=there_and_back).scale(k, about_edge=LEFT)

    def column(self, *lines):
        """The maths column, top-left aligned; positions depend only on which
        lines are there, so the same lines always land in the same place."""
        g = column_lines(self.theme, list(lines), self.col)
        for line in lines:
            v = getattr(line, "_chalk_value", None)
            if v is not None:
                line[1].set_value(v)
                _snap(line)
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
        labels = [N_TEX, SUM_TEX[p.rule], GAP_TEX, SIGMA_TEX,
                  rf"\int_{{{_num(p.a)}}}^{{{_num(p.b)}}} f(x)\,dx ="]
        return [f.tex("f"), rf"[a, b] = [{_num(p.a)}, {_num(p.b)}]", QUESTION_TEX,
                *labels, *map(_baseline_tex, labels)]

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
            s["rects"], s["slivers"] = pic.level(lvl)
        s["curve"] = pic.curve()
        head = (pic.legend(), pic.interval())
        if k == 0:
            s["col"] = pic.column(*head, pic.question())
        elif k in (1, 2):
            lvl = 0 if k == 1 else DOUBLINGS
            total = pic.sums[lvl]
            s["col"] = pic.column(*head, pic.n_line(pic.levels[lvl]), pic.sum_line(total),
                                  pic.gap_line(abs(total - pic.true)))
        else:
            s["col"] = pic.column(*head, pic.n_line(pic.levels[-1]), pic.sigma_line(),
                                  pic.integral_line())
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
        grow = LaggedStart(*[GrowFromEdge(r, DOWN) for r in end["rects"]], lag_ratio=0.15)
        tl.play("rects", grow, FadeOut(on["region"]), FadeOut(question), FadeIn(n_line))
        tl.play("sum", FadeIn(sum_line))
        tl.sequence("gap", [([FadeIn(end["slivers"]), FadeIn(gap_line)], 1),
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
                memo.update(s=v, d=pic.drawn(v))
            return memo["d"]

        pool, cut_pool = pic.levels[-1], pic.levels[-2]
        rects = always_redraw(lambda: pic.rects(now().cells, now().stroke, pool))
        cuts = always_redraw(lambda: pic.cuts(now().cells, now().cut, now().stroke, cut_pool))
        slivers = always_redraw(lambda: pic.slivers(now().cells, pool))
        # The column's readouts are the very objects the entry put up; only
        # their numbers follow the picture now (_live), so frame 0 is
        # unchanged and every frame's sum is the area of the rectangles drawn.
        _live(n_line, lambda: now().n)
        _live(sum_line, lambda: pic.area_of(now().cells))
        _live(gap_line, lambda: abs(pic.area_of(now().cells) - pic.true))
        scene.remove(on["rects"], on["slivers"])
        scene.add(rects, cuts, slivers)
        tl.play("split", t.animate.set_value(1.0), rate_func=linear)
        tl.play("more", t.animate.set_value(float(DOUBLINGS)), rate_func=linear)
        for m in (rects, cuts, slivers, n_line[1], sum_line[1], gap_line[1]):
            m.clear_updaters()
        # At s = DOUBLINGS the redrawn pieces are level DOUBLINGS exactly; swap
        # in the end state's own (identical) mobjects before the last moment.
        end = self._state(pic, 2)
        scene.remove(rects, cuts, slivers, on["col"])
        scene.add(end["rects"], end["slivers"], end["col"])
        _, _, _, sum_end, gap_end = end["col"]
        tl.play("settle", pic.emphasis(sum_end), pic.emphasis(gap_end))
        self._settle_into(scene, end, *scene.mobjects)

    def _name(self, scene, tl, pic: _Picture) -> None:
        on, end = self._on, self._state(pic, 3)
        _, _, _, sum_line, gap_line = on["col"]
        _, _, _, sigma, integral = end["col"]
        # Where the gap line sits once the sigma line (taller) replaces the sum.
        mid = pic.column(pic.legend(), pic.interval(), pic.n_line(pic.levels[-1]),
                         pic.sigma_line(), pic.gap_line(0.0))
        tl.play("sigma", ReplacementTransform(sum_line, sigma),
                gap_line.animate.move_to(mid[4], aligned_edge=LEFT))
        tl.play("integral", FadeOut(on["rects"]), FadeOut(on["slivers"]), FadeIn(end["region"]),
                FadeOut(gap_line), FadeIn(integral))
        self._settle_into(scene, end, *scene.mobjects)

    # fixtures -------------------------------------------------------------------------

    @classmethod
    def examples(cls) -> list[dict[str, Any]]:
        setup = {"function": "x^2/4 + 1", "x_range": [-1, 4], "a": 0, "b": 3,
                 "rule": "left", "n0": 4}
        out = [{"phase": ph, **setup} for ph in cls.PHASES]
        # Midpoint rule on a curve that bends both ways: slivers above and below.
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
