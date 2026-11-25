"""Plotting kit for animation components (scenes/components/animation.py).

What every calculus animation needs and none should write twice: a function
the spec gives as an expression, the numbers computed from it (values,
slopes, areas), axes that fit themselves into a region, and a small set of
colour roles that mean one thing across a whole video.

Deep on purpose: a component says "plot f on [a, b] in this box" and gets back
axes whose window was chosen from f's own values, with tick labels that never
sit under a curve (they live on the frame edges), and a `to_point` that maps
data to the scene. It never says where anything goes or what colour it is
(SCENE_SPEC.md §3, §11).

Everything drawn is computed here from the parameters: a slope is the
derivative evaluated in code, an area is a quadrature. A spec cannot type a
number onto the screen through this kit (the brief's "everything drawn is
true"; FINDINGS item 4, the answer the validator could not see was wrong).

Expressions use GraphPlot's restricted language and evaluator
(graph_plot.evaluate): parsed with `ast` against an allowlist, never eval'd.
"""

from __future__ import annotations

import math as _math
from dataclasses import dataclass
from functools import cached_property
from typing import Annotated, Callable

import numpy as np
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    Circle,
    DashedLine,
    DecimalNumber,
    Dot,
    Line,
    Mobject,
    Polygon,
    VGroup,
    VMobject,
)
from pydantic import AfterValidator

from chalkdust.scenes.components.graph_plot import (
    _auto_window,
    _check,
    _parse,
    _runs,
    axis_ticks,
    evaluate,
    legend_tex,
)
from chalkdust.scenes.regions import (
    MIN_FONT_SIZE,
    LayoutError,
    Rect,
    scale_with_tags,
    smallest_font_size,
    tag_font_size,
)
from chalkdust.scenes.theme import Theme, caption_text, math

# --- colour roles ---------------------------------------------------------------
#
# One meaning per colour across every animation, so a colour means the same
# thing for a whole video (the brief: "colour means one thing throughout").
# Roles map onto palette entries; a component asks for a role, never a colour.

ROLES = {
    "function": "accent",      # f itself: its curve, its name, its values
    "rate": "accent_alt",      # slopes: secants, tangents, f', rise/run
    "area": "success",         # accumulated area, integrals, Riemann sums
    "alert": "danger",         # what breaks or misses: a hole, a jump, an error
    "text": "fg",              # ordinary labels and maths
    "guide": "muted",          # axes, ticks, dashed guides
}


def role(theme: Theme, name: str) -> str:
    """The palette colour of a role."""
    return getattr(theme.palette, ROLES[name])


# --- expressions ------------------------------------------------------------------


def check_expr(expr: str) -> str:
    """Pydantic validator: a parseable expression in x from the allowlist."""
    expr = expr.strip()
    _check(_parse(expr), expr)
    return expr


Expr = Annotated[str, AfterValidator(check_expr)]


@dataclass(frozen=True)
class Fn:
    """A function of x from a spec expression, with the calculus done in code."""

    expr: str

    def np(self, xs: np.ndarray) -> np.ndarray:
        with np.errstate(all="ignore"):
            return evaluate(self.expr, np.asarray(xs, dtype=float))

    def at(self, x: float) -> float:
        return float(self.np(np.array([x]))[0])

    def defined(self, x: float) -> bool:
        return _math.isfinite(self.at(x))

    def d(self, x: float) -> float:
        """f'(x) by a five-point central difference: error O(h^4), about 1e-9
        relative at h = 1e-3 * scale for the smooth functions the language
        allows. Not finite where f is not differentiable or not defined."""
        h = 1e-3 * max(1.0, abs(x))
        ys = self.np(np.array([x - 2 * h, x - h, x + h, x + 2 * h]))
        return float((ys[0] - 8 * ys[1] + 8 * ys[2] - ys[3]) / (12 * h))

    def integral(self, a: float, b: float) -> float:
        """∫_a^b f by adaptive quadrature (scipy, a Manim dependency)."""
        from scipy.integrate import quad

        value, _err = quad(lambda t: self.at(t), a, b, limit=200)
        return float(value)

    def tex(self, name: str | None = "f") -> str:
        """LaTeX printed from the parsed expression, e.g. 'f(x) = x^{2}'."""
        return legend_tex(self.expr, name)

    def continuous_on(self, a: float, b: float, n: int = 801) -> bool:
        """True if f is finite on a dense grid over [a, b] with no jump. Used
        to refuse parameters that would draw a broken picture."""
        xs = np.linspace(a, b, n)
        ys = self.np(xs)
        if not np.isfinite(ys).all():
            return False
        span = float(ys.max() - ys.min()) or 1.0
        return len(_runs(self.expr, xs, ys, float(ys.min()) - 1, float(ys.max()) + 1)) == 1 \
            or span == 0


# Draw order: points (dots, holes) above lines and areas, which stay at 0.
POINT_Z = 2


def y_window(fns: list[Fn], x0: float, x1: float, extra: list[float] = (),
             include_zero: bool = True, n: int = 600) -> tuple[float, float]:
    """A y-window that shows the shape of every function on [x0, x1] and every
    value in `extra` (points the animation will visit), padded a little.

    Raises ValueError (a param error, rung 1) if a function is nowhere defined.
    """
    xs = np.linspace(x0, x1, n)
    vals = []
    for f in fns:
        ys = f.np(xs)
        ys = ys[np.isfinite(ys)]
        if ys.size == 0:
            raise ValueError(f"{f.expr!r} is undefined everywhere on [{x0}, {x1}]")
        vals.append(ys)
    allv = np.concatenate(vals + [np.asarray(extra, dtype=float)]) if extra else np.concatenate(vals)
    y0, y1 = _auto_window(allv)
    if extra:
        lo, hi = min(extra), max(extra)
        pad = 0.08 * max(y1 - y0, 1e-9)
        y0, y1 = min(y0, lo - pad), max(y1, hi + pad)
    if include_zero:
        y0, y1 = min(y0, 0.0), max(y1, 0.0)
    if y1 - y0 < 1e-9:
        y0, y1 = y0 - 1, y1 + 1
    return float(y0), float(y1)


# --- the plot -------------------------------------------------------------------------

TICK_LEN = 0.12
LABEL_GAP = 0.1
AXIS_STROKE = 2.0
CURVE_STROKE = 5.0
N_CURVE_SAMPLES = 480
MIN_PLOT_SIDE = 2.0
X_TICKS = 6
Y_TICKS = 5


class Plot:
    """Axes for the window [x0, x1] x [y0, y1], fitted into `box`.

    Tick labels sit outside the plot rect (left and bottom edges), so curves,
    which are clipped to the rect, never run through them. Axis lines cross at
    the origin when it is in view, otherwise run along the nearest edge.
    """

    def __init__(self, theme: Theme, box: Rect, x0: float, x1: float,
                 y0: float, y1: float, x_ticks: int = X_TICKS, y_ticks: int = Y_TICKS) -> None:
        if not (x1 > x0 and y1 > y0):
            raise ValueError(f"empty plot window x [{x0}, {x1}], y [{y0}, {y1}]")
        self.theme = theme
        self.x0, self.x1, self.y0, self.y1 = float(x0), float(x1), float(y0), float(y1)
        muted = role(theme, "guide")

        def tick_mob(s: str, tex: bool) -> Mobject:
            if tex:
                return math(s, theme, size=theme.type.caption, color=muted,
                            what="axis tick label")
            return caption_text(s, theme, color=muted)

        _, xt = axis_ticks(self.x0, self.x1, x_ticks)
        _, yt = axis_ticks(self.y0, self.y1, y_ticks)
        xlabels = [(v, tick_mob(s, tex)) for _, v, s, tex in xt]
        ylabels = [(v, tick_mob(s, tex)) for _, v, s, tex in yt]
        max_yw = max((m.width for _, m in ylabels), default=0.0)
        max_xh = max((m.height for _, m in xlabels), default=0.0)
        max_xw = max((m.width for _, m in xlabels), default=0.0)
        max_yh = max((m.height for _, m in ylabels), default=0.0)

        left = box.left + max_yw + LABEL_GAP + TICK_LEN
        right = box.right - max_xw / 2
        bottom = box.bottom + max_xh + LABEL_GAP + TICK_LEN
        top = box.top - max_yh / 2
        if right - left < MIN_PLOT_SIDE or top - bottom < MIN_PLOT_SIDE:
            raise LayoutError(f"plot: only {right - left:.2f} x {top - bottom:.2f} "
                              "left after tick labels", kind="overflow")
        self.rect = Rect((left + right) / 2, (bottom + top) / 2, right - left, top - bottom)
        self.sx = self.rect.width / (self.x1 - self.x0)
        self.sy = self.rect.height / (self.y1 - self.y0)

        stroke = dict(color=muted, stroke_width=AXIS_STROKE)
        ax_y = min(max(0.0, self.y0), self.y1)
        ax_x = min(max(0.0, self.x0), self.x1)
        self.x_axis = Line(self.p(self.x0, ax_y), self.p(self.x1, ax_y), **stroke)
        self.y_axis = Line(self.p(ax_x, self.y0), self.p(ax_x, self.y1), **stroke)
        parts: list[Mobject] = [self.x_axis, self.y_axis]
        # Ticks on the axis lines; their numbers on the frame edges.
        for v, m in xlabels:
            at = self.p(v, ax_y)
            parts.append(Line(at + DOWN * TICK_LEN / 2, at - DOWN * TICK_LEN / 2, **stroke))
            m.next_to(self.p(v, self.y0), DOWN, buff=LABEL_GAP + TICK_LEN / 2)
            parts.append(m)
        for v, m in ylabels:
            at = self.p(ax_x, v)
            parts.append(Line(at + LEFT * TICK_LEN / 2, at - LEFT * TICK_LEN / 2, **stroke))
            m.next_to(self.p(self.x0, v), LEFT, buff=LABEL_GAP + TICK_LEN / 2)
            parts.append(m)
        self.axes = VGroup(*parts)

    # coordinates ------------------------------------------------------------------

    def p(self, x: float, y: float) -> np.ndarray:
        """Data -> scene point."""
        return np.array([self.rect.left + (x - self.x0) * self.sx,
                         self.rect.bottom + (y - self.y0) * self.sy, 0.0])

    def points(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        xs, ys = np.asarray(xs, float), np.asarray(ys, float)
        return np.column_stack([self.rect.left + (xs - self.x0) * self.sx,
                                self.rect.bottom + (ys - self.y0) * self.sy,
                                np.zeros_like(xs)])

    def contains(self, x: float, y: float) -> bool:
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1

    # drawing ------------------------------------------------------------------------

    def curve(self, f: Fn, color: str, a: float | None = None, b: float | None = None,
              stroke: float = CURVE_STROKE, n: int = N_CURVE_SAMPLES) -> VGroup:
        """f on [a, b] (default: the window), clipped to the window and split
        where it is undefined or jumps: one VMobject per continuous run."""
        a = self.x0 if a is None else max(a, self.x0)
        b = self.x1 if b is None else min(b, self.x1)
        xs = np.linspace(a, b, n)
        runs = _runs(f.expr, xs, f.np(xs), self.y0, self.y1)
        out = VGroup()
        for r in runs:
            m = VMobject(color=color, stroke_width=stroke)
            m.set_points_smoothly(self.points(r[:, 0], r[:, 1])) if len(r) > 3 else \
                m.set_points_as_corners(self.points(r[:, 0], r[:, 1]))
            out.add(m)
        return out

    def area(self, f: Fn, a: float, b: float, color: str, opacity: float = 0.35,
             n: int = 160) -> Polygon:
        """The region between f and y = 0 over [a, b], clipped to the window
        (signed: below the axis where f is negative)."""
        a, b = max(a, self.x0), min(b, self.x1)
        xs = np.linspace(a, b, n)
        ys = np.clip(f.np(xs), self.y0, self.y1)
        base = min(max(0.0, self.y0), self.y1)
        pts = [self.p(a, base), *self.points(xs, ys), self.p(b, base)]
        return Polygon(*pts, color=color, fill_color=color, fill_opacity=opacity,
                       stroke_width=0)

    def dot(self, x: float, y: float, color: str, radius: float = 0.07) -> Dot:
        """A point on the picture. Points draw above every line (POINT_Z),
        whatever order things were added in, so a phase's last frame and the
        next phase's first frame stack the same way."""
        d = Dot(self.p(x, y), radius=radius, color=color)
        d.set_z_index(POINT_Z)
        return d

    def hole(self, x: float, y: float, color: str, radius: float = 0.08) -> Circle:
        """An open circle: f is not defined (or not equal to the limit) here."""
        c = Circle(radius=radius, color=color, stroke_width=3)
        c.set_fill(self.theme.palette.bg, opacity=1.0)
        c.set_z_index(POINT_Z)
        return c.move_to(self.p(x, y))

    def vguide(self, x: float, y: float, color: str | None = None) -> DashedLine:
        """Dashed drop line from (x, y) to the x-axis."""
        base = min(max(0.0, self.y0), self.y1)
        return DashedLine(self.p(x, base), self.p(x, y), dash_length=0.08,
                          color=color or role(self.theme, "guide"), stroke_width=2)

    def hguide(self, x: float, y: float, color: str | None = None) -> DashedLine:
        """Dashed line from (x, y) across to the y-axis."""
        ax_x = min(max(0.0, self.x0), self.x1)
        return DashedLine(self.p(ax_x, y), self.p(x, y), dash_length=0.08,
                          color=color or role(self.theme, "guide"), stroke_width=2)

    def line_through(self, x: float, y: float, slope: float, color: str,
                     half_width: float | None = None, stroke: float = 4.0) -> Line:
        """A straight line through (x, y) with `slope` (data units), clipped to
        the window, so a steep tangent never pokes out of the plot."""
        x0, x1 = (self.x0, self.x1) if half_width is None else (x - half_width, x + half_width)
        x0, x1 = max(x0, self.x0), min(x1, self.x1)
        pts = _clip_segment(x0, y + slope * (x0 - x), x1, y + slope * (x1 - x),
                            self.x0, self.x1, self.y0, self.y1)
        if pts is None:  # nowhere in the window: a zero-length line at the point
            pts = ((x, y), (x, y))
        (ax, ay), (bx, by) = pts
        return Line(self.p(ax, ay), self.p(bx, by), color=color, stroke_width=stroke)


def _clip_segment(x0, y0, x1, y1, xmin, xmax, ymin, ymax):
    """Liang-Barsky clip of a segment to a rectangle; None if outside."""
    dx, dy = x1 - x0, y1 - y0
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, x0 - xmin), (dx, xmax - x0), (-dy, y0 - ymin), (dy, ymax - y0)):
        if p == 0:
            if q < 0:
                return None
            continue
        r = q / p
        if p < 0:
            t0 = max(t0, r)
        else:
            t1 = min(t1, r)
        if t0 > t1:
            return None
    return (x0 + t0 * dx, y0 + t0 * dy), (x0 + t1 * dx, y0 + t1 * dy)


# --- readouts ------------------------------------------------------------------------


def number(theme: Theme, value: float, color: str, decimals: int = 2,
           size: float | None = None) -> DecimalNumber:
    """A number drawn from a computed value, tagged for the legibility check.
    Use .set_value() (or an updater) to keep it equal to what the picture
    shows; never write a number into the spec."""
    size = size or theme.type.body
    # x1.2 as theme.math does for MathTex, so digits match the maths they sit
    # beside; the tag records the theme size, as theme.math's does.
    d = DecimalNumber(value, num_decimal_places=decimals, color=color, font_size=size * 1.2,
                      include_sign=False)
    tag_font_size(d, size)
    return d


def fmt(value: float, decimals: int = 2) -> str:
    """A computed value as LaTeX text, trailing zeros kept to `decimals` (so a
    readout does not change width as it converges), minus sign as in maths."""
    s = f"{value:.{decimals}f}"
    if s.startswith("-") and float(s) == 0:
        s = s[1:]
    return s


def readout(theme: Theme, prefix_tex: str, value: float, color: str,
            decimals: int = 2) -> VGroup:
    """`prefix = value`: LaTeX prefix, then a DecimalNumber computed by code.

    For a value that changes on screen, keep the VGroup and give its number
    an updater (`live(readout, fn)`): DecimalNumber reuses its digit glyphs,
    where a fresh MathTex per frame would run LaTeX on every frame. The live
    and the static readout of one value are built by this same function, so
    a phase that starts from a static one matches the live one pixel for pixel.
    """
    size = theme.type.body
    head = math(prefix_tex, theme, size=size, color=color, what="readout")
    num = number(theme, value, color, decimals=decimals, size=size)
    num.next_to(head, RIGHT, buff=0.15)
    return VGroup(head, num)


def live(r: VGroup, value: Callable[[], float]) -> VGroup:
    """Make a readout's number follow `value()` every frame (left edge fixed)."""
    r[1].add_updater(lambda m: m.set_value(value()))
    return r


def column_lines(theme: Theme, lines: list[Mobject], box: Rect, gap: float = 0.35) -> VGroup:
    """Stack lines top-down at the top-left of `box` (the maths column).

    A line wider than the box is scaled down on its own, to the legibility
    floor at most (LayoutError "overflow" below it). Scaling each line by
    itself keeps a line identical in every phase it appears in, whatever its
    neighbours are -- the continuity of a picture across beats needs that.
    """
    for line in lines:
        if line.width > box.width:
            scale_with_tags(line, box.width / line.width)
            small = smallest_font_size(line)
            if small is not None and small < MIN_FONT_SIZE:
                raise LayoutError(
                    f"column line too wide: would render at font size {small:.1f} "
                    f"(floor {MIN_FONT_SIZE:.0f}); shorten it", kind="overflow")
    g = VGroup(*lines).arrange(DOWN, aligned_edge=LEFT, buff=gap)
    if g.height > box.height:
        raise LayoutError(f"maths column is {g.height:.2f} tall, room for {box.height:.2f}; "
                          "show fewer lines at once", kind="overflow")
    g.move_to(np.array([box.left + g.width / 2, box.top - g.height / 2, 0.0]))
    return g


Number = Callable[[], float]
