"""SecantToTangent: the derivative at a point, as a secant that turns into the
tangent while its slope converges.

The picture: a curve f and a point P = (a, f(a)) on it ("how steep is the
curve right here?"). A second point Q = (a + h, f(a + h)) and the secant
through P and Q, with its rise over run. Then Q slides into P: h shrinks
geometrically, the secant turns, and its slope (a live number computed from f)
settles on f'(a). Only then the name and the formula: the limit of the
difference quotient.

Phases (each one beat; animation.py):
  pose    curve -> point -> question
  secant  second -> line -> rise_run
  shrink  slide -> tangent
  name    limit -> value

Setup: `function` (an expression in x), `x_range`, `a`, `h`. Every number on
screen is computed from these (plotting.Fn): the secant slope at each h, and
f'(a) by a central difference. Refused, as param errors, before anything is
drawn: a point where f is not differentiable (left and right slopes disagree,
|x| at 0), a secant that would cross a break in f, Q outside the window, or Q
so close to P that the secant cannot be seen.
"""

from __future__ import annotations

import math as _math
from typing import Annotated, Any, Literal

import numpy as np
from manim import (
    LEFT,
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
    Write,
    always_redraw,
    linear,
)
from pydantic import Field, model_validator

from chalkdust.scenes.base import ChalkdustScene, Moment, Timeline
from chalkdust.scenes.components.animation import Animation, AnimationParams
from chalkdust.scenes.components.base import label, register
from chalkdust.scenes.plotting import Expr, Fn, Plot, column_lines, live, readout, role, y_window
from chalkdust.scenes.regions import LayoutError
from chalkdust.scenes.theme import Theme, math

FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]

# Where the slide stops: small enough that the secant's slope shows f'(a) to
# the readout's two decimals, large enough to stay a visible line.
SLOPE_TOLERANCE = 0.002
# Q must start at least this share of the x-range away from P, or the secant
# is too short to read.
MIN_H_SHARE = 0.08
MAX_SPAN = 1e4
SLOPE_TEX = r"\text{slope} ="
LIMIT_TEX = r"f'(a) = \lim_{h \to 0} \frac{f(a+h) - f(a)}{h}"


class SecantTangentParams(AnimationParams):
    phase: Literal["pose", "secant", "shrink", "name"]
    function: Expr = Field(description="f as an expression in x, e.g. 'x^2/4 + 1'")
    x_range: tuple[FiniteFloat, FiniteFloat] = Field(description="[min, max] of x shown")
    a: FiniteFloat = Field(description="x of the point P where the slope is asked for")
    h: FiniteFloat = Field(description="starting gap from P to the second point Q (may be negative)")

    @model_validator(mode="after")
    def _true_picture(self) -> SecantTangentParams:
        x0, x1 = self.x_range
        if not x1 > x0:
            raise ValueError(f"x_range {list(self.x_range)} must be [min, max] with min < max")
        if x1 - x0 > MAX_SPAN:
            raise ValueError(f"x_range is {x1 - x0:g} wide; keep it under {MAX_SPAN:g}")
        if not x0 < self.a < x1:
            raise ValueError(f"a = {self.a:g} must lie strictly inside x_range {list(self.x_range)}")
        q = self.a + self.h
        if not x0 <= q <= x1:
            raise ValueError(f"a + h = {q:g} lies outside x_range {list(self.x_range)}; "
                             "choose a smaller h")
        if abs(self.h) < MIN_H_SHARE * (x1 - x0):
            raise ValueError(f"|h| = {abs(self.h):g} is under {MIN_H_SHARE:.0%} of the x-range; "
                             "the secant would be too short to see")
        f = Fn(self.function)
        lo, hi = sorted((self.a, q))
        xs = np.linspace(lo, hi, 401)
        ys = f.np(xs)
        if not np.isfinite(ys).all():
            raise ValueError(f"{self.function!r} is undefined somewhere between a and a + h")
        # A jump between P and Q: a secant across it would show a slope of a
        # function that is not one piece there.
        if len(_pieces(f, lo, hi)) != 1:
            raise ValueError(f"{self.function!r} breaks between a = {self.a:g} and a + h = {q:g}")
        # Differentiable at a: one-sided difference quotients must agree.
        eps = 1e-4 * max(1.0, x1 - x0)
        fa = f.at(self.a)
        right = (f.at(self.a + eps) - fa) / eps
        left = (fa - f.at(self.a - eps)) / eps
        if not (_math.isfinite(right) and _math.isfinite(left)):
            raise ValueError(f"{self.function!r} has no slope at a = {self.a:g}")
        if abs(right - left) > 1e-2 * max(1.0, abs(right), abs(left)):
            raise ValueError(f"{self.function!r} is not differentiable at a = {self.a:g} "
                             f"(slope {left:.3g} from the left, {right:.3g} from the right); "
                             "pick a point where the curve is smooth")
        return self


def _pieces(f: Fn, lo: float, hi: float) -> list:
    from chalkdust.scenes.components.graph_plot import _runs

    xs = np.linspace(lo, hi, 401)
    ys = f.np(xs)
    pad = max(1.0, float(np.ptp(ys)))
    return _runs(f.expr, xs, ys, float(ys.min()) - pad, float(ys.max()) + pad)


class _Picture:
    """Every mobject of the animation, built from the setup alone, so each
    phase's start is exactly the previous phase's end (animation.py)."""

    def __init__(self, p: SecantTangentParams, theme: Theme) -> None:
        self.p, self.theme = p, theme
        self.f = f = Fn(p.function)
        self.a, self.h0 = p.a, p.h
        self.fa = f.at(p.a)
        self.slope = f.d(p.a)
        self.h_end = _end_gap(f, p.a, p.h, self.slope)
        x0, x1 = p.x_range
        extra = [self.fa, f.at(p.a + p.h)]
        y0, y1 = y_window([f], x0, x1, extra=extra)
        self.plot = Plot(theme, SecantToTangent.plot_box(), x0, x1, y0, y1)
        self.col = SecantToTangent.column_box()
        self.c_fn, self.c_rate = role(theme, "function"), role(theme, "rate")
        self.c_text = role(theme, "text")

    # the picture -------------------------------------------------------------------

    def axes(self):
        return label(self.plot.axes, "axes")

    def curve(self):
        return label(self.plot.curve(self.f, self.c_fn), "curve f")

    def point(self, x: float, name: str):
        return label(self.plot.dot(x, self.f.at(x), self.c_text, radius=0.08), name)

    def guide(self, x: float):
        return self.plot.vguide(x, self.f.at(x))

    def secant(self, h: float):
        q = self.a + h
        m = (self.f.at(q) - self.fa) / h
        return label(self.plot.line_through(self.a, self.fa, m, self.c_rate), "secant")

    def tangent(self):
        return label(self.plot.line_through(self.a, self.fa, self.slope, self.c_rate,
                                            stroke=5.0), "tangent")

    def legs(self, h: float):
        """Run (horizontal, from P) and rise (vertical, up to Q), dashed."""
        q, fq = self.a + h, self.f.at(self.a + h)
        run = DashedLine(self.plot.p(self.a, self.fa), self.plot.p(q, self.fa),
                         dash_length=0.08, color=self.c_rate, stroke_width=3)
        rise = DashedLine(self.plot.p(q, self.fa), self.plot.p(q, fq),
                          dash_length=0.08, color=self.c_rate, stroke_width=3)
        return label(VGroup(run, rise), "rise and run")

    # the maths column ---------------------------------------------------------------

    def legend(self):
        return label(math(self.f.tex("f"), self.theme, size=self.theme.type.body,
                          color=self.c_fn, what="legend"), "legend")

    def a_line(self):
        return label(math(f"a = {_num(self.a)}", self.theme, size=self.theme.type.body,
                          color=self.c_text, what="a"), "a")

    def secant_slope(self, h: float) -> float:
        return (self.f.at(self.a + h) - self.fa) / h

    def h_line(self, h: float):
        return label(readout(self.theme, "h =", h, self.c_text), "h")

    def slope_line(self, h: float):
        return label(readout(self.theme, SLOPE_TEX, self.secant_slope(h), self.c_rate),
                     "slope")

    def deriv_line(self):
        return label(readout(self.theme, f"f'({_num(self.a)}) =", self.slope, self.c_rate),
                     "f'(a)")

    def limit_line(self):
        return label(math(LIMIT_TEX, self.theme,
                          size=self.theme.type.body, color=self.c_text, what="limit"), "limit")

    def column(self, *lines):
        """The maths column, top-left aligned; positions depend only on which
        lines are there, so the same lines always land in the same place."""
        return column_lines(self.theme, list(lines), self.col)

    def question(self):
        q = math(SLOPE_TEX + r" \,?", self.theme, size=self.theme.type.body,
                 color=self.c_rate, what="question")
        return label(q, "question")


def _num(v: float) -> str:
    """A setup value as written: integers without decimals."""
    return str(int(v)) if float(v).is_integer() else f"{v:g}"


def _end_gap(f: Fn, a: float, h: float, slope: float) -> float:
    """The h the slide stops at: halve h until the secant's slope is within
    SLOPE_TOLERANCE of f'(a), so the readout lands on the derivative."""
    fa = f.at(a)
    end = h
    for _ in range(40):
        m = (f.at(a + end) - fa) / end
        if abs(m - slope) < SLOPE_TOLERANCE:
            break
        end /= 2
    return end


@register
class SecantToTangent(Animation):
    name = "SecantToTangent"
    Params = SecantTangentParams

    PHASES = {
        "pose": (Moment("curve", 2, motion=2.0), Moment("point", 1.5, motion=1.0),
                 Moment("question", 1.5, motion=1.0)),
        "secant": (Moment("second", 1.5, motion=1.0), Moment("line", 1.5, motion=1.2),
                   Moment("rise_run", 2, motion=1.2)),
        "shrink": (Moment("slide", 4), Moment("tangent", 2, motion=1.2)),
        "name": (Moment("limit", 2, motion=1.8), Moment("value", 1.5, motion=1.0)),
    }

    def latex_strings(self) -> list[str]:
        f = Fn(self.params.function)
        return [f.tex("f"), r"f'(a) = \lim_{h \to 0} \frac{f(a+h) - f(a)}{h}"]

    # state at the END of each phase, which is the START of the next -------------

    def _state(self, pic: _Picture, k: int) -> dict[str, Any]:
        """Name -> mobject on screen when phase k has finished (k = -1: none),
        in drawing order: axes, curve, guides, lines, dots, then the column."""
        if k < 0:
            return {}
        a, h = pic.a, pic.h0
        s: dict[str, Any] = {"axes": pic.axes(), "curve": pic.curve(), "gP": pic.guide(a)}
        if k == 0:
            s["P"] = pic.point(a, "P")
            s["col"] = pic.column(pic.legend(), pic.a_line(), pic.question())
        elif k == 1:
            s.update(gQ=pic.guide(a + h), sec=pic.secant(h), legs=pic.legs(h),
                     P=pic.point(a, "P"), Q=pic.point(a + h, "Q"))
            s["col"] = pic.column(pic.legend(), pic.a_line(), pic.h_line(h), pic.slope_line(h))
        elif k == 2:
            s.update(tan=pic.tangent(), P=pic.point(a, "P"))
            s["col"] = pic.column(pic.legend(), pic.a_line(), pic.deriv_line())
        else:
            s.update(tan=pic.tangent(), P=pic.point(a, "P"))
            s["col"] = pic.column(pic.legend(), pic.limit_line(), pic.deriv_line())
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
        legend, a_line, question = end["col"]
        tl.sequence("curve", [([FadeIn(end["axes"])], 1), ([Create(end["curve"])], 3)])
        tl.play("point", GrowFromCenter(end["P"]), Create(end["gP"]), FadeIn(legend),
                FadeIn(a_line))
        tl.sequence("question", [([Write(question)], 1),
                                 ([Indicate(end["P"], color=pic.c_rate)], 1)])
        self._settle_into(scene, end, *scene.mobjects)

    def _secant(self, scene, tl, pic: _Picture) -> None:
        on, end = self._on, self._state(pic, 1)
        question = on["col"][2]
        _, _, h_line, slope_line = end["col"]
        scene.bring_to_front(on["P"])
        tl.play("second", GrowFromCenter(end["Q"]), Create(end["gQ"]), FadeOut(question),
                FadeIn(h_line))
        tl.play("line", Create(end["sec"]))
        scene.bring_to_front(on["P"], end["Q"])
        tl.play("rise_run", Create(end["legs"]), FadeIn(slope_line))
        self._settle_into(scene, end, *scene.mobjects)

    def _shrink(self, scene, tl, pic: _Picture) -> None:
        a, h0, h1 = pic.a, pic.h0, pic.h_end
        on = self._on
        _, _, h_line, slope_line = on["col"]
        t = ValueTracker(0.0)

        def h_now() -> float:
            return h0 * (h1 / h0) ** t.get_value()

        live_parts = {
            "gQ": always_redraw(lambda: pic.guide(a + h_now())),
            "sec": always_redraw(lambda: pic.secant(h_now())),
            "legs": always_redraw(lambda: pic.legs(h_now())),
            "Q": always_redraw(lambda: pic.point(a + h_now(), "Q")),
        }
        # The column's readouts are the very objects the entry put up; only
        # their numbers follow h now (plotting.live), so frame 0 is unchanged.
        live(h_line, h_now)
        live(slope_line, lambda: pic.secant_slope(h_now()))
        scene.remove(on["gQ"], on["sec"], on["legs"], on["Q"])
        scene.add(*live_parts.values())
        scene.bring_to_front(on["P"], live_parts["Q"])
        tl.play("slide", t.animate.set_value(1.0), rate_func=linear)
        for m in [*live_parts.values(), h_line[1], slope_line[1]]:
            m.clear_updaters()
        end = self._state(pic, 2)
        deriv = end["col"][2]
        tl.play("tangent", ReplacementTransform(live_parts["sec"], end["tan"]),
                FadeOut(live_parts["legs"]), FadeOut(live_parts["Q"]), FadeOut(live_parts["gQ"]),
                FadeOut(h_line), ReplacementTransform(slope_line, deriv))
        self._settle_into(scene, end, *scene.mobjects)

    def _name(self, scene, tl, pic: _Picture) -> None:
        on, end = self._on, self._state(pic, 3)
        _, a_line, deriv = on["col"]
        _, limit, deriv3 = end["col"]
        tl.play("limit", ReplacementTransform(a_line, limit), deriv.animate.move_to(deriv3))
        tl.play("value", Indicate(deriv, color=pic.c_rate),
                Indicate(on["tan"], color=pic.c_rate))
        self._settle_into(scene, end, *scene.mobjects)

    # fixtures -------------------------------------------------------------------------

    @classmethod
    def examples(cls) -> list[dict[str, Any]]:
        setup = {"function": "x^2/4 + 1", "x_range": [-1, 4], "a": 1, "h": 2}
        out = [{"phase": ph, **setup} for ph in cls.PHASES]
        out.append({"phase": "shrink", "function": "sin(x)", "x_range": [0, 6.3],
                    "a": 2, "h": -1.5})
        return out

    @classmethod
    def stress(cls) -> list[dict[str, Any]]:
        return [
            # Steep and huge: the window and the clipped lines must cope.
            {"phase": "shrink", "function": "exp(x)", "x_range": [-2, 9], "a": 8, "h": -3},
            {"phase": "secant", "function": "1000000*x^3", "x_range": [-3, 3], "a": 2, "h": -1.5},
            {"phase": "name", "function": "x^5 - 40*x^3 + 3", "x_range": [-6, 6], "a": -5, "h": 4},
            # A near-vertical tangent next to a sharp-but-smooth bend.
            {"phase": "shrink", "function": "sqrt(x)", "x_range": [0, 9], "a": 0.2, "h": 6},
        ]
