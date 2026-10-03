"""GraphPlot pins: the expression allowlist, discontinuity handling, tick
labels, the semantic-rung hooks, timing against the audio budget, and every
param rejection that keeps a broken graph from reaching build().

Layout of examples/stress is covered generically by test_layout.py.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from manim import MathTex, tempconfig
from pydantic import ValidationError

from chalkdust.scenes.base import ChalkdustScene
from chalkdust.scenes.components.graph_plot import (
    MIN_AXES,
    MIN_CURVE,
    MIN_HOLD,
    MIN_LABEL,
    MIN_TRACE,
    GraphPlot,
    evaluate,
    legend_tex,
    plan_for,
    tick_label,
)


def _params(**overrides) -> dict:
    base = {"functions": [{"expr": "x^2"}], "x_range": [-2, 2]}
    base.update(overrides)
    return base


def _fn(expr: str) -> dict:
    return _params(functions=[{"expr": expr}])


class TestExpressionAllowlist:
    """Expressions are data. Anything that is not arithmetic in x must fail at
    schema validation -- nothing may ever be executed."""

    @pytest.mark.parametrize("expr", [
        "__import__('os').system('echo pwned')",
        "x.__class__",
        "(lambda: 1)()",
        "open('f')",
        "x if x else 1",
        "[x][0]",
        "x == 1",
        "x // 2",
        "x % 2",
        "1j * x",
        "True",
        "'abc'",
        "1e999",           # parses to inf
        "foo(x)",
        "y + 1",
        "sin(x, 2)",
        "sin(x=1)",
        "sin(*[x])",
        "2x",              # implicit multiplication is not supported
        "(",
    ])
    def test_rejects(self, expr):
        with pytest.raises(ValidationError):
            GraphPlot(_fn(expr))

    def test_rejects_overlong(self):
        with pytest.raises(ValidationError):
            GraphPlot(_fn("x+" * 60 + "x"))

    def test_evaluates_arithmetic(self):
        x = np.array([0.0, 1.0, 2.0])
        assert np.allclose(evaluate("x^2 - 2*x + 1", x), [1, 0, 1])
        assert np.allclose(evaluate("2^x", x), [1, 2, 4])          # ^ is power
        assert np.allclose(evaluate("-x^2", x), [0, -1, -4])       # binds as -(x^2)
        assert np.allclose(evaluate("3", x), [3, 3, 3])            # constant broadcasts
        assert np.allclose(evaluate("ln(e^x)", x), x)

    def test_out_of_domain_is_nan_not_an_error(self):
        y = evaluate("sqrt(x) + log(x) + 1/x", np.array([-1.0, 0.0, 1.0]))
        assert np.isnan(y[0]) and np.isnan(y[1]) and y[2] == pytest.approx(2.0)

    def test_tower_of_powers_overflows_instead_of_hanging(self):
        # Integer arithmetic would try to build 9^(9^9); float64 gives inf,
        # which is then off-scale (nan).
        assert np.isnan(evaluate("9^9^9", np.array([0.0]))).all()

    @pytest.mark.parametrize("expr,name,tex", [
        ("x^2 - 2*x - 3", "f", "f(x) = x^{2} - 2 x - 3"),
        ("sin(x)/x", None, r"y = \frac{\sin(x)}{x}"),
        ("-3*x + 1", None, "y = -3 x + 1"),
        ("x - (x + 1)", None, r"y = x - \left(x + 1\right)"),
        ("(x + 1)^2", None, r"y = \left(x + 1\right)^{2}"),
        ("exp(-x^2/2)", None, r"y = e^{\frac{-x^{2}}{2}}"),
        ("sqrt(abs(x))", "g", r"g(x) = \sqrt{\left|x\right|}"),
        ("sin(x^2)", None, r"y = \sin(x^{2})"),
        ("sin(1/x)", None, r"y = \sin\left(\frac{1}{x}\right)"),
    ])
    def test_legend_tex(self, expr, name, tex):
        assert legend_tex(expr, name) == tex


class TestDiscontinuities:
    """Curves break at poles and holes; they never draw a bridge across one."""

    def test_reciprocal_splits_at_zero(self):
        runs = plan_for(GraphPlot(_params(functions=[{"expr": "1/x"}],
                                          x_range=[-1, 1])).params).runs[0]
        assert len(runs) == 2
        assert all(r[:, 0].max() < 0 or r[:, 0].min() > 0 for r in runs)

    def test_tan_splits_at_every_pole(self):
        p = GraphPlot(_params(functions=[{"expr": "tan(x)"}], x_range=[-4.5, 4.5],
                              y_range=[-5, 5])).params
        plan = plan_for(p)
        runs = plan.runs[0]
        poles = [math.pi / 2 + k * math.pi for k in range(-3, 3)]
        assert len(runs) == 3  # poles at +-pi/2 inside [-4.5, 4.5]
        for r in runs:
            lo, hi = r[:, 0].min(), r[:, 0].max()
            assert not any(lo < pole < hi for pole in poles)
            assert (r[:, 1] >= plan.y0).all() and (r[:, 1] <= plan.y1).all()

    def test_steep_continuous_curve_stays_connected(self):
        plan = plan_for(GraphPlot(_params(functions=[{"expr": "exp(10*x)"}],
                                          x_range=[0, 1])).params)
        assert len(plan.runs[0]) == 1

    def test_auto_window_trims_asymptote_spikes(self):
        plan = plan_for(GraphPlot(_params(functions=[{"expr": "tan(x)"}],
                                          x_range=[-20, 20])).params)
        assert plan.y1 < 20 and plan.y0 > -20

    def test_markers_are_never_cropped(self):
        p = GraphPlot(_params(functions=[{"expr": "1/x"}], x_range=[-5, 5],
                              markers=[{"x": 0.05, "label": "spike"}])).params
        plan = plan_for(p)
        assert plan.y0 <= plan.marker_ys[0] <= plan.y1


class TestTicks:
    def test_plain_labels_use_true_minus(self):
        assert tick_label(-0.5, 0.5, 1.0) == ("−0.5", False)
        assert tick_label(0.0, 0.5, 1.0) == ("0", False)
        assert tick_label(20000.0, 5000.0, 20000.0) == ("20000", False)

    def test_huge_and_tiny_axes_go_scientific_through_latex(self):
        assert tick_label(1.5e6, 5e5, 2e6) == (r"1.5 \times 10^{6}", True)
        assert tick_label(-8e5, 2e5, 1e6) == (r"-8 \times 10^{5}", True)
        assert tick_label(2e-7, 1e-7, 1e-6) == (r"2 \times 10^{-7}", True)
        assert tick_label(0.0, 1e-7, 1e-6) == ("0", True)

    def test_latex_strings_include_scientific_ticks(self):
        tex = GraphPlot(_params(functions=[{"expr": "x^3"}], x_range=[-1e6, 1e6])).latex_strings()
        assert tex[0] == "y = x^{3}"
        assert any(r"\times 10^{" in s for s in tex[1:])


class TestSemanticHooks:
    @pytest.mark.parametrize("params", GraphPlot.examples() + GraphPlot.stress())
    def test_every_latex_string_compiles(self, params):
        # Rung 2 compiles these standalone; they are generated, so they must
        # always compile. Requires a LaTeX install -- never skip this.
        for s in GraphPlot(params).latex_strings():
            MathTex(s)

    def test_min_seconds_counts_every_step(self):
        c = GraphPlot(GraphPlot.examples()[0])  # one function, two labelled markers
        assert c.min_seconds() == pytest.approx(
            MIN_AXES + MIN_CURVE + MIN_HOLD + 2 * (MIN_TRACE + MIN_LABEL))


class TestTiming:
    """Animation must consume exactly the beat's audio budget (D-002)."""

    @staticmethod
    def _elapsed(params: dict, budget: float, tmp_path) -> float:
        with tempconfig({"media_dir": str(tmp_path), "verbosity": "WARNING"}):
            scene = ChalkdustScene(GraphPlot(params), duration=budget, skip_animations=True)
            scene.setup()
            scene.construct()
            return scene.renderer.time

    @pytest.mark.parametrize("params", GraphPlot.examples())
    def test_short_budget(self, params, tmp_path):
        # Narration far shorter than the animation wants: half its minimum.
        budget = 0.5 * GraphPlot(params).min_seconds()
        assert self._elapsed(params, budget, tmp_path) == pytest.approx(budget, abs=1 / 60)

    @pytest.mark.parametrize("params", GraphPlot.examples())
    def test_long_budget(self, params, tmp_path):
        budget = 30.0  # ~3x a typical beat
        assert self._elapsed(params, budget, tmp_path) == pytest.approx(budget, abs=1 / 60)


class TestRejections:
    """What the param model refuses, so build() never sees it."""

    @pytest.mark.parametrize("overrides", [
        {"functions": []},                                           # empty
        {"functions": [{"expr": "x"}] * 4},                          # one colour per curve
        {"functions": [{"expr": ""}]},
        {"functions": [{"expr": "x", "colour": "red"}]},             # extra=forbid, nested
        {"functions": [{"expr": "x", "name": "fg"}]},                # single letter only
        {"x_range": [2, -2]},
        {"x_range": [1, 1]},
        {"x_range": [0, float("inf")]},
        {"x_range": [float("nan"), 1]},
        {"x_range": [1e6, 1e6 + 1]},                                 # too narrow to label
        {"x_range": [0, 1e13]},
        {"y_range": [3, 1]},
        {"markers": [{"x": 0, "function": 1}]},                      # no such function
        {"markers": [{"x": 5}]},                                     # outside x_range
        {"markers": [{"x": 0, "label": ""}]},
        {"markers": [{"x": 0}] * 5},
        {"functions": [{"expr": "1/x"}], "markers": [{"x": 0}]},     # on the pole
        {"y_range": [-1, 0.5], "markers": [{"x": 1.5}]},             # marker outside y_range
        {"functions": [{"expr": "sqrt(x)"}], "x_range": [-2, -1]},   # undefined everywhere
        {"y_range": [5, 6]},                                         # never enters window
    ])
    def test_rejected_at_schema(self, overrides):
        with pytest.raises(ValidationError):
            GraphPlot(_params(**overrides))

    def test_minimal_input_is_accepted(self):
        GraphPlot({"functions": [{"expr": "x"}], "x_range": [0, 1]})
