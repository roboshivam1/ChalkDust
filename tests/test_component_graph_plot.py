"""GraphPlot pins: the expression allowlist, discontinuity handling, tick
labels, the semantic-rung hooks, frame-exact timing against the audio budget,
carry-in, and every param rejection that keeps a broken graph from reaching
build().

Layout of examples/stress is covered generically by test_layout.py. These
tests compile LaTeX, so `latex` and `dvisvgm` must be on PATH.

There is no invalid-LaTeX case in stress(): GraphPlot takes no LaTeX. Its
maths input is an expression, refused at schema (TestExpressionAllowlist);
every LaTeX string it compiles is printed from the parsed tree or a tick
value (TestSemanticHooks compiles all of them).
"""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import asdict

import numpy as np
import pytest
from manim import DOWN, LEFT, ORIGIN, RIGHT, UP, MathTex, Restore, Square, tempconfig
from pydantic import ValidationError

from chalkdust import continuity
from chalkdust.continuity import build_artifact, carried, resolve_carry_in
from chalkdust.core.models import BeatSpec, Quality, Region, VideoSpec
from chalkdust.render.worker import TIERS
from chalkdust.scenes.base import ChalkdustScene
from chalkdust.scenes.components import Component, ComponentParams
from chalkdust.scenes.components import base as components_base
from chalkdust.scenes.components.graph_plot import (
    INK_CLEARANCE,
    MIN_AXES,
    MIN_AXIS_LABELS,
    MIN_CURVE,
    MIN_HOLD,
    MIN_LABEL,
    MIN_SPAN,
    MIN_TRACE,
    GraphPlot,
    _ink,
    _inked,
    _pick_labels,
    evaluate,
    legend_tex,
    plan_for,
    tick_label,
)
from chalkdust.scenes.regions import DEFAULT_PADDING, Rect, bbox, region_rect
from chalkdust.scenes.theme import DEFAULT
from chalkdust.validate.geometric import LayoutProbe, validate_beat

DRAFT = TIERS[Quality.DRAFT]


def _params(**overrides) -> dict:
    base = {"functions": [{"expr": "x^2"}], "x_range": [-2, 2]}
    base.update(overrides)
    return base


def _fn(expr: str) -> dict:
    return _params(functions=[{"expr": expr}])


EXAMPLES = GraphPlot.examples()
EXAMPLE_IDS = [f"ex{i}" for i in range(len(EXAMPLES))]


def _probe(params: dict, tmp_path) -> LayoutProbe:
    """The settled scene, built without rendering."""
    with tempconfig({"media_dir": str(tmp_path), "verbosity": "WARNING"}):
        scene = LayoutProbe(GraphPlot(params), duration=8.0)
        scene.construct()
    return scene


def _axes(scene):
    return next(m for m in scene.mobjects if getattr(m, "_chalk_label", "") == "axes")


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

    def test_zero_tick_survives_a_subnormal_step(self):
        # A relative zero test (|v| < step * 1e-6) underflows to "< 0" for a
        # subnormal step and sent 0 on to log10(0): a raw ValueError.
        assert tick_label(0.0, 5e-321, 1e-320) == ("0", True)

    def test_ink_catches_a_steep_segment_between_samples(self):
        # Near an asymptote two neighbouring samples can straddle a label with
        # neither endpoint inside it; the densified ink still hits.
        segment = np.array([[0.0, -2.0, 0.0], [0.0, 2.0, 0.0]])
        box = Rect(0.0, 0.0, 0.3, 0.2)
        assert not _inked(segment, box)
        assert _inked(_ink([segment]), box)

    # Stress 0-1 are clean overflow refusals (pinned by test_layout.py); every
    # case that builds is checked.
    @pytest.mark.parametrize("params", GraphPlot.examples() + GraphPlot.stress()[2:])
    def test_no_curve_runs_through_a_tick_label(self, params, tmp_path):
        # A curve stroke over muted caption text hides glyphs: "-2" with its
        # minus covered reads as "2". Ink is taken from the built curves, not
        # from the plan the component used.
        with tempconfig({"media_dir": str(tmp_path), "verbosity": "WARNING"}):
            scene = LayoutProbe(GraphPlot(params), duration=8.0)
            scene.construct()
        mobs = {getattr(m, "_chalk_label", ""): m for m in scene.mobjects}
        paths = [np.vstack([sp[::4], sp[-1:]])
                 for name, curve in mobs.items() if name.startswith("curve")
                 for sp in curve.get_subpaths()]
        ink = _ink(paths)
        labels = [m for m in mobs["axes"].submobjects if getattr(m, "_chalk_font_size", None)]
        assert labels
        assert not [m for m in labels if _inked(ink, bbox(m), INK_CLEARANCE)]

    @pytest.mark.parametrize("params", GraphPlot.examples() + GraphPlot.stress()[2:])
    def test_every_axis_keeps_at_least_two_labels(self, params, tmp_path):
        # One number on an axis gives no scale. sin + cos on [-6.5, 6.5] once
        # ended with a single y label after curve-covered labels were dropped.
        axes = _axes(_probe(params, tmp_path))
        names = [getattr(m, "_chalk_label", "") for m in axes.submobjects]
        assert sum(n.startswith("xtick") for n in names) >= MIN_AXIS_LABELS
        assert sum(n.startswith("ytick") for n in names) >= MIN_AXIS_LABELS

    def test_blocked_label_tries_the_other_side_before_dropping(self):
        # Only the right of the tick is clear: the label goes there, not away.
        text = Square(side_length=0.2)
        kept = _pick_labels([(1, ORIGIN, text)], 1, (LEFT, RIGHT), lambda box: box.x > 0)
        assert [t for *_, t in kept] == [text] and bbox(text).x > 0

    def test_too_few_labels_falls_back_to_ticks_between(self):
        # Stride 2 picks ticks 0 and 2; tick 2 is blocked, so tick 1 (well
        # clear of tick 0) is labelled rather than leaving a single label.
        ticks = [(k, np.array([2.0 * k, 0.0, 0.0]), Square(side_length=0.2)) for k in range(3)]
        kept = _pick_labels(ticks, 2, (DOWN, UP), lambda box: box.x < 3)
        assert [k for k, *_ in kept] == [0, 1]

    def test_crowded_interior_axis_is_labelled_at_the_plot_edge(self, tmp_path):
        # tan(x) on [-20, 20]: branches run up past the y-axis beside every
        # label spot, so its labels move to the plot's left edge -- a margin
        # no curve enters -- rather than leaving the axis unlabelled.
        axes = _axes(_probe({"functions": [{"expr": "tan(x)"}], "x_range": [-20, 20]},
                            tmp_path))
        x_axis, y_axis = axes.submobjects[:2]
        plot_left = x_axis.get_left()[0]
        assert y_axis.get_center()[0] > plot_left + 1  # the axis is interior
        ylabels = [m for m in axes.submobjects
                   if getattr(m, "_chalk_label", "").startswith("ytick")]
        assert len(ylabels) >= MIN_AXIS_LABELS
        assert all(bbox(t).right < plot_left for t in ylabels)

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
    """The beat lasts exactly ceil(audio * fps) frames (D-002): every play and
    wait takes its run time from scene.budget(), which hands out whole frames
    summing to beat_frames."""

    @staticmethod
    def _frames(params: dict, duration: float, tmp_path) -> tuple[int, int]:
        """(frames Manim's own frame loop drew, beat_frames), at draft fps.
        Nothing is encoded; renderer.time advances 1/fps per frame drawn."""
        with tempconfig({**asdict(DRAFT), "media_dir": str(tmp_path),
                         "write_to_movie": False, "save_last_frame": False,
                         "disable_caching": True, "progress_bar": "none",
                         "verbosity": "WARNING"}):
            scene = ChalkdustScene(GraphPlot(params), duration=duration)
            scene.setup()
            scene.construct()
            return round(scene.renderer.time * DRAFT.frame_rate), scene.beat_frames

    @pytest.mark.parametrize("factor", [0.5, 3.0], ids=["short", "long"])
    @pytest.mark.parametrize("params", EXAMPLES, ids=EXAMPLE_IDS)
    def test_frames_equal_beat_frames(self, params, factor, tmp_path):
        # Narration far shorter (half the minimum) and far longer (3x) than
        # the animation wants: either way, exactly the audio's frames.
        duration = factor * GraphPlot(params).min_seconds()
        drawn, beat = self._frames(params, duration, tmp_path)
        assert beat == math.ceil(round(duration * DRAFT.frame_rate, 6))
        assert drawn == beat

    def test_draft_render_is_exactly_the_beat(self, tmp_path):
        # A real encoded clip, frames counted by ffprobe: 3.879 s of audio is
        # ceil(3.879 * 15) = 59 frames.
        with tempconfig({**asdict(DRAFT), "media_dir": str(tmp_path),
                         "disable_caching": True, "progress_bar": "none",
                         "verbosity": "WARNING", "output_file": "graph_frames"}):
            scene = ChalkdustScene(GraphPlot(EXAMPLES[0]), duration=3.879)
            scene.render()
            movie = scene.renderer.file_writer.movie_file_path
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
             "-show_entries", "stream=nb_read_frames", "-of", "json", str(movie)],
            capture_output=True, text=True, check=True).stdout
        assert int(json.loads(out)["streams"][0]["nb_read_frames"]) == 59


class _TargetParams(ComponentParams):
    target_id: str


class _Target(Component):
    """Acts on a carried artifact the way ZoomHighlight will: build() raises
    CarryInError unless the artifact was carried in. Registered per test."""

    name = "_GraphTarget"
    Params = _TargetParams

    def regions(self) -> set[Region]:
        return {Region.STAGE}

    def build(self, scene) -> None:
        scene.play(Restore(carried(scene, self.params.target_id)),
                   run_time=scene.budget(1)[0])


class TestCarryIn:
    """A finished graph can be carried into a later beat (SCENE_SPEC.md §6),
    e.g. as a ZoomHighlight or Callout target."""

    @staticmethod
    def _video(params: dict) -> VideoSpec:
        return VideoSpec(video_id="v", beats=(
            BeatSpec(id="b01", narration="placeholder narration", component="GraphPlot",
                     params=params, registers="plot"),
            BeatSpec(id="b02", narration="placeholder narration", component=_Target.name,
                     params={"target_id": "plot"}, carry_in=["plot"]),
        ))

    def test_graph_plot_registers_a_builder(self):
        assert "GraphPlot" in continuity._BUILDERS
        recipes = resolve_carry_in(self._video(EXAMPLES[0]))["b02"]
        assert [(r.name, r.producer) for r in recipes] == [("plot", "GraphPlot")]

    def test_rebuild_is_deterministic(self):
        recipe = resolve_carry_in(self._video(EXAMPLES[0]))["b02"][0]
        a, b = build_artifact(recipe, DEFAULT), build_artifact(recipe, DEFAULT)
        pa = [m.points for m in a.family_members_with_points()]
        pb = [m.points for m in b.family_members_with_points()]
        assert len(pa) == len(pb) and all(np.array_equal(x, y) for x, y in zip(pa, pb))

    @pytest.mark.parametrize("params", EXAMPLES, ids=EXAMPLE_IDS)
    def test_artifact_is_the_settled_frame(self, params, tmp_path):
        # Built by the same layout() as the render: the carried graph is the
        # picture the producing beat ended on, mobject for mobject.
        scene = _probe(params, tmp_path)
        with tempconfig({"media_dir": str(tmp_path), "verbosity": "WARNING"}):
            art = GraphPlot(params).layout(scene.theme).settled()
        drawn = [m for m in scene.mobjects if getattr(m, "_chalk_label", None)]
        got = sorted((m._chalk_label, *np.round(m.get_center(), 6)) for m in art.submobjects)
        want = sorted((m._chalk_label, *np.round(m.get_center(), 6)) for m in drawn)
        assert got == want

    def test_artifact_fits_stage_unscaled(self):
        # CarryIn fits the artifact to STAGE; at full size already, nothing is
        # shrunk, so caption-size tick labels stay above the legibility floor.
        art = GraphPlot(EXAMPLES[1]).layout(DEFAULT).settled()
        assert region_rect(Region.STAGE).inset(DEFAULT_PADDING).contains(bbox(art))

    @pytest.mark.parametrize("params", EXAMPLES, ids=EXAMPLE_IDS)
    def test_carried_graph_validates_clean(self, params, monkeypatch, tmp_path):
        monkeypatch.setitem(components_base._REGISTRY, _Target.name, _Target)
        video = self._video(params)
        report = validate_beat(video.beats[1], recipes=resolve_carry_in(video)["b02"],
                               media_dir=tmp_path)
        assert report.ok, f"\n{report}"


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
        {"y_range": [0, 1e-200]},                                    # below MIN_SPAN
    ])
    def test_rejected_at_schema(self, overrides):
        with pytest.raises(ValidationError):
            GraphPlot(_params(**overrides))

    def test_subnormal_x_range_is_a_param_error(self):
        # Once a raw ValueError from log10(0) in latex_strings() and build():
        # a span this small has no finite data -> scene scale.
        with pytest.raises(ValidationError, match="spans less than"):
            GraphPlot({"functions": [{"expr": "x"}], "x_range": [0, 1e-320]})

    def test_subnormal_auto_y_span_is_drawn_flat(self, tmp_path):
        # 1e-310*x on [-1, 1] varies by 2e-310: scaling that to the plot gave
        # inf, and every curve point came out inf/nan. Now flat at this scale.
        params = {"functions": [{"expr": "1e-310*x"}], "x_range": [-1, 1]}
        plan = plan_for(GraphPlot(params).params)
        assert plan.y1 - plan.y0 >= MIN_SPAN
        curve = next(m for m in _probe(params, tmp_path).mobjects
                     if getattr(m, "_chalk_label", "") == "curve[0]")
        assert len(curve.points) and np.isfinite(curve.points).all()

    def test_minimal_input_is_accepted(self):
        GraphPlot({"functions": [{"expr": "x"}], "x_range": [0, 1]})
