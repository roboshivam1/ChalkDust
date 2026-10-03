"""VectorField: timing, expression safety, refusals, and the magnitude clamp.

Layout validity of examples() and stress() is covered by tests/test_layout.py,
which walks the registry; this file pins what is specific to VectorField.
"""

from __future__ import annotations

import builtins

import pytest
from manim import tempconfig
from pydantic import ValidationError

from chalkdust.core.models import BeatSpec, Region
from chalkdust.scenes.base import ChalkdustScene
from chalkdust.scenes.components.vector_field import (
    ARROW_FILL,
    MIN_ARROWS_SECONDS,
    MIN_HOLD_SECONDS,
    MIN_PLANE_SECONDS,
    VectorField,
    grid_shape,
)
from chalkdust.scenes.regions import DEFAULT_PADDING, bbox, region_rect
from chalkdust.validate.geometric import LayoutProbe, validate_beat

ROTATION = {"field_fn": {"x": "-y", "y": "x"}}


def _probe(params: dict, duration: float = 8.0) -> LayoutProbe:
    probe = LayoutProbe(VectorField(params), duration=duration)
    probe.construct()
    return probe


def _labelled(scene, name: str):
    return next(m for m in scene.mobjects if getattr(m, "_chalk_label", "") == name)


def _field(**overrides) -> dict:
    return {**ROTATION, **overrides}


# --- timing (D-002) ---------------------------------------------------------


@pytest.mark.parametrize("budget", [1.0, 24.0], ids=["far-short", "far-long"])
@pytest.mark.parametrize("example", range(len(VectorField.examples())))
def test_consumes_budget_exactly(tmp_path, example, budget):
    # Natural length is ~8s; 1s is below min_seconds() and 24s is 3x natural.
    # skip_animations advances renderer.time by each play/wait's run time
    # without encoding frames.
    with tempconfig({"media_dir": str(tmp_path), "frame_rate": 15}):
        scene = ChalkdustScene(VectorField(VectorField.examples()[example]),
                               duration=budget, skip_animations=True)
        scene.setup()
        scene.construct()
        assert scene.renderer.time == pytest.approx(budget, abs=1 / 15)


def test_min_seconds_gives_every_step_its_floor():
    vf = VectorField(ROTATION)
    assert vf.min_seconds() == MIN_PLANE_SECONDS + MIN_ARROWS_SECONDS + MIN_HOLD_SECONDS


def test_compiles_no_latex():
    assert VectorField(ROTATION).latex_strings() == []


# --- expression safety ------------------------------------------------------


@pytest.mark.parametrize("expr", [
    "__import__('os').system('echo pwned')",
    "x.real",
    "(lambda: 1)()",
    "x^2",                       # BitXor, the classic model mistake
    "foo(x)",
    "sin(x, y)",
    "sin(x=1)",
    "'a'",
    "True",
    "[x]",
    "x if y else 1",
    "a_sixty_character_identifier_that_no_parser_should_accept_ok",
    "x +",
    "",
    "x" * 121,                   # over MAX_EXPR_CHARS
])
def test_rejects_expression(expr):
    with pytest.raises(ValidationError):
        VectorField({"field_fn": {"x": expr, "y": "x"}})


def test_caret_error_points_at_the_fix():
    with pytest.raises(ValidationError, match=r"use \*\* for powers"):
        VectorField({"field_fn": {"x": "x^2", "y": "y"}})


def test_never_calls_eval(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("VectorField must not eval field expressions")

    monkeypatch.setattr(builtins, "eval", refuse)
    monkeypatch.setattr(builtins, "exec", refuse)
    _probe({"field_fn": {"x": "sin(y)*exp(-x**2/4)", "y": "cos(x)+pi*e"}})


# --- schema refusals --------------------------------------------------------


@pytest.mark.parametrize("overrides", [
    {"x_range": [2, 1]},             # decreasing
    {"x_range": [1, 1]},             # empty
    {"x_range": [-50, 50]},          # aspect 25: a strip, not a plane
    {"y_range": [-20, 20]},          # aspect 0.25
    {"x_range": [-1e9, 1e9]},        # outside the bound
    {"sample_density": 3},
    {"sample_density": 21},
], ids=lambda o: str(o))
def test_rejects_bad_domain_or_density(overrides):
    with pytest.raises(ValidationError):
        VectorField(_field(**overrides))


@pytest.mark.parametrize("fn", [
    {"x": "0", "y": "0"},                      # zero everywhere
    {"x": "sqrt(-1-x**2)", "y": "log(-1-y**2)"},  # undefined everywhere
])
def test_rejects_field_with_nothing_to_draw(fn):
    with pytest.raises(ValidationError, match="nothing to draw"):
        VectorField({"field_fn": fn})


def test_minimal_field_renders():
    probe = _probe({"field_fn": {"x": "1", "y": "0"}, "sample_density": 4})
    assert len(_labelled(probe, "field arrows").submobjects) == 8  # 4 x 2 grid


# --- layout -----------------------------------------------------------------


def test_too_dense_for_the_plane_refuses_with_overflow():
    spec = BeatSpec(id="b01", narration="n", component="VectorField",
                    params=_field(sample_density=20, x_range=[-2, 2], y_range=[-2, 2]))
    assert validate_beat(spec).kinds() == {"overflow"}


# stress()[0] is the too-dense case, which refuses instead of building.
@pytest.mark.parametrize("params", [*VectorField.examples(), *VectorField.stress()[1:]])
def test_arrows_stay_inside_the_plane(params):
    probe = _probe(params)
    plane = bbox(_labelled(probe, "field plane"))
    for arrow in _labelled(probe, "field arrows").submobjects:
        assert plane.contains(bbox(arrow), tol=1e-6), arrow._chalk_label


def test_singular_field_is_clamped_and_skips_undefined_points():
    # 9x3 grid: the origin is a sample point, where the field is inf/nan.
    params = {"field_fn": {"x": "x/(x**2+y**2)**1.5", "y": "y/(x**2+y**2)**1.5"},
              "sample_density": 9, "x_range": [-3, 3], "y_range": [-1, 1]}
    probe = _probe(params)
    arrows = _labelled(probe, "field arrows").submobjects
    nx, ny = grid_shape(VectorField(params).params)
    assert len(arrows) < nx * ny
    assert not any(a._chalk_label == "arrow(0.00, 0.00)" for a in arrows)

    inner = region_rect(Region.STAGE).inset(DEFAULT_PADDING)
    unit = min(inner.width / 6, inner.height / 2)
    full = ARROW_FILL * unit * min(6 / nx, 2 / ny)
    lengths = [a.get_length() for a in arrows]
    assert max(lengths) == pytest.approx(full, rel=1e-6)
