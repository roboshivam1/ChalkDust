"""GeometryConstruct: timing, schema refusals, intersection semantics, layout.

test_layout.py already runs every example and stress fixture through the
probe; these pin the behaviour specific to this component.
"""

from __future__ import annotations

import re

import pytest
from manim import tempconfig
from pydantic import ValidationError

from chalkdust.core.models import BeatSpec, Region
from chalkdust.scenes.base import ChalkdustScene
from chalkdust.scenes.components.geometry_construct import (
    HOLD_MIN,
    INTRO_MIN,
    STEP_MIN,
    GeometryConstruct,
    GeometryConstructParams,
    _resolve,
)
from chalkdust.scenes.regions import bbox
from chalkdust.validate.geometric import LayoutProbe, validate_beat

FPS = 15


def _pt(name, x, y, **kw):
    return {"kind": "point", "name": name, "at": [x, y], **kw}


def _euclid(scale=1.0, offset=0.0):
    """Euclid I.1 with its local frame scaled and shifted."""
    return {
        "shapes": [_pt("A", offset, offset), _pt("B", offset + scale, offset),
                   {"kind": "segment", "ends": ["A", "B"]}],
        "construction": [
            {"kind": "circle", "center": "A", "through": "B", "id": "cA"},
            {"kind": "circle", "center": "B", "through": "A", "id": "cB"},
            {"kind": "intersect", "of": ["cA", "cB"], "names": ["C", "D"]},
        ],
    }


def _validate(params):
    return validate_beat(BeatSpec(id="b01", narration="placeholder",
                                  component="GeometryConstruct", params=params))


# --- timing (D-002) ------------------------------------------------------------


def _elapsed(params, duration, tmp_path) -> float:
    """Run the real scene with rendering skipped; Manim still advances its
    clock by every play's run_time, which is what the mp4 length follows."""
    with tempconfig({"dry_run": True, "media_dir": str(tmp_path),
                     "frame_rate": FPS, "verbosity": "WARNING"}):
        scene = ChalkdustScene(GeometryConstruct(params), duration=duration,
                               skip_animations=True)
        scene.render()
        return scene.time


@pytest.mark.parametrize("factor", [0.5, 3.0])
@pytest.mark.parametrize("index", range(len(GeometryConstruct.examples())))
def test_consumes_budget_exactly(index, factor, tmp_path):
    params = GeometryConstruct.examples()[index]
    budget = GeometryConstruct(params).min_seconds() * factor
    assert _elapsed(params, budget, tmp_path) == pytest.approx(budget, abs=1 / FPS)


def test_consumes_budget_for_a_lone_point(tmp_path):
    # Minimal input: no given figure, one step -- no intro play, no weight.
    params = {"construction": [_pt("A", 0, 0)]}
    budget = 0.4
    assert _elapsed(params, budget, tmp_path) == pytest.approx(budget, abs=1 / FPS)


def test_min_seconds_is_sum_of_step_minimums():
    six_words = "one two three four five six"
    comp = GeometryConstruct({
        "shapes": [_pt("A", 0, 0), _pt("B", 1, 0)],
        "construction": [
            {"kind": "circle", "center": "A", "through": "B", "note": six_words},
            {"kind": "segment", "ends": ["A", "B"]},
        ],
    })
    # A captioned step lasts long enough to read its caption (3 words/s).
    expected = INTRO_MIN + max(STEP_MIN["circle"], 2.0) + STEP_MIN["segment"] + HOLD_MIN
    assert comp.min_seconds() == pytest.approx(expected)


def test_compiles_no_latex():
    for params in GeometryConstruct.examples() + GeometryConstruct.stress():
        assert GeometryConstruct(params).latex_strings() == []


# --- regions -------------------------------------------------------------------


def test_lower_third_claimed_only_with_notes():
    bare = _euclid()
    noted = {**bare, "construction": [
        {**bare["construction"][0], "note": "Compass on A"},
        *bare["construction"][1:]]}
    assert GeometryConstruct(bare).regions() == {Region.STAGE}
    assert GeometryConstruct(noted).regions() == {Region.STAGE, Region.LOWER_THIRD}


# --- schema refusals (rung 1) ----------------------------------------------------


@pytest.mark.parametrize("params,needle", [
    ({}, "at least one shape or step"),
    ({"shapes": [], "construction": []}, "at least one shape or step"),
    ({"shapes": [{"kind": "segment", "ends": ["A", "B"]}]}, "not defined yet"),
    ({"shapes": [_pt("A", 0, 0), _pt("A", 1, 0)]}, "already defined"),
    ({"shapes": [_pt("A", 0, 0), _pt("B", 0, 0),
                 {"kind": "segment", "ends": ["A", "B"]}]}, "zero length"),
    ({"shapes": [_pt("A", 0, 0), _pt("B", 1, 0), _pt("C", 2, 0),
                 {"kind": "polygon", "vertices": ["A", "B", "C"]}]}, "collinear"),
    ({"shapes": [_pt("A", 0, 0, note="given")]}, "notes belong to construction"),
    ({"shapes": [_pt("A", 0, 0)],
      "construction": [{"kind": "intersect", "of": ["x", "y"], "names": ["P"]}]},
     "no segment or circle with id 'x'"),
    # Circles of radius 1 centred 5 apart never meet.
    ({"shapes": [_pt("A", 0, 0), _pt("B", 1, 0), _pt("C", 5, 0), _pt("D", 6, 0)],
      "construction": [
          {"kind": "circle", "center": "A", "through": "B", "id": "c1"},
          {"kind": "circle", "center": "C", "through": "D", "id": "c2"},
          {"kind": "intersect", "of": ["c1", "c2"], "names": ["P"]}]},
     "meet in 0 new point(s)"),
])
def test_rejects_unresolvable_spec(params, needle):
    with pytest.raises(ValidationError, match=re.escape(needle)):
        GeometryConstructParams.model_validate(params)


@pytest.mark.parametrize("params", [
    # A 60-char point name: names are labels, so they are short by schema.
    {"shapes": [_pt("P" + "x" * 59, 0, 0)]},
    {"shapes": [_pt("A", float("inf"), 0)]},
    {"shapes": [_pt("A", 0, 0, colour="red")]},          # nested extra key
    {"shapes": [{"kind": "ellipse", "center": "A"}]},     # unknown kind
    {"construction": [_pt(f"P{i}", i, 0) for i in range(17)]},  # over the cap
])
def test_rejects_malformed_elements(params):
    with pytest.raises(ValidationError):
        GeometryConstructParams.model_validate(params)


def test_error_names_the_failing_step():
    params = _euclid()
    params["construction"][2] = {"kind": "intersect", "of": ["cA", "cX"],
                                 "names": ["C"]}
    with pytest.raises(ValidationError, match=r"construction\[2\] \(intersect\)"):
        GeometryConstructParams.model_validate(params)


# --- intersection semantics -------------------------------------------------------


def test_intersection_names_are_assigned_top_first():
    pts = _resolve(GeometryConstructParams.model_validate(_euclid()))
    assert pts["C"][1] > 0 > pts["D"][1]


def test_intersection_skips_already_named_points():
    # Angle bisector: circles at R and S meet at O and T. O is named, so the
    # single name goes to T even though O sorts first (lower-left vs upper).
    angle = GeometryConstruct.examples()[2]
    pts = _resolve(GeometryConstructParams.model_validate(angle))
    assert pts["T"] == pytest.approx([2.25, 1.299038], abs=1e-6)


# --- layout ------------------------------------------------------------------------


def _figure_box(params):
    probe = LayoutProbe(GeometryConstruct(params), duration=8.0)
    probe.construct()
    (fig,) = [m for m in probe.mobjects if getattr(m, "_chalk_label", "") == "figure"]
    return bbox(fig)


def test_local_frame_is_unitless():
    # The model picks the shape; the component picks the size and place.
    small = _figure_box(_euclid(scale=1e-3))
    large = _figure_box(_euclid(scale=1e3, offset=1e6))
    for attr in ("x", "y", "width", "height"):
        assert getattr(small, attr) == pytest.approx(getattr(large, attr), abs=1e-6)


@pytest.mark.parametrize("params,kind", [
    # Two points 1e-4 apart beside a segment of length 10: one visible dot.
    ({"shapes": [_pt("A", 0, 0), _pt("B", 1e-4, 0), _pt("C", 10, 0),
                 {"kind": "segment", "ends": ["A", "C"]}]}, "illegible"),
    # A circle that is a speck beside a long segment.
    ({"shapes": [_pt("A", 0, 0), _pt("B", 1000, 0),
                 {"kind": "segment", "ends": ["A", "B"]},
                 _pt("C", 500, 300), _pt("D", 500.01, 300)],
      "construction": [{"kind": "circle", "center": "C", "through": "D"}]},
     "illegible"),
    # Captions three lines deep cannot sit in LOWER_THIRD legibly.
    (GeometryConstruct.stress()[0], "overflow"),
])
def test_refuses_unreadable_figures(params, kind):
    report = _validate(params)
    assert report.kinds() == {kind}, f"\n{report}"
