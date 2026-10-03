"""FreeBodyDiagram: the behaviours the registry-wide layout tests cannot pin.

tests/test_layout.py already proves examples() validate clean and stress()
fits or refuses cleanly. This file pins what is specific to this component:
timing against a real renderer clock, the schema's refusals, the typed LaTeX
failure, the magnitude clamp, side-by-side clustering, and the promise that a
label never touches another force's arrow.

These tests compile LaTeX, so `latex` and `dvisvgm` must be on PATH.
"""

from __future__ import annotations

import pytest
from manim import Arrow, MathTex, tempconfig
from pydantic import ValidationError

from chalkdust.core.models import BeatSpec
from chalkdust.scenes.base import ChalkdustScene
from chalkdust.scenes.components import make_component
from chalkdust.scenes.components.free_body_diagram import (
    ARROW_MIN_FRACTION,
    BODY_S,
    FORCE_S,
    HOLD_S,
    FreeBodyDiagram,
    _segment_hits_rect,
)
from chalkdust.scenes.regions import LayoutError, bbox
from chalkdust.validate.geometric import LayoutProbe, validate_beat

NAME = "FreeBodyDiagram"
EXAMPLES = FreeBodyDiagram.examples()
DRAFT_FPS = 15


def _spec(params: dict) -> BeatSpec:
    return BeatSpec(id="b01", narration="placeholder narration",
                    component=NAME, params=params)


def _probe(params: dict, duration: float = 8.0) -> LayoutProbe:
    probe = LayoutProbe(make_component(NAME, params), duration=duration)
    probe.construct()
    return probe


def _parts(scene) -> tuple[list[Arrow], list[MathTex]]:
    """The force arrows and their labels, in force order."""
    arrows = [m for m in scene.mobjects if isinstance(m, Arrow)]
    labels = [m for m in scene.mobjects if isinstance(m, MathTex)]
    return arrows, labels


# --- timing (D-002) -----------------------------------------------------------


@pytest.mark.parametrize("factor", [0.5, 3.0], ids=["short", "long"])
@pytest.mark.parametrize("params", EXAMPLES, ids=[f"ex{i}" for i in range(len(EXAMPLES))])
def test_timing_consumes_budget_exactly(params, factor, tmp_path):
    # Narration far shorter and far longer than the animation wants: either
    # way the beat must last exactly as long as its audio. Measured on the
    # renderer's own clock at draft frame rate, because Manim rounds every
    # segment to whole frames and that rounding is what drifts.
    budget = make_component(NAME, params).min_seconds() * factor
    settings = {"dry_run": True, "pixel_width": 160, "pixel_height": 90,
                "frame_rate": DRAFT_FPS, "media_dir": str(tmp_path),
                "disable_caching": True, "verbosity": "WARNING",
                "progress_bar": "none"}
    with tempconfig(settings):
        scene = ChalkdustScene(make_component(NAME, params), duration=budget)
        scene.render()
        elapsed = scene.renderer.time
    assert elapsed == pytest.approx(budget, abs=1 / DRAFT_FPS)


def test_min_seconds_is_sum_of_segment_minimums():
    comp = make_component(NAME, EXAMPLES[0])   # four forces
    assert comp.min_seconds() == pytest.approx(BODY_S + 4 * FORCE_S + HOLD_S)


def test_latex_strings_lists_every_label_in_order():
    comp = make_component(NAME, EXAMPLES[0])
    assert comp.latex_strings() == [f["label"] for f in EXAMPLES[0]["forces"]]


# --- schema ---------------------------------------------------------------------


def _one(**force) -> dict:
    return {"body": "m", "forces": [{"label": "F", "angle": 0, **force}]}


@pytest.mark.parametrize("params", [
    {"body": "m", "forces": []},
    {"body": "   ", "forces": [{"label": "F", "angle": 0}]},
    {"body": "m", "forces": [{"label": " ", "angle": 0}]},
    {"body": "m", "forces": [{"label": "F", "angle": 0}] * 13},
    {"body": "m", "forces": [{"label": "F"}]},
    _one(magnitude=0),
    _one(magnitude=-3),
    _one(magnitude=float("inf")),
    _one(magnitude=float("nan")),
    _one(angle=400),
    _one(angle=float("nan")),
    _one(colour="red"),
], ids=["no-forces", "blank-body", "blank-label", "too-many-forces",
        "no-angle", "zero-magnitude", "negative-magnitude", "inf-magnitude",
        "nan-magnitude", "angle-out-of-range", "nan-angle", "unknown-force-param"])
def test_schema_rejects(params):
    with pytest.raises(ValidationError):
        FreeBodyDiagram(params)


def test_magnitude_defaults_to_equal_arrows():
    probe = _probe({"body": "m", "forces": [{"label": "A", "angle": 0},
                                             {"label": "B", "angle": 180}]})
    a, b = _parts(probe)[0]
    assert a.get_length() == pytest.approx(b.get_length())


# --- typed LaTeX failure -----------------------------------------------------------


@pytest.mark.parametrize("bad", [r"\frac{m", r"\notacommand{g}", r"\,"],
                         ids=["unclosed-brace", "undefined-command", "draws-nothing"])
def test_invalid_latex_is_a_typed_refusal(bad):
    params = {"body": "m", "forces": [{"label": "N", "angle": 90},
                                      {"label": bad, "angle": 270}]}
    with pytest.raises(LayoutError) as exc:
        _probe(params)
    assert exc.value.kind == "invalid_latex"
    assert "forces[1]" in str(exc.value)
    # And the probe reports it as that kind, not as a crash.
    assert validate_beat(_spec(params)).kinds() == {"invalid_latex"}


# --- arrow sizing ----------------------------------------------------------------


def test_lengths_proportional_above_the_clamp():
    # Example 0: mg = N = 19.6 (the largest), T = 10, f_k = 4.
    arrows, _ = _parts(_probe(EXAMPLES[0]))
    mg, n, t, fk = (a.get_length() for a in arrows)
    assert n == pytest.approx(mg)
    assert t / mg == pytest.approx(10 / 19.6)
    # 4 / 19.6 = 0.20 is under the floor, so f_k is held there.
    assert fk / mg == pytest.approx(ARROW_MIN_FRACTION)


def test_tiny_force_held_at_floor_and_huge_force_at_ceiling():
    params = {"body": "grain", "forces": [{"label": "a", "angle": 270, "magnitude": 1e-9},
                                          {"label": "b", "angle": 90, "magnitude": 1e9}]}
    tiny, huge = (a.get_length() for a in _parts(_probe(params))[0])
    assert tiny / huge == pytest.approx(ARROW_MIN_FRACTION)


def test_every_arrowhead_is_the_same_size():
    # Manim's default thins short arrows; length must be the only encoding.
    arrows, _ = _parts(_probe(EXAMPLES[0]))
    tips = [a.get_tip().height for a in arrows if abs(a.get_unit_vector()[0]) < 1e-6]
    sides = [a.get_tip().width for a in arrows if abs(a.get_unit_vector()[1]) < 1e-6]
    assert tips and sides
    assert tips + sides == pytest.approx([tips[0]] * len(tips + sides))
    assert {a.get_stroke_width() for a in arrows} == {arrows[0].get_stroke_width()}


# --- clustering and label placement -------------------------------------------------


def test_parallel_forces_drawn_side_by_side_from_the_face():
    params = {"body": "crate",
              "forces": [{"label": f"F_{i}", "angle": 270} for i in range(4)]}
    probe = _probe(params)
    arrows, _ = _parts(probe)
    body = next(m for m in probe.mobjects if getattr(m, "_chalk_label", "") == "body")
    xs = sorted(a.get_start()[0] for a in arrows)
    assert all(b - a > 0.1 for a, b in zip(xs, xs[1:]))
    box = bbox(body)
    for a in arrows:
        assert a.get_start()[1] == pytest.approx(box.bottom, abs=1e-6)
        assert box.left < a.get_start()[0] < box.right


def test_directions_either_side_of_zero_cluster_together():
    # 355 and 5 degrees are 10 degrees apart, not 350.
    params = {"body": "puck", "forces": [{"label": "a", "angle": 355},
                                         {"label": "b", "angle": 5}]}
    a, b = _parts(_probe(params))[0]
    # Side by side, in angle order: 355 below, 5 above, never crossing.
    assert a.get_start()[1] < b.get_start()[1]
    assert a.get_end()[1] < b.get_end()[1]


def _fixtures_that_fit():
    out = [pytest.param(p, id=f"ex{i}") for i, p in enumerate(EXAMPLES)]
    for i, p in enumerate(FreeBodyDiagram.stress()):
        out.append(pytest.param(p, id=f"stress{i}"))
    return out


@pytest.mark.parametrize("params", _fixtures_that_fit())
def test_labels_clear_every_other_arrow_and_each_other(params):
    try:
        probe = _probe(params)
    except LayoutError as exc:
        assert exc.kind == "overflow"   # refused cleanly; nothing was drawn
        return
    arrows, labels = _parts(probe)
    assert len(arrows) == len(labels) == len(params["forces"])
    for i, tex in enumerate(labels):
        box = bbox(tex)
        for j, arrow in enumerate(arrows):
            if j != i:
                assert not _segment_hits_rect(arrow.get_start(), arrow.get_end(), box), (
                    f"label[{i}] touches force[{j}]")
        for j, other in enumerate(labels[i + 1:], start=i + 1):
            assert not box.intersects(bbox(other)), f"label[{i}] overlaps label[{j}]"


def test_triple_volume_refuses_as_overflow():
    # stress()[0] is 12 long labels round a long-named body: too dense to stay
    # legible, and it must say so as overflow (split the beat), not crash.
    with pytest.raises(LayoutError) as exc:
        _probe(FreeBodyDiagram.stress()[0])
    assert exc.value.kind == "overflow"
