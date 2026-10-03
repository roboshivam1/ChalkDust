"""AnswerBox: the behaviours the registry-wide layout tests cannot pin.

tests/test_layout.py already proves examples() validate clean and stress()
fits or refuses cleanly. This file pins what is specific to this component:
timing, the schema's refusals, the typed LaTeX failures, the composition
(label above the box, box around the answer), and that the cheap layout probe
agrees with a real draft render.

These tests compile LaTeX, so `latex` and `dvisvgm` must be on PATH.
"""

from __future__ import annotations

import pytest
from manim import tempconfig
from pydantic import ValidationError

from chalkdust.core.models import BeatSpec, Region
from chalkdust.scenes.base import ChalkdustScene
from chalkdust.scenes.components import make_component
from chalkdust.scenes.components.answer_box import (
    BOX_S,
    DWELL_S,
    OPTION_S,
    REVEAL_S,
    AnswerBox,
)
from chalkdust.scenes.regions import MIN_FONT_SIZE, LayoutError, bbox, region_rect
from chalkdust.validate.geometric import LayoutProbe, validate_beat

NAME = "AnswerBox"
EXAMPLES = AnswerBox.examples()
DRAFT_FPS = 15


def _spec(params: dict) -> BeatSpec:
    return BeatSpec(id="b01", narration="placeholder narration",
                    component=NAME, params=params)


def _probe(params: dict, duration: float = 8.0) -> LayoutProbe:
    probe = LayoutProbe(make_component(NAME, params), duration=duration)
    probe.construct()
    return probe


def _labelled(scene) -> list[tuple[str, object]]:
    """(label, mobject) for every top-level scene mobject that draws anything,
    in scene order.

    Mobjects with no points anywhere are skipped: a real Scene.wait() leaves
    its Wait animation's empty placeholder Mobject in scene.mobjects for good,
    while the probe's wait() is a no-op. Those placeholders draw nothing."""
    return [(getattr(m, "_chalk_label", type(m).__name__), m) for m in scene.mobjects
            if any(sm.has_points() for sm in m.get_family())]


class _Clock(LayoutProbe):
    """A probe that also adds up the time every play() and wait() asks for --
    the scene's elapsed time, without encoding a frame. A play() with no
    explicit run_time counts at its animations' own default, so a forgotten
    run_time shows up as a budget mismatch rather than passing silently."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.elapsed = 0.0

    def play(self, *animations, **kwargs) -> None:  # type: ignore[override]
        self.elapsed += kwargs.get("run_time", max(a.run_time for a in animations
                                                   if hasattr(a, "run_time")))
        super().play(*animations, **kwargs)

    def wait(self, duration: float = 1.0, *args, **kwargs) -> None:  # type: ignore[override]
        self.elapsed += duration


# --- timing (D-002) -----------------------------------------------------------


@pytest.mark.parametrize("factor", [0.5, 3.0], ids=["short", "long"])
@pytest.mark.parametrize("params", EXAMPLES, ids=[f"ex{i}" for i in range(len(EXAMPLES))])
def test_timing_consumes_budget_exactly(params, factor):
    # Narration far shorter and far longer than the animation wants: either
    # way the beat must last exactly as long as its audio.
    budget = make_component(NAME, params).min_seconds() * factor
    clock = _Clock(make_component(NAME, params), duration=budget)
    clock.construct()
    assert clock.elapsed == pytest.approx(budget, abs=1 / DRAFT_FPS)


@pytest.mark.parametrize("params,expected", [
    (EXAMPLES[0], REVEAL_S + OPTION_S + BOX_S + DWELL_S),   # value + option
    (EXAMPLES[1], REVEAL_S + BOX_S + DWELL_S),              # value only
    (EXAMPLES[2], REVEAL_S + BOX_S + DWELL_S),              # option only
], ids=["value+option", "value", "option"])
def test_min_seconds_is_sum_of_segment_minimums(params, expected):
    assert make_component(NAME, params).min_seconds() == pytest.approx(expected)


def test_latex_strings_is_exactly_what_build_compiles():
    # Value and units are one MathTex; units are set upright with thin spaces
    # between their factors.
    comp = make_component(NAME, {"value": "F = 9.8", "units": "kg m s^{-2}"})
    assert comp.latex_strings() == [r"F = 9.8 \; \mathrm{kg\,m\,s^{-2}}"]
    assert make_component(NAME, {"value": "7"}).latex_strings() == ["7"]
    # An option-only answer compiles no LaTeX at all.
    assert make_component(NAME, {"option": "C"}).latex_strings() == []


# --- schema ---------------------------------------------------------------------


@pytest.mark.parametrize("params", [
    {},
    {"value": "   "},
    {"units": "N"},
    {"units": "N", "option": "A"},
    {"value": "", "option": "A"},
    {"option": "E"},
    {"option": "b"},
    {"option": "AB"},
    {"option": ""},
], ids=["empty", "blank-value", "units-only", "units-without-value",
        "empty-value", "option-out-of-range", "option-lowercase",
        "option-two-letters", "option-empty"])
def test_schema_rejects(params):
    with pytest.raises(ValidationError):
        AnswerBox(params)


@pytest.mark.parametrize("params,labels", [
    ({"value": "0"}, ["answer", "box", "label"]),
    ({"option": "A"}, ["box", "label", "option"]),
], ids=["value", "option"])
def test_minimal_answers_build(params, labels):
    probe = _probe(params)
    assert sorted(name for name, _ in _labelled(probe)) == labels


# --- typed refusals ---------------------------------------------------------------


@pytest.mark.parametrize("params", [
    {"value": r"\notacommand{x} = 1"},       # LaTeX compile error
    {"value": r"\quad"},                     # compiles, draws nothing
    {"value": r"\frac{1}{"},                 # LaTeX silently recovers from this
    {"value": "x = 1", "units": r"m}\frac{1}{2"},   # escapes the \mathrm wrapper
], ids=["compile-error", "renders-nothing", "unbalanced-value", "unbalanced-units"])
def test_invalid_latex_is_a_typed_finding(params):
    report = validate_beat(_spec(params))
    assert report.kinds() == {"invalid_latex"}, f"\n{report}"


def test_over_long_answer_refuses_rather_than_shrinks():
    # 3x volume (stress case 0) must refuse, never render at an illegible size.
    with pytest.raises(LayoutError) as exc:
        _probe(AnswerBox.stress()[0])
    assert exc.value.kind == "overflow"


# --- layout -----------------------------------------------------------------------


@pytest.mark.parametrize("params", EXAMPLES, ids=[f"ex{i}" for i in range(len(EXAMPLES))])
def test_box_encloses_answer_under_its_label(params):
    mobs = dict(_labelled(_probe(params)))
    box, tag = bbox(mobs["box"]), bbox(mobs["label"])
    for name in ("answer", "option"):
        if name in mobs:
            assert box.contains(bbox(mobs[name])), name
    assert tag.bottom > box.top
    assert tag.x == pytest.approx(box.x, abs=1e-6)
    # The composition sits inside the stage, centred on it.
    stage = region_rect(Region.STAGE)
    assert stage.contains(tag) and stage.contains(box)


def test_badge_sits_left_of_value_on_its_centre_line():
    mobs = dict(_labelled(_probe(EXAMPLES[0])))
    badge, answer = bbox(mobs["option"]), bbox(mobs["answer"])
    assert badge.right < answer.left
    assert badge.y == pytest.approx(answer.y, abs=1e-6)


def test_label_is_never_scaled_with_the_answer():
    # The caption-sized label stays out of the fitted group; if it were scaled
    # with a long answer it would drop below the legibility floor first.
    probe = _probe(AnswerBox.stress()[1])   # 60-char identifier, scaled down
    mobs = dict(_labelled(probe))
    assert mobs["answer"]._chalk_font_size < 60
    assert mobs["label"]._chalk_font_size == pytest.approx(probe.theme.type.caption)
    assert mobs["answer"]._chalk_font_size >= MIN_FONT_SIZE


def test_probe_matches_real_draft_render(tmp_path):
    """The cheap probe (validation rung 3) must reach the same final frame as
    a real 480p15 render: same mobjects, same boxes, same tracked font sizes.
    If these diverge, the probe is validating a scene nobody will see."""
    params = EXAMPLES[0]
    duration = make_component(NAME, params).min_seconds() * 2

    probe = _probe(params, duration=duration)

    with tempconfig({"media_dir": str(tmp_path), "pixel_width": 854,
                     "pixel_height": 480, "frame_rate": DRAFT_FPS,
                     "disable_caching": True, "progress_bar": "none",
                     "verbosity": "WARNING", "output_file": "probe_vs_render"}):
        scene = ChalkdustScene(make_component(NAME, params), duration=duration)
        scene.render()

    got, want = _labelled(scene), _labelled(probe)
    assert sorted(n for n, _ in got) == sorted(n for n, _ in want)
    real_by_name, probed_by_name = dict(got), dict(want)
    for name, probed in probed_by_name.items():
        r, p = bbox(real_by_name[name]), bbox(probed)
        assert (r.x, r.y, r.width, r.height) == pytest.approx(
            (p.x, p.y, p.width, p.height), abs=1e-3), name
        assert getattr(real_by_name[name], "_chalk_font_size", None) == pytest.approx(
            getattr(probed, "_chalk_font_size", None)), name

    # Manim quantises every play()/wait() to whole frames, so a real render can
    # differ from the budget by up to one frame per timed segment. Closing that
    # gap is the mux stage's job; bound it here so it can never grow beyond
    # quantisation.
    segments = len(make_component(NAME, params)._segments())
    assert scene.renderer.time == pytest.approx(duration, abs=segments / DRAFT_FPS)
