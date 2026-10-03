"""EquationDerivation: the behaviours the registry-wide layout tests cannot pin.

tests/test_layout.py already proves examples() validate clean and stress()
fits or refuses cleanly. This file pins what is specific to this component:
timing, the schema's refusals, the typed LaTeX failure, font-size tracking
through transforms, annotation placement, and that the cheap layout probe
agrees with a real draft render.

These tests compile LaTeX, so `latex` and `dvisvgm` must be on PATH.
"""

from __future__ import annotations

import pytest
from manim import tempconfig
from pydantic import ValidationError

from chalkdust.core.models import BeatSpec
from chalkdust.scenes.base import ChalkdustScene
from chalkdust.scenes.components import make_component
from chalkdust.scenes.components.equation_derivation import (
    ANNOTATE_S,
    DWELL_S,
    TRANSFORM_S,
    WRITE_S,
    EquationDerivation,
)
from chalkdust.scenes.regions import MIN_FONT_SIZE, LayoutError, bbox
from chalkdust.validate.geometric import LayoutProbe, validate_beat

NAME = "EquationDerivation"
EXAMPLES = EquationDerivation.examples()
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


def test_min_seconds_is_sum_of_segment_minimums():
    # Example 1: three steps, one annotation (on the last step).
    comp = make_component(NAME, EXAMPLES[1])
    expected = WRITE_S + 2 * TRANSFORM_S + ANNOTATE_S + 3 * DWELL_S
    assert comp.min_seconds() == pytest.approx(expected)


def test_latex_strings_lists_every_step_in_order():
    comp = make_component(NAME, EXAMPLES[0])
    assert comp.latex_strings() == EXAMPLES[0]["steps"]


# --- schema ---------------------------------------------------------------------


@pytest.mark.parametrize("params", [
    {"steps": []},
    {"steps": ["   "]},
    {"steps": ["x"] * 13},
    {"steps": ["x = 1"], "annotations": [{"step": 0, "text": " "}]},
    {"steps": ["x = 1"], "annotations": [{"step": 1, "text": "out of range"}]},
    {"steps": ["x = 1"], "annotations": [{"step": -1, "text": "negative"}]},
    {"steps": ["x = 1"], "annotations": [{"step": 0, "text": "one"},
                                         {"step": 0, "text": "two"}]},
], ids=["no-steps", "blank-step", "too-many-steps", "blank-note",
        "note-out-of-range", "note-negative", "two-notes-one-step"])
def test_schema_rejects(params):
    with pytest.raises(ValidationError):
        EquationDerivation(params)


def test_minimal_single_step_builds():
    probe = _probe({"steps": ["x"]})
    assert [name for name, _ in _labelled(probe)] == ["step[0]"]


# --- typed refusals ---------------------------------------------------------------


@pytest.mark.parametrize("bad", [r"\notacommand{x} = 1", r"\quad"],
                         ids=["compile-error", "renders-nothing"])
def test_invalid_latex_is_a_typed_finding(bad):
    report = validate_beat(_spec({"steps": ["x = 1", bad]}))
    assert report.kinds() == {"invalid_latex"}, f"\n{report}"
    assert "step[1]" in report.findings[0].message


def test_over_long_derivation_refuses_rather_than_shrinks():
    # 3x volume (stress case 0) must refuse, never render at an illegible size.
    with pytest.raises(LayoutError) as exc:
        _probe(EquationDerivation.stress()[0])
    assert exc.value.kind == "overflow"


# --- layout -----------------------------------------------------------------------


def test_font_size_tracking_survives_transforms():
    # After every TransformMatchingTex, the step left on screen must be the
    # fitted, tagged target -- not an untagged transform group, and not the
    # temporary source copy. Otherwise the probe's legibility check is blind.
    params = EXAMPLES[0]
    probe = _probe(params)
    steps = [m for name, m in _labelled(probe) if name.startswith("step[")]
    assert len(steps) == len(params["steps"])
    sizes = {round(m._chalk_font_size, 6) for m in steps}
    assert len(sizes) == 1 and min(sizes) >= MIN_FONT_SIZE
    # Nothing else lingers: exactly the steps and the notes.
    assert len(probe.mobjects) == len(params["steps"]) + len(params["annotations"])


def test_annotations_sit_beside_their_step():
    probe = _probe(EXAMPLES[0])
    mobs = dict(_labelled(probe))
    column_right = max(bbox(m).right for name, m in mobs.items() if name.startswith("step["))
    for a in EXAMPLES[0]["annotations"]:
        note, step = bbox(mobs[f"annotation[{a['step']}]"]), bbox(mobs[f"step[{a['step']}]"])
        assert note.y == pytest.approx(step.y, abs=1e-6)
        assert note.left > column_right


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
    assert [n for n, _ in got] == [n for n, _ in want]
    for (name, real), (_, probed) in zip(got, want):
        r, p = bbox(real), bbox(probed)
        assert (r.x, r.y, r.width, r.height) == pytest.approx(
            (p.x, p.y, p.width, p.height), abs=1e-3), name
        assert getattr(real, "_chalk_font_size", None) == pytest.approx(
            getattr(probed, "_chalk_font_size", None)), name

    # Manim quantises every play()/wait() to whole frames, so a real render can
    # differ from the budget by up to one frame per timed segment. Closing that
    # gap is the mux stage's job (register item D-3); bound it here so it can
    # never grow beyond quantisation.
    segments = len(make_component(NAME, params)._segments())
    assert scene.renderer.time == pytest.approx(duration, abs=segments / DRAFT_FPS)
