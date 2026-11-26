"""AnswerBox: the behaviours the registry-wide layout tests cannot pin.

tests/test_layout.py already proves examples() validate clean and stress()
fits or refuses cleanly. This file pins what is specific to this component:
timing, the schema's refusals, the typed LaTeX failures, the composition
(label above the box, box around the answer), the carry-in artifact, and that
the cheap layout probe agrees with a real draft render.

These tests compile LaTeX, so `latex` and `dvisvgm` must be on PATH, and the
real renders count frames with ffprobe.
"""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import asdict

import pytest
from manim import tempconfig
from pydantic import ValidationError

from chalkdust.continuity import ArtifactRecipe, build_artifact, resolve_carry_in
from chalkdust.core.models import BeatSpec, Quality, Region, VideoSpec
from chalkdust.render.worker import TIERS
from chalkdust.scenes.base import ChalkdustScene, whole_frames
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
DRAFT = TIERS[Quality.DRAFT]
DRAFT_FPS = DRAFT.frame_rate


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
    """A probe that also counts the frames every play() and wait() renders --
    the scene's length, without encoding one. Each run time is counted the
    way ChalkdustScene.get_run_time makes Manim render it (nearest whole
    frame, at least one). A play() with no explicit run_time counts at its
    animations' own default, so a forgotten run_time shows up as a frame
    mismatch rather than passing silently."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.frames = 0

    def play(self, *animations, **kwargs) -> None:  # type: ignore[override]
        run_time = kwargs.get("run_time", max(a.run_time for a in animations
                                              if hasattr(a, "run_time")))
        self.frames += whole_frames(run_time, self.fps)
        super().play(*animations, **kwargs)

    def wait(self, duration: float = 1.0, *args, **kwargs) -> None:  # type: ignore[override]
        self.frames += whole_frames(duration, self.fps)


def _render(params: dict, duration: float, out_dir, name: str) -> tuple[ChalkdustScene, int]:
    """Render at the draft tier; return the scene and the clip's frame count,
    read back from the encoded file with ffprobe -count_frames."""
    with tempconfig({**asdict(DRAFT), "media_dir": str(out_dir),
                     "disable_caching": True, "progress_bar": "none",
                     "verbosity": "WARNING", "output_file": name}):
        scene = ChalkdustScene(make_component(NAME, params), duration=duration)
        scene.render()
        movie = scene.renderer.file_writer.movie_file_path
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames", "-of", "json", str(movie)],
        capture_output=True, text=True, check=True).stdout
    return scene, int(json.loads(out)["streams"][0]["nb_read_frames"])


# --- timing (D-002) -----------------------------------------------------------


@pytest.mark.parametrize("factor", [0.5, 3.0], ids=["short", "long"])
@pytest.mark.parametrize("params", EXAMPLES, ids=[f"ex{i}" for i in range(len(EXAMPLES))])
def test_timing_consumes_budget_exactly(params, factor):
    # Narration far shorter and far longer than the animation wants: either
    # way the beat lasts exactly its audio rounded up to whole frames.
    budget = make_component(NAME, params).min_seconds() * factor
    with tempconfig({"frame_rate": DRAFT_FPS}):
        clock = _Clock(make_component(NAME, params), duration=budget)
    clock.construct()
    assert clock.beat_frames == math.ceil(budget * DRAFT_FPS)
    assert clock.frames == clock.beat_frames


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


@pytest.mark.parametrize("params,field", [
    ({"value": r"\notacommand{x} = 1"}, "answer"),       # LaTeX compile error
    ({"value": r"\quad"}, "answer"),                     # compiles, draws nothing
    ({"value": r"\frac{1}{"}, "value"),                  # LaTeX silently recovers
    ({"value": "x = 1", "units": r"m}\frac{1}{2"}, "units"),   # escapes \mathrm
    ({"value": "50%", "units": "N"}, "value"),            # a comment, named as such
], ids=["compile-error", "renders-nothing", "unbalanced-value", "unbalanced-units",
        "unescaped-percent"])
def test_invalid_latex_is_a_typed_finding(params, field):
    # Each is a stress() case, so the registry-wide stress test covers it too.
    assert params in AnswerBox.stress()
    report = validate_beat(_spec(params))
    assert report.kinds() == {"invalid_latex"}, f"\n{report}"
    assert report.findings[0].message.startswith(f"{field} is not valid LaTeX")


def test_escaped_percent_is_valid():
    # \% is a percent sign, not a comment: it must not be refused.
    probe = _probe({"value": r"\eta = 40\%"})
    assert "answer" in dict(_labelled(probe))


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

    scene, frames = _render(params, duration, tmp_path, "probe_vs_render")

    got, want = _labelled(scene), _labelled(probe)
    assert sorted(n for n, _ in got) == sorted(n for n, _ in want)
    real_by_name, probed_by_name = dict(got), dict(want)
    for name, probed in probed_by_name.items():
        r, p = bbox(real_by_name[name]), bbox(probed)
        assert (r.x, r.y, r.width, r.height) == pytest.approx(
            (p.x, p.y, p.width, p.height), abs=1e-3), name
        assert getattr(real_by_name[name], "_chalk_font_size", None) == pytest.approx(
            getattr(probed, "_chalk_font_size", None)), name

    # The clip is exactly the beat: its audio rounded up to whole frames.
    assert frames == math.ceil(duration * DRAFT_FPS)


@pytest.mark.parametrize("factor", [0.5, 3.0], ids=["short", "long"])
def test_real_render_is_exactly_the_beat(tmp_path, factor):
    """_Clock counts the frames build() asks for. This renders for real at
    480p15 and counts the frames in the encoded clip, at narration far
    shorter and far longer than the answer wants: ceil(audio * fps), exactly.
    (1.625 s -> 25 frames; 9.75 s -> 147.)"""
    params = EXAMPLES[0]
    budget = make_component(NAME, params).min_seconds() * factor
    _, frames = _render(params, budget, tmp_path, f"budget_{factor}")
    assert frames == math.ceil(budget * DRAFT_FPS)


# --- continuity (SCENE_SPEC.md §6) ------------------------------------------------


def _parts_by_label(mob) -> dict[str, object]:
    return {m._chalk_label: m for m in mob.get_family()
            if getattr(m, "_chalk_label", None) in ("answer", "box", "label", "option")}


@pytest.mark.parametrize("params", EXAMPLES, ids=[f"ex{i}" for i in range(len(EXAMPLES))])
def test_carried_artifact_is_the_settled_answer(params):
    # Rebuilt from the same params, the artifact is the answer the viewer saw:
    # the same parts, the same sizes, the same arrangement (up to where it is
    # placed). Realistic answers fit the stage unscaled, so sizes match 1:1.
    probe = _probe(params)
    seen = dict(_labelled(probe))
    recipe = ArtifactRecipe(name="answer", producer=NAME, params=params)
    rebuilt = _parts_by_label(build_artifact(recipe, probe.theme))
    assert sorted(rebuilt) == sorted(seen)
    shift = bbox(seen["box"]).center - bbox(rebuilt["box"]).center
    for name, mob in rebuilt.items():
        r, s = bbox(mob), bbox(seen[name])
        assert (r.x + shift[0], r.y + shift[1], r.width, r.height) == pytest.approx(
            (s.x, s.y, s.width, s.height), abs=1e-6), name
        assert getattr(mob, "_chalk_font_size", None) == pytest.approx(
            getattr(seen[name], "_chalk_font_size", None)), name


def test_answer_carries_into_a_later_beat(hold_consumer):
    # An AnswerBox beat can register its answer and a later beat carry it in:
    # the spec resolves (an AnswerBox producer has a builder), and the carried
    # answer passes the layout ladder in the consuming beat, at its own font
    # sizes. Where the consumer draws relative to it is the consumer's job.
    # Carried into _Hold, a consumer: a beat that carries an artifact in
    # without acting on it must leave it a free STAGE region (D-G4c-1).
    video = VideoSpec(video_id="v", beats=(
        BeatSpec(id="b01", narration="placeholder narration", component=NAME,
                 params=EXAMPLES[0], registers="answer"),
        BeatSpec(id="b02", narration="placeholder narration",
                 component=hold_consumer, params={"target_id": "answer"},
                 carry_in=["answer"]),
    ))
    recipes = resolve_carry_in(video)["b02"]
    assert [r.producer for r in recipes] == [NAME]
    report = validate_beat(video.beats[1], recipes=recipes)
    assert report.ok, f"\n{report}"
