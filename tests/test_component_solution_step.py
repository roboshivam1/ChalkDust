"""SolutionStep: the behaviours the registry-wide layout tests cannot pin.

tests/test_layout.py already proves examples() validate clean and stress()
fits or refuses cleanly. This file pins what is specific to this component:
timing in whole frames, the schema's refusals, the typed LaTeX failure, the
maths/prose split behind latex_strings(), where each part lands, the carry-in
artifact, and that the cheap layout probe agrees with a real draft render.

These tests compile LaTeX, so `latex` and `dvisvgm` must be on PATH.
"""

from __future__ import annotations

import json
import math
import subprocess

import numpy as np
import pytest
from manim import tempconfig
from manim.animation.animation import prepare_animation
from pydantic import ValidationError

from chalkdust.continuity import (
    ArtifactRecipe,
    beat_component,
    build_artifact,
    carried,
    resolve_carry_in,
)
from chalkdust.core.models import BeatSpec, Region, VideoSpec
from chalkdust.scenes.base import ChalkdustScene
from chalkdust.scenes.components import make_component
from chalkdust.scenes.components.solution_step import (
    HEADER_S,
    HOLD_S,
    JUSTIFY_S,
    LINE_S,
    READ_S,
    SolutionStep,
)
from chalkdust.scenes.regions import (
    DEFAULT_PADDING,
    MIN_FONT_SIZE,
    LayoutError,
    bbox,
    region_rect,
    smallest_font_size,
)
from chalkdust.scenes.theme import get_theme
from chalkdust.validate.geometric import LayoutProbe, validate_beat

NAME = "SolutionStep"
EXAMPLES = SolutionStep.examples()
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
    """A probe that also records the run time every play() and wait() asks
    for -- the scene's timeline, without encoding a frame. A play() with no
    explicit run_time counts at its animations' own default, so a forgotten
    run_time shows up as a frame mismatch rather than passing silently."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.run_times: list[float] = []

    def play(self, *animations, **kwargs) -> None:  # type: ignore[override]
        run_time = kwargs.get("run_time")
        if run_time is None:
            # prepare_animation turns `.animate` builders into Animations.
            run_time = max(prepare_animation(a).run_time for a in animations)
        self.run_times.append(run_time)
        super().play(*animations, **kwargs)

    def wait(self, duration: float = 1.0, *args, **kwargs) -> None:  # type: ignore[override]
        self.run_times.append(duration)


def _movie_frames(scene: ChalkdustScene) -> int:
    """Frames in a rendered scene's movie file, counted by ffprobe."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames", "-of", "json",
         str(scene.renderer.file_writer.movie_file_path)],
        capture_output=True, text=True, check=True).stdout
    return int(json.loads(out)["streams"][0]["nb_read_frames"])


# --- timing (D-002) -----------------------------------------------------------


@pytest.mark.parametrize("factor", [0.5, 3.0], ids=["short", "long"])
@pytest.mark.parametrize("params", EXAMPLES, ids=[f"ex{i}" for i in range(len(EXAMPLES))])
def test_timing_is_whole_frames_summing_to_the_beat(params, factor, tmp_path):
    # Narration far shorter and far longer than the animation wants: either
    # way every play and wait is a whole number of draft frames, and together
    # they are exactly the beat's frames -- ceil(audio * fps), no drift.
    budget = make_component(NAME, params).min_seconds() * factor
    with tempconfig({"frame_rate": DRAFT_FPS}):
        clock = _Clock(make_component(NAME, params), duration=budget,
                       media_dir=tmp_path)
        clock.construct()
    frames = [t * DRAFT_FPS for t in clock.run_times]
    assert all(f == pytest.approx(round(f), abs=1e-9) and round(f) >= 1 for f in frames)
    assert sum(round(f) for f in frames) == clock.beat_frames
    assert clock.beat_frames == math.ceil(budget * DRAFT_FPS)


def test_min_seconds_is_sum_of_segment_minimums():
    # Example 0: two lines of work and a justification.
    comp = make_component(NAME, EXAMPLES[0])
    expected = HEADER_S + 2 * (LINE_S + READ_S) + JUSTIFY_S + HOLD_S
    assert comp.min_seconds() == pytest.approx(expected)
    # Example 1: one line, no justification -- no justification segment.
    comp = make_component(NAME, EXAMPLES[1])
    assert comp.min_seconds() == pytest.approx(HEADER_S + LINE_S + READ_S + HOLD_S)


# --- maths vs prose -------------------------------------------------------------


def test_latex_strings_lists_math_work_in_order():
    comp = make_component(NAME, EXAMPLES[0])
    assert comp.latex_strings() == EXAMPLES[0]["work"]


def test_prose_work_compiles_no_latex():
    # Prose work is plain Text: there is nothing for the semantic rung to
    # compile, and a stray "$" or "\" in it must not be treated as maths.
    comp = make_component(NAME, {"n": 1, "claim": "Price", "work": [r"costs $5 \ each"],
                                 "work_format": "text"})
    assert comp.latex_strings() == []
    probe = _probe(comp.params.model_dump())
    assert [n for n, _ in _labelled(probe)] == ["header", "work[0]"]


# --- schema ---------------------------------------------------------------------


@pytest.mark.parametrize("params", [
    {"n": 1, "claim": "c", "work": []},
    {"n": 1, "claim": "c", "work": ["  "]},
    {"n": 1, "claim": "  ", "work": ["x"]},
    {"n": 1, "claim": "", "work": ["x"]},
    {"n": 0, "claim": "c", "work": ["x"]},
    {"n": 100, "claim": "c", "work": ["x"]},
    {"n": 1, "claim": "c", "work": ["x"] * 7},
    {"n": 1, "claim": "c", "work": ["x"], "justification": " "},
    {"n": 1, "claim": "c", "work": ["x"], "work_format": "markdown"},
    {"n": 1, "claim": "c", "work": "x = 1"},
    {"claim": "c", "work": ["x"]},
], ids=["no-work", "blank-work-line", "blank-claim", "empty-claim", "n-zero",
        "n-three-digits", "too-much-work", "blank-justification", "unknown-format",
        "work-not-a-list", "missing-n"])
def test_schema_rejects(params):
    with pytest.raises(ValidationError):
        SolutionStep(params)


def test_minimal_step_builds():
    probe = _probe({"n": 1, "claim": "x", "work": ["x"]})
    assert [n for n, _ in _labelled(probe)] == ["header", "work[0]"]


def test_regions_follow_justification():
    with_j = make_component(NAME, EXAMPLES[0])
    without = make_component(NAME, EXAMPLES[1])
    assert with_j.regions() == {Region.TITLE_BAR, Region.STAGE, Region.LOWER_THIRD}
    assert without.regions() == {Region.TITLE_BAR, Region.STAGE}


# --- typed refusals ---------------------------------------------------------------


def _latex_stress_cases() -> list[dict]:
    return [c for c in SolutionStep.stress() if c.get("claim") == "Simplify"]


@pytest.mark.parametrize("params", _latex_stress_cases(),
                         ids=["compile-error", "renders-nothing"])
def test_invalid_latex_is_a_typed_finding_naming_the_line(params):
    # The stress() cases with bad LaTeX in work[1]: test_layout only checks
    # they refuse cleanly; this pins that they refuse with exactly
    # invalid_latex, and that the refusal names the line to regenerate.
    report = validate_beat(_spec(params))
    assert report.kinds() == {"invalid_latex"}, f"\n{report}"
    assert "work[1]" in report.findings[0].message


def test_over_long_work_refuses_rather_than_shrinks():
    # 3x volume (stress case 0) must refuse, never render at an illegible size.
    with pytest.raises(LayoutError) as exc:
        _probe(SolutionStep.stress()[0])
    assert exc.value.kind == "overflow"


# --- layout -----------------------------------------------------------------------


@pytest.mark.parametrize("params", EXAMPLES, ids=[f"ex{i}" for i in range(len(EXAMPLES))])
def test_each_part_sits_in_its_region(params):
    # The header lives in the title bar, the work on the stage, the
    # justification in the lower third -- whatever the content's size.
    mobs = dict(_labelled(_probe(params)))
    placed = {"header": Region.TITLE_BAR, "justification": Region.LOWER_THIRD}
    placed |= {f"work[{i}]": Region.STAGE for i in range(len(params["work"]))}
    expected = [n for n in placed if n != "justification" or "justification" in params]
    assert sorted(mobs) == sorted(expected)
    for name, region in placed.items():
        if name in mobs:
            assert region_rect(region).contains(bbox(mobs[name])), name


def test_header_is_pinned_left_with_badge_before_claim():
    mobs = dict(_labelled(_probe(EXAMPLES[0])))
    header = mobs["header"]
    badge, claim = header.submobjects
    assert bbox(badge).right < bbox(claim).left
    # Pinned to the title bar's padded left edge, not centred.
    inner = region_rect(Region.TITLE_BAR).inset(DEFAULT_PADDING)
    assert bbox(header).left == pytest.approx(inner.left, abs=1e-6)


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

    # The rendered clip is exactly the beat: ceil(audio * fps) frames.
    assert _movie_frames(scene) == scene.beat_frames == math.ceil(duration * DRAFT_FPS)


# --- carry-in (SCENE_SPEC.md §6) --------------------------------------------------


def test_artifact_rebuild_is_deterministic():
    recipe = ArtifactRecipe(name="step", producer=NAME, params=EXAMPLES[0])
    theme = get_theme("default")
    a, b = build_artifact(recipe, theme), build_artifact(recipe, theme)
    pa = [m.points for m in a.family_members_with_points()]
    pb = [m.points for m in b.family_members_with_points()]
    assert len(pa) == len(pb) > 0
    assert all(np.array_equal(x, y) for x, y in zip(pa, pb))
    # The work, line by line as build() stacks it -- not the header or the
    # justification, which caption the work in their own regions.
    assert [m._chalk_label for m in a.submobjects] == ["work[0]", "work[1]"]


def test_carried_step_is_legible_in_a_later_beat(carry_in_video):
    # carry_in_video registers the _Highlight consumer; the producer here is a
    # SolutionStep beat, rebuilt by SolutionStep's own artifact builder.
    video = VideoSpec(video_id="v", beats=(
        BeatSpec(id="b01", narration="placeholder narration", component=NAME,
                 params=EXAMPLES[0], registers="accel"),
        BeatSpec(id="b02", narration="placeholder narration", component="_Highlight",
                 params={"target_id": "accel"}, carry_in=["accel"]),
    ))
    recipes = resolve_carry_in(video)["b02"]
    report = validate_beat(video.beats[1], duration=4.0, recipes=recipes)
    assert report.ok, f"\n{report}"
    probe = LayoutProbe(beat_component(video.beats[1], recipes), duration=4.0)
    probe.construct()
    target = carried(probe, "accel")
    assert region_rect(Region.STAGE).contains(bbox(target))
    assert smallest_font_size(target) >= MIN_FONT_SIZE
