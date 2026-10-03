"""SplitCompare behaviour the registry-wide layout tests do not pin.

test_layout.py already proves examples() validate clean and stress() fits or
refuses cleanly. This file pins timing against the audio budget, the schema's
refusals, which refusal each hostile stress case gives, and the carry-in
artifact (SCENE_SPEC.md §6).

Needs LaTeX on PATH (the maths sides compile through MathTex).
"""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import asdict

import numpy as np
import pytest
from manim import tempconfig
from pydantic import ValidationError

from chalkdust.continuity import ArtifactRecipe, build_artifact, resolve_carry_in
from chalkdust.core.models import BeatSpec, Quality, Region, VideoSpec
from chalkdust.render.worker import TIERS, long_path
from chalkdust.scenes.base import ChalkdustScene
from chalkdust.scenes.components.split_compare import (
    BODY_WRAP,
    HOLD_S,
    REVEAL_S,
    VERDICT_S,
    SplitCompare,
    _wrap_balanced,
)
from chalkdust.scenes.regions import bbox
from chalkdust.scenes.theme import DEFAULT
from chalkdust.validate.geometric import LayoutProbe, validate_beat

NAME = "SplitCompare"
EXAMPLES = SplitCompare.examples()
STRESS = SplitCompare.stress()
DRAFT = TIERS[Quality.DRAFT]
NATURAL = 8.0  # a typical beat's narration, seconds


def _spec(params: dict, bid: str = "b01", **kw) -> BeatSpec:
    return BeatSpec(id=bid, narration="placeholder narration",
                    component=NAME, params=params, **kw)


def _spec_example() -> dict:
    return EXAMPLES[0]


class _Clock(LayoutProbe):
    """A probe that also counts the frames every play() and wait() asks for,
    without encoding any. A play() with no explicit run_time would raise
    here, so a forgotten run_time cannot pass as a default."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.frames: list[float] = []

    def play(self, *animations, run_time: float, **kwargs) -> None:  # type: ignore[override]
        self.frames.append(run_time * self.fps)
        super().play(*animations, **kwargs)

    def wait(self, duration: float, *args, **kwargs) -> None:  # type: ignore[override]
        self.frames.append(duration * self.fps)


def _clock(params: dict, duration: float) -> _Clock:
    with tempconfig({"frame_rate": DRAFT.frame_rate}):
        clock = _Clock(SplitCompare(params), duration=duration)
    clock.construct()
    return clock


class TestTiming:
    # Narration far shorter (0.5x) and far longer (3x) than the animation
    # wants: either way the beat lasts exactly its audio, in whole frames.
    @pytest.mark.parametrize("factor", [0.5, 3.0], ids=["short", "long"])
    @pytest.mark.parametrize("which", range(len(EXAMPLES)),
                             ids=[f"ex{i}" for i in range(len(EXAMPLES))])
    def test_clocked_frames_equal_beat_frames(self, which, factor):
        params = EXAMPLES[which]
        # +0.0123 s so the audio is never a whole number of frames.
        budget = SplitCompare(params).min_seconds() * factor + 0.0123
        clock = _clock(params, budget)
        assert all(f == pytest.approx(round(f), abs=1e-6) for f in clock.frames), clock.frames
        assert round(sum(clock.frames)) == clock.beat_frames == math.ceil(budget * DRAFT.frame_rate)
        assert len(clock.frames) == len(SplitCompare(params)._segments())

    @pytest.mark.parametrize("factor", [0.5, 3.0], ids=["short", "long"])
    def test_draft_render_is_exactly_the_beat(self, factor, tmp_path):
        # A real 480p15 clip, frames counted by ffprobe: ceil(audio * fps).
        params = _spec_example()
        budget = SplitCompare(params).min_seconds() * factor + 0.0123
        media = long_path(tmp_path)
        with tempconfig({**asdict(DRAFT), "media_dir": str(media),
                         "disable_caching": True, "progress_bar": "none",
                         "verbosity": "WARNING", "output_file": "split_compare"}):
            scene = ChalkdustScene(SplitCompare(params), duration=budget)
            scene.render()
            movie = scene.renderer.file_writer.movie_file_path
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
             "-show_entries", "stream=nb_read_frames", "-of", "json", str(movie)],
            capture_output=True, text=True, check=True).stdout
        frames = int(json.loads(out)["streams"][0]["nb_read_frames"])
        assert frames == math.ceil(budget * DRAFT.frame_rate)

    def test_min_seconds_is_sum_of_segment_minimums(self):
        with_verdict = SplitCompare(_spec_example())
        without = SplitCompare({k: v for k, v in _spec_example().items()
                                if k not in ("verdict", "emphasis")})
        assert with_verdict.min_seconds() == pytest.approx(2 * REVEAL_S + VERDICT_S + HOLD_S)
        assert without.min_seconds() == pytest.approx(2 * REVEAL_S + HOLD_S)

    def test_every_segment_gets_its_minimum_at_min_seconds(self):
        # The weights are the minimums, so narration exactly min_seconds() long
        # gives each segment (to the frame) what it needs, the verdict included.
        params = _spec_example()
        clock = _clock(params, SplitCompare(params).min_seconds())
        mins = SplitCompare(params)._segments()
        assert all(f >= m * DRAFT.frame_rate - 1 for f, m in zip(clock.frames, mins))


class TestSchema:
    @pytest.mark.parametrize("bad", [
        {"left": {"title": ""}, "right": {"title": "b"}},
        {"left": {"title": "   "}, "right": {"title": "b"}},
        {"left": {"title": "a", "body": ""}, "right": {"title": "b"}},
        {"left": {"title": "a", "math": " "}, "right": {"title": "b"}},
        {"left": {"title": "a"}, "right": {"title": "b"}, "verdict": ""},
        {"left": {"title": "a"}},
        {},
    ], ids=["empty_title", "blank_title", "empty_body", "blank_math",
            "empty_verdict", "missing_right", "empty_params"])
    def test_rejects_empty_or_missing_content(self, bad):
        with pytest.raises(ValidationError):
            SplitCompare(bad)

    def test_rejects_unknown_key_inside_a_side(self):
        # The nested model must be as strict as the outer one.
        with pytest.raises(ValidationError):
            SplitCompare({"left": {"title": "a", "colour": "red"},
                          "right": {"title": "b"}})

    def test_rejects_verdict_emphasis_without_a_verdict(self):
        with pytest.raises(ValidationError):
            SplitCompare({"left": {"title": "a"}, "right": {"title": "b"},
                          "emphasis": "verdict"})

    def test_minimal_sides_are_allowed(self):
        assert SplitCompare({"left": {"title": "a"}, "right": {"title": "b"}})


class TestRegions:
    def test_lower_third_only_with_a_verdict(self):
        halves = {Region.STAGE_LEFT, Region.STAGE_RIGHT}
        assert SplitCompare(_spec_example()).regions() == halves | {Region.LOWER_THIRD}
        assert SplitCompare({"left": {"title": "a"},
                             "right": {"title": "b"}}).regions() == halves


class TestLayout:
    def test_sides_share_one_scale_and_title_line(self):
        # Lopsided content: the sparse side must not render in bigger type,
        # and the two titles must sit on one line.
        probe = LayoutProbe(SplitCompare(STRESS[3]), duration=NATURAL)
        probe.construct()
        sides = {m._chalk_label: m for m in probe.mobjects
                 if getattr(m, "_chalk_label", "").endswith(" side")}
        left_title = sides["left side"][1][0]
        right_title = sides["right side"][1][0]
        assert left_title._chalk_font_size == pytest.approx(right_title._chalk_font_size)
        assert left_title.get_top()[1] == pytest.approx(right_title.get_top()[1])

    def test_wrapped_lines_are_balanced_not_widowed(self):
        # Plain wrap() at BODY_WRAP leaves "head" alone on the second line;
        # narrowing without adding a line evens them out.
        assert _wrap_balanced("Walk node by node from the head", BODY_WRAP) == \
            "Walk node by node\nfrom the head"
        assert _wrap_balanced("hash → 4", BODY_WRAP) == "hash → 4"

    def test_overloaded_sides_refuse_as_overflow(self):
        report = validate_beat(_spec(STRESS[0]))
        assert report.kinds() == {"overflow"}, f"\n{report}"

    @pytest.mark.parametrize("which", [1, 2], ids=["unwrappable", "minimal"])
    def test_unwrappable_and_minimal_content_fits(self, which):
        report = validate_beat(_spec(STRESS[which]))
        assert report.ok, f"\n{report}"


class TestLatex:
    def test_latex_strings_lists_every_maths_side(self):
        assert SplitCompare(EXAMPLES[1]).latex_strings() == ["O(1)", "O(n)"]
        assert SplitCompare(_spec_example()).latex_strings() == []

    @pytest.mark.parametrize("which,side", [(4, "left"), (5, "right"), (6, "left")],
                             ids=["undefined_command", "dangling_superscript",
                                  "renders_nothing"])
    def test_invalid_latex_stress_refuses_as_invalid_latex(self, which, side):
        # test_layout accepts any clean refusal; this pins WHICH one, and
        # that the message names the side whose maths must be regenerated.
        report = validate_beat(_spec(STRESS[which]))
        assert report.kinds() == {"invalid_latex"}, f"\n{report}"
        assert f"{side}.math" in report.findings[0].message


class TestContinuity:
    """SplitCompare as producer and consumer of carried artifacts (§6)."""

    def _recipe(self, params: dict) -> ArtifactRecipe:
        return ArtifactRecipe(name="comparison", producer=NAME, params=params)

    def test_artifact_is_the_settled_frame(self):
        # The rebuilt artifact is exactly what the producing beat ended on:
        # same parts, same boxes, same tracked font sizes.
        params = _spec_example()
        probe = LayoutProbe(SplitCompare(params), duration=NATURAL)
        probe.construct()
        settled = [m for m in probe.mobjects
                   if getattr(m, "_chalk_label", None) in ("left side", "right side", "verdict")]
        # The scene's theme, fonts resolved, as CarryIn hands the builder.
        artifact = build_artifact(self._recipe(params), probe.theme)
        assert len(artifact.submobjects) == len(settled) == 3
        for got, want in zip(artifact.submobjects, settled):
            g, w = bbox(got), bbox(want)
            assert (g.x, g.y, g.width, g.height) == pytest.approx(
                (w.x, w.y, w.width, w.height), abs=1e-6)
            assert [getattr(m, "_chalk_font_size", None) for m in got.get_family()] == \
                pytest.approx([getattr(m, "_chalk_font_size", None) for m in want.get_family()])

    def test_artifact_rebuild_is_deterministic(self):
        recipe = self._recipe(EXAMPLES[1])
        a, b = build_artifact(recipe, DEFAULT), build_artifact(recipe, DEFAULT)
        pa = [m.points for m in a.family_members_with_points()]
        pb = [m.points for m in b.family_members_with_points()]
        assert len(pa) == len(pb) > 0
        assert all(np.array_equal(x, y) for x, y in zip(pa, pb))

    def test_carried_comparison_validates_in_a_later_beat(self):
        # b01 registers the comparison; b02 (another SplitCompare, as in the
        # §6 example) carries it in, dimmed in STAGE beneath its own cards.
        video = VideoSpec(video_id="v", beats=(
            _spec(EXAMPLES[1], "b01", registers="comparison"),
            _spec(_spec_example(), "b02", carry_in=["comparison"]),
        ))
        recipes = resolve_carry_in(video)["b02"]
        assert recipes == (self._recipe(EXAMPLES[1]),)
        report = validate_beat(video.beats[1], duration=NATURAL, recipes=recipes)
        assert report.ok, f"\n{report}"
