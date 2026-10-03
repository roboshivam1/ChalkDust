"""SplitCompare behaviour the registry-wide layout tests do not pin.

test_layout.py already proves examples() validate clean and stress() fits or
refuses cleanly. This file pins timing against the audio budget, the schema's
refusals, and the typed LaTeX failure path.

Needs LaTeX on PATH (the maths sides compile through MathTex).
"""

from __future__ import annotations

import pytest
from manim import tempconfig
from pydantic import ValidationError

from chalkdust.core.models import BeatSpec, Region
from chalkdust.scenes.base import ChalkdustScene
from chalkdust.scenes.components.split_compare import SplitCompare
from chalkdust.validate.geometric import LayoutProbe, validate_beat

FPS = 15
# SCENE_SPEC.md §2: a beat is 8-25 s of narration. Budgets either side of that
# cover narration far shorter and far longer than the animation wants.
NATURAL = 8.0
BUDGETS = [0.5, NATURAL * 0.5, NATURAL * 3, 60.0]


def _spec_example() -> dict:
    return SplitCompare.examples()[0]


def _elapsed(params: dict, duration: float) -> float:
    """Render for real (frames are counted, not skipped) without writing a file."""
    with tempconfig({"pixel_width": 854, "pixel_height": 480, "frame_rate": FPS,
                     "write_to_movie": False, "disable_caching": True,
                     "verbosity": "WARNING", "progress_bar": "none"}):
        scene = ChalkdustScene(SplitCompare(params), duration=duration)
        scene.render()
        return scene.renderer.time


class TestTiming:
    @pytest.mark.parametrize("budget", BUDGETS)
    @pytest.mark.parametrize("which", [0, 1], ids=["with_verdict", "no_verdict"])
    def test_consumes_budget_within_a_frame(self, budget, which):
        elapsed = _elapsed(SplitCompare.examples()[which], budget)
        assert abs(elapsed - budget) <= 1 / FPS, (
            f"budget {budget}s rendered as {elapsed}s"
        )

    def test_min_seconds_counts_the_verdict_step(self):
        with_verdict = SplitCompare(_spec_example())
        without = SplitCompare({k: v for k, v in _spec_example().items()
                                if k not in ("verdict", "emphasis")})
        assert with_verdict.min_seconds() > without.min_seconds() > 0


class TestSchema:
    @pytest.mark.parametrize("bad", [
        {"left": {"title": ""}, "right": {"title": "b"}},
        {"left": {"title": "   "}, "right": {"title": "b"}},
        {"left": {"title": "a", "body": ""}, "right": {"title": "b"}},
        {"left": {"title": "a", "math": " "}, "right": {"title": "b"}},
        {"left": {"title": "a"}, "right": {"title": "b"}, "verdict": ""},
        {"left": {"title": "a"}},
    ], ids=["empty_title", "blank_title", "empty_body", "blank_math",
            "empty_verdict", "missing_right"])
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
        probe = LayoutProbe(SplitCompare(SplitCompare.stress()[3]), duration=NATURAL)
        probe.construct()
        sides = {m._chalk_label: m for m in probe.mobjects
                 if getattr(m, "_chalk_label", "").endswith(" side")}
        left_title = sides["left side"][1][0]
        right_title = sides["right side"][1][0]
        assert left_title._chalk_font_size == pytest.approx(right_title._chalk_font_size)
        assert left_title.get_top()[1] == pytest.approx(right_title.get_top()[1])

    def test_overloaded_sides_refuse_as_overflow(self):
        report = validate_beat(BeatSpec(id="b01", narration="n", component="SplitCompare",
                                        params=SplitCompare.stress()[0]))
        assert report.kinds() == {"overflow"}, f"\n{report}"


class TestLatex:
    def test_latex_strings_lists_every_maths_side(self):
        assert SplitCompare(SplitCompare.examples()[1]).latex_strings() == ["O(1)", "O(n)"]
        assert SplitCompare(_spec_example()).latex_strings() == []

    @pytest.mark.parametrize("tex", [r"\notacommand{x}", r"x^"],
                             ids=["undefined_command", "dangling_superscript"])
    def test_invalid_latex_is_a_typed_finding_not_a_crash(self, tex):
        params = {"left": {"title": "Broken", "math": tex},
                  "right": {"title": "Fine", "math": r"x^2"}}
        report = validate_beat(BeatSpec(id="b01", narration="n",
                                        component="SplitCompare", params=params))
        assert report.kinds() == {"latex"}, f"\n{report}"
