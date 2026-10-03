"""ProblemStatement: timing, schema limits, the LaTeX error path, inline layout.

Layout of examples() and stress() is covered by test_layout.py walking the
registry; these pin what that walk cannot see. LaTeX must be installed --
these compile real maths and are never skipped without it.
"""

from __future__ import annotations

import pytest
from manim import config, tempconfig
from pydantic import ValidationError

from chalkdust.core.models import BeatSpec
from chalkdust.scenes.base import ChalkdustScene
from chalkdust.scenes.components.problem_statement import (
    STRUT,
    WORD_SPACE,
    LatexError,
    ProblemStatement,
)
from chalkdust.scenes.regions import bbox
from chalkdust.scenes.theme import body_cap_height
from chalkdust.validate.geometric import LayoutProbe, validate_beat

INCLINE = ProblemStatement.examples()[0]
MINIMAL = {"text": "x", "find": "y"}
FRAME = 1 / 60  # one frame at the final tier's 60 fps (D-006)


def _elapsed(params: dict, budget: float) -> float:
    """Build through the real scene with Manim's own clock running.

    skip_animations makes the renderer add each play's duration to its clock
    without drawing a frame, so this measures exactly what a render would
    spend, minus the encoding.
    """
    scene = ChalkdustScene(ProblemStatement(params), duration=budget,
                           skip_animations=True)
    scene.setup()
    scene.construct()
    return scene.renderer.time


def _probe(params: dict) -> LayoutProbe:
    probe = LayoutProbe(ProblemStatement(params), duration=8.0)
    probe.construct()
    return probe


def _blocks(probe: LayoutProbe) -> dict:
    return {m._chalk_label: m for m in probe.mobjects}


class TestTiming:
    """Animation consumes the beat's audio budget exactly (D-002)."""

    @pytest.mark.parametrize("factor", [0.5, 3.0])
    def test_consumes_budget_relative_to_minimum(self, factor):
        budget = ProblemStatement(INCLINE).min_seconds() * factor
        assert _elapsed(INCLINE, budget) == pytest.approx(budget, abs=FRAME)

    @pytest.mark.parametrize("budget", [1.0, 40.0])
    def test_consumes_far_short_and_far_long_narration(self, budget):
        assert _elapsed(INCLINE, budget) == pytest.approx(budget, abs=FRAME)

    def test_minimal_input_consumes_budget(self):
        assert _elapsed(MINIMAL, 6.0) == pytest.approx(6.0, abs=FRAME)

    def test_real_render_lasts_an_off_frame_budget(self, tmp_path):
        """Drawn frames, not summed run times: Manim rounds every play up to
        whole frames, which skip_animations cannot see. 7.3127 s lands on no
        frame boundary at 15 fps; unsnapped, the eight steps overran by six
        frames."""
        with tempconfig({"quality": "low_quality", "media_dir": str(tmp_path),
                         "write_to_movie": False, "disable_caching": True,
                         "progress_bar": "none", "verbosity": "WARNING"}):
            scene = ChalkdustScene(ProblemStatement(INCLINE), duration=7.3127)
            scene.render()
            frame = 1 / config.frame_rate
        assert scene.renderer.time == pytest.approx(7.3127, abs=frame)

    def test_min_seconds_grows_with_each_given(self):
        none = ProblemStatement(MINIMAL).min_seconds()
        four = ProblemStatement(INCLINE).min_seconds()
        assert 0 < none < four


class TestSchema:
    """Inputs the param model must refuse before anything compiles."""

    @pytest.mark.parametrize("params", [
        {"text": "   ", "find": "y"},
        {"text": "x", "find": ""},
        {"text": "x"},
        {"text": "x", "given": ["  "], "find": "y"},
        {"text": "x", "given": ["$a$"] * 7, "find": "y"},
        {"text": "costs $5 more", "find": "y"},
        {"text": "x", "find": "the value $ $"},
        {"text": "x", "find": "y", "colour": "red"},
    ], ids=["blank-text", "blank-find", "no-find", "blank-given", "too-many-givens",
            "unbalanced-dollar", "empty-maths", "unknown-field"])
    def test_rejects(self, params):
        with pytest.raises(ValidationError):
            ProblemStatement(params)

    def test_givens_are_optional(self):
        assert ProblemStatement(MINIMAL).params.given == []


class TestLatex:
    """Bad maths is a typed finding the repair loop can act on, never a
    crash."""

    def test_latex_strings_are_every_maths_chunk_in_order(self):
        params = ProblemStatement.examples()[2]
        assert ProblemStatement(params).latex_strings() == [
            STRUT + s for s in
            ["x", "t", "x(t) = t^3 - 6t^2 + 9t", "x(t) = t^3 - 6t^2 + 9t", r"t \ge 0"]
        ]

    def test_latex_strings_empty_without_maths(self):
        assert ProblemStatement(MINIMAL).latex_strings() == []

    def test_invalid_latex_is_a_typed_finding(self):
        params = {"text": r"The force is $\notacommand{F}$.", "find": "y"}
        report = validate_beat(BeatSpec(id="b01", narration="placeholder",
                                        component="ProblemStatement", params=params))
        assert report.kinds() == {"latex"}, f"\n{report}"

    def test_maths_that_draws_nothing_raises_latex_error(self):
        probe = LayoutProbe(ProblemStatement({"text": "a ${}$ b", "find": "y"}),
                            duration=5.0)
        with pytest.raises(LatexError) as exc:
            probe.construct()
        assert exc.value.kind == "latex"


class TestInlineLayout:
    def test_maths_shares_the_prose_baseline(self):
        # No descenders anywhere, so every piece's bottom is its baseline.
        probe = _probe({"text": "mass $m$ rests", "find": "y"})
        [line] = _blocks(probe)["statement"].submobjects
        prose, maths, more = line.submobjects
        tol = 0.05 * body_cap_height(probe.theme)
        assert maths.get_bottom()[1] == pytest.approx(prose.get_bottom()[1], abs=tol)
        assert more.get_bottom()[1] == pytest.approx(prose.get_bottom()[1], abs=tol)

    def test_source_spacing_around_maths_is_kept(self):
        # "$x$-axis": no space in the source, so none on screen.
        probe = _probe({"text": "mass $m$ rests on the $x$-axis", "find": "y"})
        [line] = _blocks(probe)["statement"].submobjects
        _, m, rests, x, axis = line.submobjects
        space = WORD_SPACE * body_cap_height(probe.theme)
        assert rests.get_left()[0] - m.get_right()[0] == pytest.approx(space, rel=0.05)
        assert axis.get_left()[0] - x.get_right()[0] < space / 2

    def test_long_statement_wraps_into_aligned_lines(self):
        lines = _blocks(_probe(INCLINE))["statement"].submobjects
        assert len(lines) > 1
        lefts = {round(line.get_left()[0], 6) for line in lines}
        assert len(lefts) == 1
        for a, b in zip(lines, lines[1:]):
            assert a.get_bottom()[1] > b.get_top()[1]

    def test_given_items_sit_at_one_rhythm(self):
        # The middle item is taller (a superscript), but items stack by
        # baseline, not by box, so the pitch stays constant. No descenders,
        # so each item's bottom is its baseline.
        probe = _probe({"text": "x", "given": ["$a = 1$", "$b = 2^{2}$", "$c = 3$"],
                        "find": "y"})
        items = _blocks(probe)["given"].submobjects[1:]
        bottoms = [item.get_bottom()[1] for item in items]
        pitches = [a - b for a, b in zip(bottoms, bottoms[1:])]
        assert pitches[0] == pytest.approx(pitches[1],
                                           abs=0.02 * body_cap_height(probe.theme))

    def test_find_sits_beside_givens_that_fit_their_column(self):
        blocks = _blocks(_probe(INCLINE))
        given, find = bbox(blocks["given"]), bbox(blocks["find"])
        assert find.left > given.right
        assert find.top == pytest.approx(given.top, abs=1e-6)

    def test_find_drops_below_givens_that_overrun_their_column(self):
        wide = ProblemStatement.stress()[2]
        blocks = _blocks(_probe(wide))
        assert bbox(blocks["find"]).top < bbox(blocks["given"]).bottom

    def test_no_givens_means_no_given_column(self):
        assert set(_blocks(_probe(MINIMAL))) == {"statement", "find"}
