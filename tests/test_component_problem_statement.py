"""ProblemStatement: timing, schema limits, the LaTeX refusals, inline layout,
and the carry-in artifact.

Layout of examples() and stress() -- including stress()'s invalid LaTeX, which
must refuse as invalid_latex -- is covered by test_layout.py walking the
registry; these pin what that walk cannot see. LaTeX must be installed --
these compile real maths and are never skipped without it.
"""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import asdict

import numpy as np
import pytest
from manim import DL, tempconfig
from pydantic import ValidationError

from chalkdust.continuity import (
    DIM_DARKNESS,
    ArtifactRecipe,
    beat_component,
    build_artifact,
    carried,
    resolve_carry_in,
)
from chalkdust.core.models import BeatSpec, Quality, Region, VideoSpec
from chalkdust.render.worker import TIERS, long_path
from chalkdust.scenes.base import ChalkdustScene
from chalkdust.scenes.components.problem_statement import (
    ITALIC_GAP,
    PROSE_STRUT,
    STRUT,
    WORD_SPACE,
    ProblemStatement,
    _Flow,
    _baselined,
    _char_width,
    _prose_strut,
    _strut_outline,
)
from chalkdust.scenes.regions import (
    INVALID_LATEX,
    UNRENDERABLE_TEXT,
    LayoutError,
    bbox,
    region_rect,
)
from chalkdust.scenes.theme import DEFAULT, body_cap_height, body_text, resolve_fonts
from chalkdust.validate.geometric import LayoutProbe, validate_beat

INCLINE = ProblemStatement.examples()[0]
MINIMAL = {"text": "x", "find": "y"}
DRAFT = TIERS[Quality.DRAFT]


class _FrameClock(ChalkdustScene):
    """The real scene, nothing encoded, recording the frames Manim would draw
    for every play() and wait(): the run time the scene hands Manim, turned
    into frames the way the renderer turns it. A play() without a budget() run time
    shows up as a frame-count mismatch, not a silent pass."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.frames: list[int] = []

    def compile_animation_data(self, *args, **kwargs):
        # Once per play() and wait(), wherever Manim then asks for the run time
        # again; `duration` is the run time the scene's get_run_time() gave.
        out = super().compile_animation_data(*args, **kwargs)
        self.frames.append(int(self.duration * self.fps))
        return out


def _clocked(params: dict, budget: float) -> _FrameClock:
    """Build at the draft tier's frame rate with Manim's own clock running."""
    with tempconfig({"frame_rate": DRAFT.frame_rate}):
        scene = _FrameClock(ProblemStatement(params), duration=budget,
                            skip_animations=True)
    scene.setup()
    scene.construct()
    return scene


def _probe(params: dict) -> LayoutProbe:
    probe = LayoutProbe(ProblemStatement(params), duration=8.0)
    probe.construct()
    return probe


def _blocks(probe: LayoutProbe) -> dict:
    return {m._chalk_label: m for m in probe.mobjects}


def _spec(params: dict) -> BeatSpec:
    return BeatSpec(id="b01", narration="placeholder narration",
                    component="ProblemStatement", params=params)


class TestTiming:
    """Animation consumes the beat's audio exactly: ceil(audio * fps) frames
    (D-002), however far the narration is from what the animation wants."""

    @pytest.mark.parametrize("factor", [0.5, 3.0], ids=["short", "long"])
    @pytest.mark.parametrize("index", range(len(ProblemStatement.examples())),
                             ids=lambda i: f"ex{i}")
    def test_clocked_frames_equal_beat_frames(self, index, factor):
        params = ProblemStatement.examples()[index]
        budget = ProblemStatement(params).min_seconds() * factor
        scene = _clocked(params, budget)
        assert sum(scene.frames) == scene.beat_frames == \
            math.ceil(round(budget * DRAFT.frame_rate, 6))

    @pytest.mark.parametrize("budget", [1.0, 40.0], ids=["far-short", "far-long"])
    def test_far_short_and_far_long_narration(self, budget):
        scene = _clocked(INCLINE, budget)
        assert sum(scene.frames) == scene.beat_frames

    def test_minimal_input_consumes_budget(self):
        scene = _clocked(MINIMAL, 6.0)
        assert sum(scene.frames) == scene.beat_frames == 90

    def test_draft_render_is_exactly_the_beat(self, tmp_path):
        """Encoded frames, counted by ffprobe -- not summed run times. 7.3127 s
        lands on no frame boundary at 15 fps: ceil(109.69) = 110 frames. Before
        budget() handed out whole frames, unsnapped, the eight plays here
        overran it by six."""
        with tempconfig({**asdict(DRAFT), "media_dir": str(long_path(tmp_path)),
                         "disable_caching": True, "progress_bar": "none",
                         "verbosity": "WARNING", "output_file": "frames"}):
            scene = ChalkdustScene(ProblemStatement(INCLINE), duration=7.3127)
            scene.render()
            movie = scene.renderer.file_writer.movie_file_path
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
             "-show_entries", "stream=nb_read_frames", "-of", "json", str(movie)],
            capture_output=True, text=True, check=True).stdout
        frames = int(json.loads(out)["streams"][0]["nb_read_frames"])
        assert frames == math.ceil(7.3127 * DRAFT.frame_rate) == 110

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
        {"text": "a\x00b", "find": "y"},
    ], ids=["blank-text", "blank-find", "no-find", "blank-given", "too-many-givens",
            "unbalanced-dollar", "empty-maths", "unknown-field", "control-character"])
    def test_rejects(self, params):
        with pytest.raises(ValidationError):
            ProblemStatement(params)

    @pytest.mark.parametrize("params", [
        {"text": "\u200b", "find": "y"},
        {"text": "x", "find": "\u2060\u200d\u00ad"},
        {"text": "x", "given": ["\u00a0\u200b\ufeff"], "find": "y"},
    ], ids=["zero-width-space", "format-characters", "nbsp-and-bom"])
    def test_rejects_prose_with_no_visible_character(self, params):
        # str.strip() keeps format characters; such a field drew nothing and
        # was misreported as an overlap with whatever sat at the origin.
        with pytest.raises(ValidationError, match="no visible character"):
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
        # Refused through theme.math, naming the field to regenerate.
        params = {"text": "x", "given": ["$a = 1$", r"$\notacommand{F}$"], "find": "y"}
        report = validate_beat(_spec(params))
        assert report.kinds() == {INVALID_LATEX}, f"\n{report}"
        assert "given[1]" in report.findings[0].message

    def test_maths_that_draws_nothing_is_invalid_latex(self):
        probe = LayoutProbe(ProblemStatement({"text": "a ${}$ b", "find": "y"}),
                            duration=5.0)
        with pytest.raises(LayoutError) as exc:
            probe.construct()
        assert exc.value.kind == INVALID_LATEX
        assert "renders nothing" in str(exc.value)

    @pytest.mark.parametrize("chunk", [r"\frac{1}{2", r"a}{b", r"x^{2"],
                             ids=["unclosed-frac", "closed-first", "unclosed-superscript"])
    def test_unbalanced_braces_are_invalid_latex(self, chunk):
        # Manim's wrapper would close an open brace for us, so LaTeX alone
        # accepts "\frac{1}{2"; it must be refused all the same.
        report = validate_beat(_spec({"text": f"the ratio ${chunk}$", "find": "y"}))
        assert report.kinds() == {INVALID_LATEX}, f"\n{report}"
        assert "unbalanced braces" in report.findings[0].message

    def test_escaped_braces_are_not_counted(self):
        assert validate_beat(_spec({"text": r"the set $\{1, 2\}$", "find": "y"})).ok


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

    def test_unspaced_prose_after_maths_keeps_an_italic_gap(self):
        # "$\mu$." -- the period must not touch the mu's tail (seen in a draft
        # render: they joined), yet stay far tighter than a word space.
        probe = _probe({"text": r"friction is $\mu$.", "find": "y"})
        [line] = _blocks(probe)["statement"].submobjects
        _, mu, period = line.submobjects
        gap = period.get_left()[0] - mu.get_right()[0]
        assert gap == pytest.approx(ITALIC_GAP * body_cap_height(probe.theme), rel=0.05)

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

    @pytest.mark.parametrize("field,params", [
        ("text", {"text": "\U0001F600" * 10, "find": "y"}),
        ("given[0]", {"text": "x", "given": ["\U0001F600"], "find": "y"}),
        ("find", {"text": "x", "find": "\U0001F600 \U0001F600"}),
    ], ids=["text", "given", "find"])
    def test_prose_the_font_cannot_draw_is_unrenderable_text(self, field, params):
        # Pango drops glyphs the font lacks; a field of nothing else drew no
        # ink and was once reported as an "overlap" with its neighbour. The
        # theme's glyph guard refuses it first, naming the field and quoting
        # the spec's text -- not the strut-prefixed string typeset for it.
        report = validate_beat(_spec(params))
        assert report.kinds() == {UNRENDERABLE_TEXT}, f"\n{report}"
        message = report.findings[0].message
        assert f"ProblemStatement {field} " in message
        assert "'H" not in message

    def test_lines_of_zero_width_characters_are_dropped(self):
        # Zero-width spaces long enough to hard-break over several lines ahead
        # of real text: those lines draw nothing, so they are not stacked or
        # revealed, and the field still validates.
        params = {"text": "\u200b" * 200 + " then $x$ rests", "find": "y"}
        lines = _blocks(_probe(params))["statement"].submobjects
        assert len(lines) == 1 and lines[0].family_members_with_points()
        assert validate_beat(_spec(params)).ok

    def test_private_use_characters_are_refused_at_rung_one(self):
        # Symbol-font Greek pasted from a PDF (U+F071 theta, U+F06D mu): Pango
        # draws each as a missing-glyph box. BeatSpec refuses it before any
        # component is built.
        with pytest.raises(ValidationError, match="private-use"):
            _spec({"text": "inclined at \uf071 with friction \uf06d", "find": "a"})

    def test_private_use_characters_past_rung_one_are_unrenderable_text(self):
        # Built directly, the theme's glyph guard refuses the run. Before it,
        # Manim listed a path of the box ahead of the strut, so the strut H
        # stayed on screen and validated clean (verify-rb2).
        probe = LayoutProbe(ProblemStatement(
            {"text": "inclined at \uf071 to the horizontal", "find": "a"}), duration=5.0)
        with pytest.raises(LayoutError) as exc:
            probe.construct()
        assert exc.value.kind == UNRENDERABLE_TEXT
        assert "ProblemStatement text 'inclined at" in str(exc.value)

    def test_strut_is_found_by_shape_not_submobject_order(self):
        # Pango's path order is not the string's. Reversed, the strut is still
        # the leftmost glyph shaped like the H built alone; the run's own H
        # ("Hence") is kept, and the run sits on y=0.
        theme = resolve_fonts(DEFAULT)
        t = body_text(PROSE_STRUT + "Hence", theme)
        t.submobjects.reverse()
        strut = _prose_strut(t, theme)
        assert strut.get_left()[0] == min(g.get_left()[0]
                                          for g in t.family_members_with_points())
        run = _baselined(t, strut).family_members_with_points()
        assert len(run) == 5 and strut not in run
        ref = _strut_outline(theme)
        assert sum(g.points.shape == ref.shape
                   and np.allclose(g.points - g.get_corner(DL), ref, atol=1e-9)
                   for g in run) == 1
        assert min(g.get_bottom()[1] for g in run) == pytest.approx(
            0.0, abs=0.05 * body_cap_height(theme))

    def test_leading_combining_mark_is_not_lost_with_the_strut(self):
        # "H" + U+0302 shapes as one glyph; without the strut's space the mark
        # composed onto the H and was removed with it.
        [line] = _blocks(_probe({"text": "\u0302abc", "find": "y"}))["statement"].submobjects
        assert len(line.family_members_with_points()) == 4


class TestVolume:
    """Line breaking costs about one typeset per line, however far the text's
    real width is from the estimate, and a field too long for the stage is
    refused once that is certain -- not after typesetting all of it."""

    @staticmethod
    def _count_lines(monkeypatch) -> list[int]:
        calls = [0]
        build = _Flow._line

        def counted(self, idx):
            calls[0] += 1
            return build(self, idx)

        monkeypatch.setattr(_Flow, "_line", counted)
        return calls

    def test_wide_prose_is_not_rebuilt_line_after_line(self, monkeypatch):
        # Capitals are wider than the pangram average the planner estimates
        # with. Correcting a fixed plan by handing each overlong line's last
        # token down cascaded: 520 builds for 20 lines of "WWWW", and
        # minutes for a few hundred words (verify-w4-1).
        calls = self._count_lines(monkeypatch)
        theme = resolve_fonts(DEFAULT)
        width = 64 * _char_width(theme)
        lines = _Flow("WWWW " * 100, theme, width, "text").lines()
        assert all(line.width <= width for line in lines)
        assert len(lines) > 10
        assert calls[0] <= 2 * len(lines)

    def test_huge_statement_refuses_as_overflow_early(self, monkeypatch):
        # 100k characters: refused once its lines pass what STAGE could hold
        # at the legibility floor, after a dozen or so lines -- this took
        # over half an hour before, with no build timeout in the pipeline.
        calls = self._count_lines(monkeypatch)
        report = validate_beat(_spec({"text": "word " * 20000, "find": "y"}))
        assert report.kinds() == {"overflow"}, f"\n{report}"
        assert calls[0] < 40


class TestCarryIn:
    """A later beat can carry the stated problem in, rebuilt from these params
    (SCENE_SPEC.md §6) -- the usual JEE shape: the problem stays on screen,
    dimmed, while the solution is worked."""

    def _video(self, consumer: str = "BulletReveal") -> VideoSpec:
        # To build, b02 must be a consumer (the hold_consumer fixture's): a
        # beat that carries an artifact in without acting on it must leave it
        # a free STAGE region (D-G4c-1).
        return VideoSpec(video_id="v", beats=(
            BeatSpec(id="b01", narration="placeholder narration",
                     component="ProblemStatement", params=INCLINE,
                     registers="problem"),
            BeatSpec(id="b02", narration="placeholder narration",
                     component=consumer,
                     params=({"items": ["resolve forces"]} if consumer == "BulletReveal"
                             else {"target_id": "problem"}),
                     carry_in=["problem"]),
        ))

    def test_recipe_resolves_to_this_component(self):
        [recipe] = resolve_carry_in(self._video())["b02"]
        assert (recipe.producer, recipe.params) == ("ProblemStatement", INCLINE)

    def test_artifact_is_the_settled_problem_at_full_strength(self):
        art = build_artifact(ArtifactRecipe(name="problem", producer="ProblemStatement",
                                            params=INCLINE), resolve_fonts(DEFAULT))
        assert [m._chalk_label for m in art.submobjects] == ["statement", "given", "find"]
        assert {m.get_fill_opacity() for m in art.family_members_with_points()} == {1.0}

    def test_rebuild_is_deterministic(self):
        recipe = ArtifactRecipe(name="problem", producer="ProblemStatement", params=INCLINE)
        theme = resolve_fonts(DEFAULT)
        pa, pb = ([m.points for m in build_artifact(recipe, theme).family_members_with_points()]
                  for _ in range(2))
        assert len(pa) == len(pb) > 0
        assert all(np.array_equal(x, y) for x, y in zip(pa, pb))

    def test_carried_problem_is_placed_in_stage_and_dimmed(self, hold_consumer):
        video = self._video(hold_consumer)
        probe = LayoutProbe(beat_component(video.beats[1], resolve_carry_in(video)["b02"]),
                            duration=4.0)
        probe.construct()
        target = carried(probe, "problem")
        assert region_rect(Region.STAGE).contains(bbox(target))
        opacities = {m.get_fill_opacity() for m in target.family_members_with_points()}
        assert max(opacities) == pytest.approx(1 - DIM_DARKNESS)
