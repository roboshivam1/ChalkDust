"""NumberLineWalk: timing, schema refusals, and clean layout refusals.

Registry-wide layout invariants (examples clean, stress fits or refuses) live
in test_layout.py; this file pins what is specific to this component.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from dataclasses import asdict

import numpy as np
import pytest
from manim import Create, MoveAlongPath, tempconfig
from manim.animation.animation import prepare_animation
from pydantic import ValidationError

from chalkdust.continuity import ArtifactRecipe, build_artifact
from chalkdust.core.models import BeatSpec, Quality
from chalkdust.render.worker import TIERS
from chalkdust.scenes.base import ChalkdustScene
from chalkdust.scenes.components.number_line_walk import (
    NumberLineWalk,
    _fmt,
    tick_values,
)
from chalkdust.scenes.regions import LayoutError, bbox
from chalkdust.scenes.theme import DEFAULT, resolve_fonts
from chalkdust.validate.geometric import LayoutProbe, validate_beat

# One of every step kind, so every play() path is timed.
MIXED = {
    "range": [-5, 5],
    "steps": [
        {"at": -3, "label": "start"},
        {"to": 2},
        {"to": -1},
        {"at": 4, "label": "reset"},   # second mark: walker slides, no new walker
        {"interval": [0, None], "closed": [True, False], "label": "x ≥ 0"},
    ],
}


FPS = 15
THEME = resolve_fonts(DEFAULT, warn=False)   # as a scene resolves it
# intro + one phase per step + final hold
MIXED_PHASES = 1 + len(MIXED["steps"]) + 1


def _validate(params: dict):
    return validate_beat(BeatSpec(id="b01", narration="placeholder",
                                  component="NumberLineWalk", params=params))


def _render(params: dict, budget: float, tmp_path) -> tuple[int, int]:
    """Render for real at draft fps (frames produced, nothing written).
    Returns (frames rendered, the scene's beat_frames)."""
    with tempconfig({"pixel_width": 128, "pixel_height": 72, "frame_rate": FPS,
                     "media_dir": str(tmp_path), "write_to_movie": False,
                     "disable_caching": True, "verbosity": "WARNING",
                     "progress_bar": "none"}):
        scene = ChalkdustScene(NumberLineWalk(params), duration=budget)
        scene.render()
        # Frames were really produced, so `time` is frames x 1/fps rather than
        # a sum of requested run times (which would hide per-call rounding).
        assert not scene.renderer.skip_animations
        return round(scene.renderer.time * FPS), scene.beat_frames


class TestTiming:
    """The beat lasts exactly its audio rounded up to whole frames (D-002):
    ceil(audio x fps) rendered frames, the thing that has to line up with
    the wav. Every run time comes from scene.budget()."""

    @pytest.mark.parametrize("factor", [0.5, 3.0], ids=["short", "long"])
    def test_renders_exactly_the_beat_frames(self, factor, tmp_path):
        # 0.5x and 3x min_seconds: narration far shorter and far longer than
        # the walk wants. 2.875 s and 17.25 s are not whole frames at 15 fps.
        budget = NumberLineWalk(MIXED).min_seconds() * factor
        rendered, beat_frames = _render(MIXED, budget, tmp_path)
        assert beat_frames == math.ceil(budget * FPS)
        assert rendered == beat_frames

    @pytest.mark.parametrize("frames", [MIXED_PHASES, MIXED_PHASES + 1],
                             ids=["one-frame-each", "a-phase-rounds-to-zero"])
    def test_phases_that_round_to_nothing_still_get_a_frame(self, frames, tmp_path):
        # At 8 frames the second mark's share rounds to 0 frames, and Manim
        # rejects run_time 0 with a bare ValueError. Every phase must get at
        # least one frame and the beat still end on its last frame.
        rendered, beat_frames = _render(MIXED, frames / FPS, tmp_path)
        assert rendered == beat_frames == frames

    def test_draft_render_probed_by_ffprobe_is_exactly_the_beat(self, tmp_path):
        # The encoded clip, counted frame by frame: ceil(audio x fps).
        audio = 7.37
        tier = TIERS[Quality.DRAFT]
        with tempconfig({**asdict(tier), "media_dir": str(tmp_path),
                         "disable_caching": True, "progress_bar": "none",
                         "verbosity": "WARNING", "output_file": "nlw"}):
            scene = ChalkdustScene(NumberLineWalk(MIXED), duration=audio)
            scene.render()
            movie = scene.renderer.file_writer.movie_file_path
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
             "-show_entries", "stream=nb_read_frames", "-of", "json", str(movie)],
            capture_output=True, text=True, check=True).stdout
        frames = int(json.loads(out)["streams"][0]["nb_read_frames"])
        assert frames == math.ceil(audio * tier.frame_rate) == 111

    def test_budget_below_one_frame_per_phase_refuses_typed(self, tmp_path):
        # 0.05x min_seconds is 4 frames for 7 phases: nothing honest can be
        # shown, and the refusal must be a LayoutError the repair loop reads.
        budget = NumberLineWalk(MIXED).min_seconds() * 0.05
        with pytest.raises(LayoutError) as exc:
            _render(MIXED, budget, tmp_path)
        assert exc.value.kind == "overflow"


class TestSemanticHooks:
    def test_min_seconds_is_sum_of_phase_minimums(self):
        # intro 0.75 + 2 marks 1.0 + 2 jumps 2.0 + interval 1.0 + hold 1.0
        assert NumberLineWalk(MIXED).min_seconds() == pytest.approx(5.75)

    def test_no_latex(self):
        assert NumberLineWalk(MIXED).latex_strings() == []


class TestSchema:
    """Inputs that cannot be drawn honestly are refused before anything builds."""

    @pytest.mark.parametrize("params", [
        pytest.param({"range": [0, 10], "steps": []}, id="empty-steps"),
        pytest.param({"range": [0, 10], "steps": [{"to": 3}]}, id="jump-before-mark"),
        pytest.param({"range": [0, 10], "steps": [{"at": 11}]}, id="out-of-range"),
        pytest.param({"range": [5, 5], "steps": [{"at": 5}]}, id="empty-range"),
        pytest.param({"range": [0, 100, 1], "steps": [{"at": 0}]}, id="too-many-ticks"),
        pytest.param({"range": [0, 1, 5], "steps": [{"at": 0}]}, id="too-few-ticks"),
        pytest.param({"range": [0, 10], "steps": [{"at": 3}, {"to": 3}]},
                     id="zero-length-jump"),
        pytest.param({"range": [0, 10], "steps": [{"interval": [None, None]}]},
                     id="interval-unbounded-both"),
        pytest.param({"range": [0, 10], "steps": [{"interval": [6, 2]}]},
                     id="interval-reversed"),
        pytest.param({"range": [0, 10],
                      "steps": [{"interval": [2, None], "closed": [True, True]}]},
                     id="closed-unbounded-end"),
        pytest.param({"range": [0, 10], "steps": [{"at": 1}] * 9}, id="too-many-steps"),
        pytest.param({"range": [0, 10], "steps": [{"at": 1, "label": "   "}]},
                     id="blank-mark-label"),
        pytest.param({"range": [0, 10], "steps": [{"at": 1}, {"to": 5, "label": ""}]},
                     id="empty-jump-label"),
        pytest.param({"range": [0, 10, -1], "steps": [{"at": 1}]},
                     id="negative-tick-step"),
        pytest.param({"range": [0, 1e-12], "steps": [{"at": 0}]},
                     id="tick-step-finer-than-printable"),
        pytest.param({"range": [0, 2e9], "steps": [{"at": 0}]},
                     id="beyond-max-magnitude"),
        pytest.param({"range": [999999999.99, 999999999.999, 0.001],
                      "steps": [{"at": 999999999.99}]},
                     id="tick-step-unresolvable-at-this-magnitude"),
    ])
    def test_rejects(self, params):
        with pytest.raises(ValidationError):
            NumberLineWalk(params)

    @pytest.mark.parametrize("params", [
        pytest.param({"range": [0, 1e400], "steps": [{"at": 0}]}, id="range-inf"),
        pytest.param({"range": [-1e308, 1e308], "steps": [{"at": 0}]},
                     id="span-overflows"),
        pytest.param({"range": [0, 10, float("nan")], "steps": [{"at": 0}]},
                     id="tick-step-nan"),
        pytest.param({"range": [0, 10], "steps": [{"at": float("nan")}]}, id="at-nan"),
        pytest.param({"range": [0, 10], "steps": [{"at": 0}, {"to": float("inf")}]},
                     id="to-inf"),
        pytest.param({"range": [0, 10], "steps": [{"interval": [float("-inf"), 3]}]},
                     id="interval-end-inf"),
    ])
    def test_rejects_non_finite_numbers(self, params):
        # These used to reach math.log10(inf) and raise a raw OverflowError;
        # they must be a pydantic refusal like any other bad param.
        with pytest.raises(ValidationError):
            NumberLineWalk(params)

    @pytest.mark.parametrize("rng", [
        pytest.param([0, 1000000, 0.00001], id="1e11-ticks"),
        pytest.param([0, 1000000, 0.001], id="1e9-ticks"),
    ])
    def test_tick_count_refused_without_building_the_ticks(self, rng):
        # Counted arithmetically: building these lists first meant 1e9..1e11
        # floats (a hang, then MemoryError) before the count check ran.
        with pytest.raises(ValidationError, match="ticks"):
            NumberLineWalk({"range": rng, "steps": [{"at": 0}]})


class TestLayout:
    def test_minimal_walk_validates_clean(self):
        report = _validate({"range": [0, 1], "steps": [{"at": 0}]})
        assert report.ok, f"\n{report}"

    def test_jump_too_small_to_draw_refuses_as_illegible(self):
        report = _validate({"range": [0, 1000], "steps": [{"at": 500}, {"to": 501}]})
        assert report.kinds() == {"illegible"}, f"\n{report}"

    def test_walk_too_tangled_for_own_arc_labels_refuses_as_overflow(self):
        # Back and forth over one span, five jumps: the arcs cannot each carry
        # their own label inside the stage, so the beat refuses rather than
        # stacking a label over an arc it does not belong to.
        steps = [{"at": 0, "label": "start"}, {"to": 7}, {"to": -3}, {"to": 4},
                 {"to": -8}, {"to": 9}, {"interval": [-3, 4]}, {"to": 2}]
        report = _validate({"range": [-10, 10], "steps": steps})
        assert report.kinds() == {"overflow"}, f"\n{report}"

    def test_labels_that_cannot_clear_each_other_refuse_as_overflow(self):
        words = "a label far longer than any jump on a number line should carry"
        steps = [{"at": -9}] + [{"to": v, "label": words}
                                for v in (-4, 1, -2, 6, 9, -6, 3)]
        report = _validate({"range": [-10, 10], "steps": steps})
        assert report.kinds() == {"overflow"}, f"\n{report}"

    def test_glyphless_jump_label_refuses_instead_of_hanging(self):
        # Pango draws a label that opens with a right-to-left letter, or is
        # only zero-width/bidi controls, as no glyphs. Such a jump label never
        # rose with its arc, so the stacking search looped forever. Run in a
        # child process so a regression fails on the timeout instead of
        # hanging the suite.
        code = (
            "from chalkdust.core.models import BeatSpec\n"
            "from chalkdust.validate.geometric import validate_beat\n"
            "for text in ['\\u05e9\\u05dc\\u05d5\\u05dd', '\\u200b',"
            " '\\u200f\\u05e9\\u05dc\\u05d5\\u05dd']:\n"
            "    r = validate_beat(BeatSpec(id='b01', narration='x',"
            " component='NumberLineWalk', params={'range': [0, 10],"
            " 'steps': [{'at': 1}, {'to': 7, 'label': text}]}))\n"
            "    print('kinds', sorted(r.kinds()))\n"
        )
        out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                             text=True, timeout=180)
        assert out.returncode == 0, out.stderr
        kinds = [ln for ln in out.stdout.splitlines() if ln.startswith("kinds ")]
        assert kinds == ["kinds ['illegible']"] * 3, out.stdout

    @pytest.mark.parametrize("step", [
        pytest.param({"at": 1, "label": "שלום"}, id="mark-rtl"),
        pytest.param({"at": 1, "label": "​"}, id="mark-zero-width"),
        pytest.param({"interval": [2, 5], "label": "‮"},
                     id="interval-bidi-control"),
    ])
    def test_glyphless_mark_or_interval_label_refuses_typed(self, step):
        # Used to be a raw IndexError from set_x on a mobject with no points.
        report = _validate({"range": [0, 10], "steps": [step]})
        assert report.kinds() == {"illegible"}, f"\n{report}"


class TestJumpLabelsReadAsTheirOwnArc:
    """A jump label must sit on its own arc: directly under the label's centre
    the first arc is its own, and no other arc is nearer the label. A backward
    jump nested under a forward one used to get its label nudged up over the
    forward arc, where it read as the forward jump's.

    Checked on the built scene with independent geometry (densely sampled
    curves), not with the component's own placement predicate.
    """

    @pytest.mark.parametrize("params,pairs", [
        pytest.param(NumberLineWalk.examples()[0],
                     {1: "jump 2->7", 2: "jump 7->4"},
                     id="example0-back-under-forward"),
        pytest.param({"range": [-1000, 1000],
                      "steps": [{"at": -1000}, {"to": 1000}, {"to": -1000},
                                {"to": 0, "label": "back to zero"}]},
                     {1: "jump −1000->1000", 2: "jump 1000->−1000",
                      3: "jump −1000->0"},
                     id="same-span-there-and-back"),
        pytest.param({"range": [-10, 10],
                      "steps": [{"at": 0}, {"to": 4}, {"to": -2}, {"to": 5},
                                {"to": -1}]},
                     {1: "jump 0->4", 2: "jump 4->−2", 3: "jump −2->5",
                      4: "jump 5->−1"},
                     id="zigzag"),
    ])
    def test_each_label_reads_as_its_own_arc(self, params, pairs):
        scene = LayoutProbe(NumberLineWalk(params), duration=10.0, strict=True)
        scene.construct()
        tagged = {getattr(m, "_chalk_label", None): m for m in scene.mobjects}
        curves = {name: np.array([tagged[name].point_from_proportion(t)
                                  for t in np.linspace(0, 1, 400)])
                  for name in pairs.values()}
        for step, own in pairs.items():
            lab = tagged[f"steps[{step}] label"]
            left, right = lab.get_left()[0], lab.get_right()[0]
            bottom, top = lab.get_bottom()[1], lab.get_top()[1]

            def below_centre(pts):
                col = pts[(np.abs(pts[:, 0] - lab.get_x()) < 0.05)
                          & (pts[:, 1] < bottom)]
                return col[:, 1].max() if len(col) else -np.inf

            def gap(pts):
                dx = np.maximum(0, np.maximum(left - pts[:, 0], pts[:, 0] - right))
                dy = np.maximum(0, np.maximum(bottom - pts[:, 1], pts[:, 1] - top))
                return np.hypot(dx, dy).min()

            assert max(curves, key=lambda n: below_centre(curves[n])) == own, step
            assert min(curves, key=lambda n: gap(curves[n])) == own, step


class TestNumberFormat:
    @pytest.mark.parametrize("value,text", [
        (0.1 + 0.2, "0.3"),               # float error is not content
        (-0.0, "0"),                      # never a signed zero
        (1e9, "1000000000"),              # positional, never "1e+09"
        (1e9 + 0.1, "1000000000.1"),      # ... so neighbouring ticks differ
        (-2.5, "−2.5"),                   # a true minus sign
        (1 / 3, "0.333333"),              # at most six decimals
    ])
    def test_fmt(self, value, text):
        assert _fmt(value) == text


class TestCarryIn:
    """A later beat can carry the settled walk in (SCENE_SPEC.md §6)."""

    @staticmethod
    def _recipe(params: dict) -> ArtifactRecipe:
        return ArtifactRecipe(name="walk", producer="NumberLineWalk", params=params)

    def test_artifact_is_deterministic(self):
        recipe = self._recipe(NumberLineWalk.examples()[0])
        a, b = (build_artifact(recipe, THEME) for _ in range(2))
        labels = [[getattr(m, "_chalk_label", None) for m in art] for art in (a, b)]
        assert labels[0] == labels[1]
        np.testing.assert_allclose(a.get_all_points(), b.get_all_points())

    def test_artifact_shows_the_walk_where_it_ends(self):
        # example0 is 2 -> 7 -> 4: the walker stands on 4, both arcs drawn.
        params = NumberLineWalk.examples()[0]
        art = build_artifact(self._recipe(params), THEME)
        names = {getattr(m, "_chalk_label", None) for m in art}
        assert {"number line", "tick labels", "jump 2->7", "jump 7->4",
                "steps[0] label", "steps[1] label", "steps[2] label"} <= names
        line = next(m for m in art if getattr(m, "_chalk_label", None)
                    == "number line")[0]
        walker = art[-1]
        assert walker.get_center() == pytest.approx(line.n2p(4), abs=1e-6)

    def test_carried_into_a_beat_validates_clean(self):
        consumer = BeatSpec(id="b02", narration="placeholder", component="TitleCard",
                            params={"title": "Where did we land?"}, carry_in=["walk"])
        report = validate_beat(consumer, recipes=[
            self._recipe(NumberLineWalk.examples()[0])])
        assert report.ok, f"\n{report}"


class _HalfwayProbe(LayoutProbe):
    """Records, halfway through every jump, how far the drawn stroke's end is
    from the walker and how opaque the arrow tip is."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.halfway: list[tuple[float, float]] = []

    def play(self, *animations, **kwargs) -> None:  # type: ignore[override]
        anims = [prepare_animation(a) for a in animations]
        stroke = next((a for a in anims if isinstance(a, Create)), None)
        walk = next((a for a in anims if isinstance(a, MoveAlongPath)), None)
        if stroke is not None and walk is not None:
            for a in anims:
                a.begin()
            for a in anims:
                a.interpolate(0.5)
            tip = next(a.mobject for a in anims
                       if getattr(a.mobject, "_chalk_label", "").endswith(" tip"))
            gap = float(np.linalg.norm(stroke.mobject.get_end()
                                       - walk.mobject.get_center()))
            self.halfway.append((gap, tip.get_fill_opacity()))
            for a in anims:
                a.interpolate(1.0)
        super().play(*anims, **kwargs)


class TestJumpDrawing:
    def test_stroke_grows_under_the_walker(self):
        # Create() over an arc with its tip attached drew the whole stroke in
        # the first half of the jump (lag_ratio 1 over arc + tip), so the line
        # ran a half-chord ahead of the walker. Halfway through a jump the
        # stroke must end at the walker, and the tip must not be showing yet.
        params = NumberLineWalk.examples()[0]
        probe = _HalfwayProbe(NumberLineWalk(params), duration=10.0, strict=True)
        probe.construct()
        assert len(probe.halfway) == 2
        for gap, tip_opacity in probe.halfway:
            assert gap < 0.25, probe.halfway       # was 3.4 and 1.9 (whole stroke drawn)
            assert tip_opacity == 0.0, probe.halfway


class TestTicks:
    @pytest.mark.parametrize("rng,expected", [
        ([-2, 8], [float(v) for v in range(-2, 9)]),     # integer range: unit ticks
        ([0, 1], [k * 0.1 for k in range(11)]),          # fractional: 1/2/5 step
        ([0.5, 3.5, 1], [1.0, 2.0, 3.0]),                # explicit: multiples of it
    ])
    def test_tick_positions(self, rng, expected):
        assert tick_values(tuple(rng)) == pytest.approx(expected)
