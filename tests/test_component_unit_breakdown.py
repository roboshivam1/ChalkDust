"""UnitBreakdown: timing, schema limits, the LaTeX refusals, label layout,
carry-in.

Layout of examples() and stress() -- including the invalid-LaTeX stress cases,
which must refuse as "invalid_latex" -- is covered by test_layout.py walking
the registry; these pin what that walk cannot see. LaTeX must be installed --
these compile real units and are never skipped without it.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import uuid
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
from manim import tempconfig
from pydantic import ValidationError

from chalkdust.continuity import (
    ArtifactRecipe,
    beat_component,
    build_artifact,
    carried,
    resolve_carry_in,
)
from chalkdust.core.models import BeatSpec, Quality, VideoSpec
from chalkdust.render.worker import TIERS
from chalkdust.scenes.base import ChalkdustScene
from chalkdust.scenes.components.base import wrap
from chalkdust.scenes.components.unit_breakdown import (
    LABEL_WRAP,
    UnitBreakdown,
    _first_baseline,
)
from chalkdust.scenes.regions import INVALID_LATEX, LayoutError, bbox
from chalkdust.scenes.theme import get_theme, resolve_fonts
from chalkdust.validate.geometric import LayoutProbe, validate_beat
from chalkdust.validate.semantic import check_latex

NAME = "UnitBreakdown"
EXAMPLES = UnitBreakdown.examples()
NEWTON = EXAMPLES[0]
MINIMAL = {"quantity": {"unit": "x"}, "decomposition": [{"unit": "y"}]}
DRAFT = TIERS[Quality.DRAFT]
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def media_dir():
    """Manim scratch under the tree's git-ignored work/, never under %TEMP%:
    its 8.3 short path ('LOKAVY~1') carries a '~' that TeX treats as an
    active character, and every MathTex compile there fails."""
    path = ROOT / "work" / "test_unit_breakdown" / uuid.uuid4().hex[:12]
    path.mkdir(parents=True)
    yield path
    shutil.rmtree(path, ignore_errors=True)


def _spec(params: dict, bid: str = "b01", **kw) -> BeatSpec:
    return BeatSpec(id=bid, narration="placeholder narration",
                    component=NAME, params=params, **kw)


# --- timing (D-002) -------------------------------------------------------------


class _FrameClock(ChalkdustScene):
    """A real ChalkdustScene, skipping frame output, that counts the frames
    every play() and wait() would render. compile_animation_data is where
    Manim settles a play's run time (through the base's get_run_time); both
    of Manim's rendering paths turn that run time into int(run_time * fps)
    frames, which tests/test_budget_frames.py proves against real renders."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, skip_animations=True, **kwargs)
        self.frames = 0

    def compile_animation_data(self, *args, **kwargs):
        out = super().compile_animation_data(*args, **kwargs)
        self.frames += int(self.duration * self.fps)
        return out


def _clocked(params: dict, seconds: float, media_dir) -> _FrameClock:
    with tempconfig({**asdict(DRAFT), "media_dir": str(media_dir),
                     "verbosity": "WARNING"}):
        scene = _FrameClock(UnitBreakdown(params), duration=seconds)
        scene.setup()
        scene.construct()
    return scene


class TestTiming:
    """Animation consumes the beat's audio budget exactly, in whole frames."""

    @pytest.mark.parametrize("factor", [0.5, 3.0], ids=["short", "long"])
    @pytest.mark.parametrize("params", [*EXAMPLES, MINIMAL],
                             ids=["ex0", "ex1", "ex2", "minimal"])
    def test_clocked_frames_equal_beat_frames(self, params, factor, media_dir):
        # Narration far shorter and far longer than the animation wants.
        seconds = UnitBreakdown(params).min_seconds() * factor
        scene = _clocked(params, seconds, media_dir)
        assert scene.frames == scene.beat_frames
        # ...which is the audio rounded up to whole frames (float error aside:
        # 13.2 s is 198 frames, not 199 from 198.00000000000003).
        assert -1e-6 <= scene.beat_frames - seconds * DRAFT.frame_rate < 1

    @pytest.mark.parametrize("params", [NEWTON, MINIMAL], ids=["newton", "minimal"])
    def test_far_short_and_far_long_narration(self, params, media_dir):
        # The shortest beat that still gives every segment one frame (anything
        # shorter is refused upstream by the duration rung, SCENE_SPEC.md §8),
        # and the longest one beat may run (semantic.MAX_BEAT_SECONDS).
        segments = len(UnitBreakdown(params)._weights())
        for seconds in (segments / DRAFT.frame_rate, 25.0):
            scene = _clocked(params, seconds, media_dir)
            assert scene.frames == scene.beat_frames, seconds

    def test_draft_render_is_exactly_the_beat(self, media_dir):
        # A real 480p15 encode, frames counted by ffprobe: 6.029 s of audio is
        # ceil(90.435) = 91 frames, whatever the segment count.
        seconds = 6.029
        with tempconfig({**asdict(DRAFT), "media_dir": str(media_dir),
                         "disable_caching": True, "progress_bar": "none",
                         "verbosity": "WARNING", "output_file": "frames"}):
            scene = ChalkdustScene(UnitBreakdown(NEWTON), duration=seconds)
            scene.render()
            movie = scene.renderer.file_writer.movie_file_path
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
             "-show_entries", "stream=nb_read_frames", "-of", "json", str(movie)],
            capture_output=True, text=True, check=True).stdout
        frames = int(json.loads(out)["streams"][0]["nb_read_frames"])
        assert frames == math.ceil(seconds * DRAFT.frame_rate) == 91

    def test_min_seconds_grows_with_each_factor(self):
        one = UnitBreakdown(MINIMAL).min_seconds()
        three = UnitBreakdown(NEWTON).min_seconds()
        assert 0 < one < three


# --- schema ---------------------------------------------------------------------


class TestSchema:
    """Inputs the param model must refuse before anything compiles."""

    @pytest.mark.parametrize("params", [
        {"quantity": {"unit": "x"}, "decomposition": []},
        {"quantity": {"unit": "x"}, "decomposition": [{"unit": "y"}] * 7},
        {"quantity": {"unit": "   "}, "decomposition": [{"unit": "y"}]},
        {"quantity": {"unit": ""}, "decomposition": [{"unit": "y"}]},
        {"quantity": {"unit": "x", "label": "  "}, "decomposition": [{"unit": "y"}]},
        {"decomposition": [{"unit": "y"}]},
        {"quantity": {"unit": "x", "colour": "red"}, "decomposition": [{"unit": "y"}]},
    ], ids=["empty", "too-many", "blank-unit", "empty-unit", "blank-label",
            "no-quantity", "unknown-term-field"])
    def test_rejects(self, params):
        with pytest.raises(ValidationError):
            UnitBreakdown(params)


# --- LaTeX ------------------------------------------------------------------------


class TestLatex:
    """Bad unit LaTeX refuses through theme.math's path, kind "invalid_latex",
    naming the spec field the repair loop must regenerate."""

    def test_latex_strings_is_the_one_compiled_expression(self):
        [expr] = UnitBreakdown(NEWTON).latex_strings()
        for term in [NEWTON["quantity"], *NEWTON["decomposition"]]:
            assert term["unit"] in expr

    @pytest.mark.parametrize("params", EXAMPLES, ids=["ex0", "ex1", "ex2"])
    def test_semantic_rung_compiles_the_examples(self, params):
        # Rung 2 compiles latex_strings() standalone with plain MathTex, so
        # the {{ }} grouped row must be valid there too.
        assert check_latex(UnitBreakdown(params)) == []

    @pytest.mark.parametrize("params,field", [
        ({"quantity": {"unit": r"\mathrm{N}"},
          "decomposition": [{"unit": r"\notacommand{kg}"}]}, "decomposition[0].unit"),
        ({"quantity": {"unit": r"\mathrm{N"},
          "decomposition": [{"unit": r"\mathrm{kg}"}]}, "quantity.unit"),
        ({"quantity": {"unit": "x }} {{ y"},
          "decomposition": [{"unit": "z"}]}, "quantity.unit"),
        ({"quantity": {"unit": r"\mathrm{N}"},
          "decomposition": [{"unit": r"\quad"}]}, "decomposition[0].unit"),
    ], ids=["compile-error", "unclosed-brace", "mis-split", "renders-nothing"])
    def test_refusal_names_the_field(self, params, field):
        report = validate_beat(_spec(params))
        assert report.kinds() == {INVALID_LATEX}, f"\n{report}"
        assert report.findings[0].message.startswith(field), f"\n{report}"

    def test_refusal_is_a_layout_error_not_a_raw_exception(self, media_dir):
        params = {"quantity": {"unit": "{}"}, "decomposition": [{"unit": "z"}]}
        probe = LayoutProbe(UnitBreakdown(params), duration=5.0, media_dir=media_dir)
        with pytest.raises(LayoutError) as exc:
            probe.construct()
        assert exc.value.kind == INVALID_LATEX


# --- layout -----------------------------------------------------------------------


def _built_labels(params: dict) -> tuple:
    probe = LayoutProbe(UnitBreakdown(params), duration=5.0)
    probe.construct()
    [group] = probe.mobjects
    family = group.get_family()
    units = next(m for m in family if getattr(m, "_chalk_label", "") == "units")
    labels = [m for m in family if getattr(m, "_chalk_label", "").startswith("label[")]
    return units, labels


class TestLayout:
    def test_labels_share_a_baseline_whatever_their_letters(self):
        # "mass" has no ascenders, "length" has both; the second-line wrap of
        # "per second squared" must not move its first line either.
        _, labels = _built_labels(NEWTON)
        texts = [t["label"] for t in [NEWTON["quantity"], *NEWTON["decomposition"]]]
        lines = [wrap(s, LABEL_WRAP).split("\n", 1)[0] for s in texts]
        baselines = [_first_baseline(m, line) for m, line in zip(labels, lines)]
        assert max(baselines) - min(baselines) < 0.02, baselines

    def test_operators_sit_midway_between_their_neighbours_glyphs(self):
        # "s^-2" is far narrower than its label "per second squared"; the dot
        # before it must not hug "m" and leave a hole before "s".
        units, _ = _built_labels(NEWTON)
        pieces = units.submobjects
        for left, op, right in zip(pieces[0::2], pieces[1::2], pieces[2::2]):
            gap_before = bbox(op).left - bbox(left).right
            gap_after = bbox(right).left - bbox(op).right
            assert gap_before > 0 and gap_after > 0
            assert gap_before == pytest.approx(gap_after, abs=1e-6)

    def test_wide_labels_under_narrow_units_never_collide(self):
        params = {"quantity": {"unit": "a", "label": "a fairly long label"},
                  "decomposition": [{"unit": "b", "label": "another long one"},
                                    {"unit": "c", "label": "and a third label"}]}
        units, labels = _built_labels(params)
        assert len(labels) == 3
        boxes = [bbox(m) for m in labels]
        for i, a in enumerate(boxes):
            assert a.top < bbox(units).bottom
            for b in boxes[i + 1:]:
                assert not a.intersects(b)


# --- carry-in (SCENE_SPEC.md §6) ----------------------------------------------------


class TestCarryIn:
    def test_rebuild_is_deterministic(self):
        theme = resolve_fonts(get_theme("default"), warn=False)
        recipe = ArtifactRecipe(name="units", producer=NAME, params=NEWTON)
        a, b = build_artifact(recipe, theme), build_artifact(recipe, theme)
        pa = [m.points for m in a.family_members_with_points()]
        pb = [m.points for m in b.family_members_with_points()]
        assert len(pa) == len(pb) > 0
        assert all(np.array_equal(x, y) for x, y in zip(pa, pb))

    def test_later_beat_carries_the_settled_row_in(self):
        video = VideoSpec(video_id="v", beats=(
            _spec(NEWTON, "b01", registers="newton_units"),
            BeatSpec(id="b02", narration="placeholder narration",
                     component="TitleCard", params={"title": "Newtons"},
                     carry_in=["newton_units"]),
        ))
        recipes = resolve_carry_in(video)["b02"]
        assert [r.producer for r in recipes] == [NAME]
        report = validate_beat(video.beats[1], duration=4.0, recipes=recipes)
        assert report.ok, f"\n{report}"
        probe = LayoutProbe(beat_component(video.beats[1], recipes), duration=4.0)
        probe.construct()
        assert carried(probe, "newton_units") in probe.mobjects
