"""TitleCard: the behaviours the registry-wide layout tests cannot pin.

tests/test_layout.py already proves examples() validate clean and stress()
(3x volume, unwrappable tokens, minimal content) fits or refuses cleanly. This
file pins what is specific to this component: timing against the beat's
frames at short and long narration, min_seconds(), the schema's refusal of
blank text, and that it compiles no LaTeX.
"""

from __future__ import annotations

import json
import math
import subprocess

import pytest
from manim import tempconfig
from manim.animation.animation import prepare_animation
from pydantic import ValidationError

from chalkdust.scenes.base import ChalkdustScene
from chalkdust.scenes.components import make_component
from chalkdust.scenes.components.base import MIN_STEP_SECONDS
from chalkdust.scenes.components.title_card import TitleCard
from chalkdust.validate.geometric import LayoutProbe

NAME = "TitleCard"
EXAMPLES = TitleCard.examples()
STRESS = TitleCard.stress()
DRAFT_FPS = 15

CASES = EXAMPLES + STRESS
CASE_IDS = [f"ex{i}" for i in range(len(EXAMPLES))] + \
    [f"stress{i}" for i in range(len(STRESS))]


class _Clock(LayoutProbe):
    """A probe that also records the frames every play() and wait() asks for,
    without encoding one. A play() with no explicit run_time counts at its
    animations' own default, so a forgotten run_time shows up as a mismatch."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.segments: list[float] = []

    def play(self, *animations, **kwargs) -> None:  # type: ignore[override]
        run_time = kwargs.get("run_time")
        if run_time is None:
            run_time = max(prepare_animation(a).run_time for a in animations)
        self.segments.append(run_time * self.fps)
        super().play(*animations, **kwargs)

    def wait(self, duration: float = 1.0, *args, **kwargs) -> None:  # type: ignore[override]
        self.segments.append(duration * self.fps)


def _clock(params: dict, duration: float) -> _Clock:
    with tempconfig({"frame_rate": DRAFT_FPS}):
        clock = _Clock(make_component(NAME, params), duration=duration)
    clock.construct()
    return clock


# --- timing (D-002) -----------------------------------------------------------


@pytest.mark.parametrize("factor", [0.5, 3.0], ids=["short", "long"])
@pytest.mark.parametrize("params", CASES, ids=CASE_IDS)
def test_clocked_frames_equal_beat_frames(params, factor):
    # Narration far shorter and far longer than the card wants: every play()
    # and wait() is a whole number of frames, and together exactly the beat.
    duration = make_component(NAME, params).min_seconds() * factor
    clock = _clock(params, duration)
    assert all(f == pytest.approx(round(f), abs=1e-6) for f in clock.segments)
    assert sum(round(f) for f in clock.segments) == clock.beat_frames
    assert clock.beat_frames == math.ceil(round(duration * DRAFT_FPS, 6))


def test_short_draft_render_is_exactly_the_beat(tmp_path):
    # The kicker/title/subtitle card at half its minimum: ffprobe must count
    # ceil(audio * fps) frames in the encoded clip.
    params = EXAMPLES[1]
    duration = make_component(NAME, params).min_seconds() * 0.5
    with tempconfig({"media_dir": str(tmp_path), "pixel_width": 854,
                     "pixel_height": 480, "frame_rate": DRAFT_FPS,
                     "disable_caching": True, "progress_bar": "none",
                     "verbosity": "WARNING", "output_file": "short"}):
        scene = ChalkdustScene(make_component(NAME, params), duration=duration)
        scene.render()
        movie = scene.renderer.file_writer.movie_file_path
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames", "-of", "json", str(movie)],
        capture_output=True, text=True, check=True).stdout
    frames = int(json.loads(out)["streams"][0]["nb_read_frames"])
    assert frames == math.ceil(duration * DRAFT_FPS)


def test_min_seconds_is_one_step_per_part():
    # Each part's fade-in is a step the viewer must register; the hold is not.
    assert make_component(NAME, EXAMPLES[0]).min_seconds() == \
        pytest.approx(MIN_STEP_SECONDS * 1)
    assert make_component(NAME, EXAMPLES[1]).min_seconds() == \
        pytest.approx(MIN_STEP_SECONDS * 3)


@pytest.mark.parametrize("params", CASES, ids=CASE_IDS)
def test_compiles_no_latex(params):
    # Plain text only, so the invalid-LaTeX battery item has nothing to aim at.
    assert make_component(NAME, params).latex_strings() == []


# --- schema ---------------------------------------------------------------------


@pytest.mark.parametrize("params", [
    {},
    {"title": ""},
    {"title": "   "},
    {"title": "\n\n"},
    {"title": "Binary Search", "subtitle": ""},
    {"title": "Binary Search", "kicker": "  "},
], ids=["no-title", "empty-title", "blank-title", "newlines-title",
        "empty-subtitle", "blank-kicker"])
def test_schema_rejects(params):
    # A blank part would build an empty card that passes every layout check.
    with pytest.raises(ValidationError):
        TitleCard(params)
