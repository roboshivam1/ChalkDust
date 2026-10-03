"""BulletReveal: the behaviours the registry-wide layout tests cannot pin.

tests/test_layout.py already proves examples() validate clean and stress()
(3x volume, unwrappable tokens, minimal content) fits or refuses cleanly. This
file pins what is specific to this component: timing against the beat's
frames at short and long narration, min_seconds(), the schema's refusals, and
that 3x volume refuses rather than shrinks.
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
from chalkdust.scenes.components.bullet_reveal import BulletReveal
from chalkdust.scenes.regions import LayoutError
from chalkdust.validate.geometric import LayoutProbe

NAME = "BulletReveal"
EXAMPLES = BulletReveal.examples()
STRESS = BulletReveal.stress()
DRAFT_FPS = 15

# Stress cases that refuse as overflow at build time, so have no timeline: six
# ~150-character bullets, and six ~80-character bullets revealed all at once.
REFUSING = [0, 5]
TIMED = EXAMPLES + [p for i, p in enumerate(STRESS) if i not in REFUSING]
TIMED_IDS = [f"ex{i}" for i in range(len(EXAMPLES))] + \
    [f"stress{i}" for i in range(len(STRESS)) if i not in REFUSING]


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


def _draft_frames(params: dict, duration: float, out_dir) -> int:
    """Render at 480p15 and count the clip's frames with ffprobe."""
    with tempconfig({"media_dir": str(out_dir), "pixel_width": 854,
                     "pixel_height": 480, "frame_rate": DRAFT_FPS,
                     "disable_caching": True, "progress_bar": "none",
                     "verbosity": "WARNING", "output_file": "draft"}):
        scene = ChalkdustScene(make_component(NAME, params), duration=duration)
        scene.render()
        movie = scene.renderer.file_writer.movie_file_path
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames", "-of", "json", str(movie)],
        capture_output=True, text=True, check=True).stdout
    return int(json.loads(out)["streams"][0]["nb_read_frames"])


# --- timing (D-002) -----------------------------------------------------------


@pytest.mark.parametrize("factor", [0.5, 3.0], ids=["short", "long"])
@pytest.mark.parametrize("params", TIMED, ids=TIMED_IDS)
def test_clocked_frames_equal_beat_frames(params, factor):
    # Narration far shorter and far longer than the reveal wants: every play()
    # and wait() is a whole number of frames, and together exactly the beat.
    duration = make_component(NAME, params).min_seconds() * factor
    clock = _clock(params, duration)
    assert all(f == pytest.approx(round(f), abs=1e-6) for f in clock.segments)
    assert sum(round(f) for f in clock.segments) == clock.beat_frames
    assert clock.beat_frames == math.ceil(round(duration * DRAFT_FPS, 6))


def test_reveal_all_short_draft_render_is_exactly_the_beat(tmp_path):
    # Six bullets revealed at once, at half the minimum: 5 frames of audio.
    # The reveal is one play(), so it is budgeted as one segment; budgeting
    # it as six summed ones gave every bullet a frame and ran 7 frames long.
    params = EXAMPLES[2]
    duration = make_component(NAME, params).min_seconds() * 0.5
    assert _draft_frames(params, duration, tmp_path) == \
        math.ceil(duration * DRAFT_FPS)


def test_min_seconds_gives_the_shortest_reveal_a_minimum():
    # Weights: heading 1, each bullet 2, hold 2. At min_seconds() the shortest
    # reveal lasts MIN_STEP_SECONDS -- the heading when there is one, else a
    # bullet, else (reveal="all") the single fade of every bullet.
    assert make_component(NAME, EXAMPLES[0]).min_seconds() == \
        pytest.approx(MIN_STEP_SECONDS * (2 + 2) / 2)
    assert make_component(NAME, EXAMPLES[1]).min_seconds() == \
        pytest.approx(MIN_STEP_SECONDS * (1 + 3 * 2 + 2) / 1)
    assert make_component(NAME, EXAMPLES[2]).min_seconds() == \
        pytest.approx(MIN_STEP_SECONDS * (12 + 2) / 12)


@pytest.mark.parametrize("params", EXAMPLES + STRESS)
def test_compiles_no_latex(params):
    # Plain text only, so the invalid-LaTeX battery item has nothing to aim at.
    assert make_component(NAME, params).latex_strings() == []


# --- schema ---------------------------------------------------------------------


@pytest.mark.parametrize("params", [
    {"items": []},
    {"items": ["a"] * 7},
    {"items": [""]},
    {"items": ["a", "   "]},
    {"items": ["a"], "heading": ""},
    {"items": ["a"], "heading": "   "},
], ids=["no-items", "seven-items", "empty-item", "blank-item",
        "empty-heading", "blank-heading"])
def test_schema_rejects(params):
    # A blank heading built an empty mobject at the origin that tripped the
    # overlap check; a blank bullet is a dot with nothing beside it.
    with pytest.raises(ValidationError):
        BulletReveal(params)


# --- typed refusals ---------------------------------------------------------------


@pytest.mark.parametrize("index", REFUSING)
def test_three_x_volume_refuses_rather_than_shrinks(index):
    # 3x volume must refuse as overflow, never render below the font floor.
    probe = LayoutProbe(make_component(NAME, STRESS[index]), duration=8.0)
    with pytest.raises(LayoutError) as exc:
        probe.construct()
    assert exc.value.kind == "overflow"
