"""ChalkdustScene.budget() hands out whole frames (D-002).

The beat lasts ceil(audio * fps) frames -- never shorter than its narration,
at most one frame longer -- and every play() and wait() renders exactly the
frames budget() gave it, so a rendered clip is exactly that long. Before this,
Manim rounded each play up and each frozen wait down on its own, and a beat
drifted from its audio by a frame per segment.

The render tests encode real clips and count their frames with ffprobe.
"""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import asdict

import pytest
from manim import FadeIn, tempconfig

from chalkdust.core.models import Quality, Region
from chalkdust.render.worker import TIERS
from chalkdust.scenes.base import ChalkdustScene, frames_covering, split_frames
from chalkdust.scenes.components import Component, ComponentParams, get_component, make_component
from chalkdust.scenes.theme import body_text

DRAFT = TIERS[Quality.DRAFT]
FINAL = TIERS[Quality.FINAL]


class _PlayThenWait(Component):
    """One play, one frozen wait, weighted 23:31. At 15 fps those are frame
    counts stock Manim gets wrong from float error: a 23/15 s play renders 24
    frames. Not registered."""

    name = "_PlayThenWait"
    Params = ComponentParams

    def regions(self) -> set[Region]:
        return {Region.STAGE}

    def build(self, scene) -> None:
        play, hold = scene.budget(23, 31)
        scene.play(FadeIn(body_text("Frames", scene.theme)), run_time=play)
        scene.wait(hold)


def _scene(duration: float, fps: int) -> ChalkdustScene:
    # The scene keeps the frame rate it was built under, as its camera does.
    with tempconfig({"frame_rate": fps}):
        return ChalkdustScene(_PlayThenWait({}), duration=duration)


def _render_frames(component: Component, duration: float, tier, out_dir) -> int:
    """Render `component` at `tier` and count the clip's frames with ffprobe."""
    with tempconfig({**asdict(tier), "media_dir": str(out_dir),
                     "disable_caching": True, "progress_bar": "none",
                     "verbosity": "WARNING", "output_file": "frames"}):
        scene = ChalkdustScene(component, duration=duration)
        scene.render()
        movie = scene.renderer.file_writer.movie_file_path
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames", "-of", "json", str(movie)],
        capture_output=True, text=True, check=True).stdout
    return int(json.loads(out)["streams"][0]["nb_read_frames"])


# --- the arithmetic -----------------------------------------------------------


@pytest.mark.parametrize("seconds,fps,frames", [
    (3.879, 15, 59), (3.879, 60, 233), (4.0, 15, 60), (4.0, 60, 240),
    (0.01, 15, 1),
], ids=["draft", "final", "exact-draft", "exact-final", "tiny"])
def test_beat_rounds_up_to_whole_frames(seconds, fps, frames):
    # Exact multiples stay put (4.0 s is 60 frames, not 61 from float error);
    # anything else rounds UP, so video never ends before the narration.
    assert frames_covering(seconds, fps) == frames


@pytest.mark.parametrize("total,weights", [
    (59, (1, 3)), (59, (1, 1, 1, 3)), (107, (1, 2, 2, 2, 2)),
    (233, (1.0, 0.75, 1.0, 0.5, 0.75, 1.0, 0.75)), (5, (1, 1, 1, 1, 1)),
])
def test_split_frames_sums_exactly_and_tracks_weights(total, weights):
    out = split_frames(total, weights)
    assert sum(out) == total
    assert all(n >= 1 for n in out)
    # Each share is its exact proportion, rounded one way or the other.
    exact = [total * w / sum(weights) for w in weights]
    assert all(abs(n - x) < 1 for n, x in zip(out, exact))


def test_split_frames_gives_every_segment_a_frame():
    # Manim cannot play zero frames. A segment whose share rounds to zero takes
    # one from the largest; only with more segments than frames does the total
    # grow, and then only to one frame per segment.
    assert split_frames(8, (100, 1, 1)) == [6, 1, 1]
    assert split_frames(3, (1, 1, 1, 1, 1)) == [1, 1, 1, 1, 1]


@pytest.mark.parametrize("fps", [DRAFT.frame_rate, FINAL.frame_rate])
def test_budget_is_whole_frames_summing_to_the_beat(fps):
    scene = _scene(3.879, fps)
    times = scene.budget(1, 0.75, 1, 0.5, 0.75)
    frames = [t * fps for t in times]
    assert all(f == pytest.approx(round(f), abs=1e-9) for f in frames)
    assert sum(round(f) for f in frames) == math.ceil(3.879 * fps)


def test_budget_rejects_negative_weights():
    with pytest.raises(ValueError):
        _scene(4.0, 15).budget(1, -1, 2)


# --- real renders -------------------------------------------------------------


@pytest.mark.parametrize("tier", [DRAFT, FINAL], ids=["draft", "final"])
def test_rendered_frames_equal_audio_rounded_up(tier, tmp_path):
    # 23 + 31 = 54 frames at draft; the same weights at final. Stock Manim
    # rendered 55 at draft (a 23/15 s play is 24 frames there).
    duration = 53.5 / 15
    assert _render_frames(_PlayThenWait({}), duration, tier, tmp_path) == \
        math.ceil(duration * tier.frame_rate)


@pytest.mark.parametrize("name", ["TitleCard", "BulletReveal", "EquationDerivation"])
def test_component_draft_render_is_exactly_the_beat(name, tmp_path):
    # Every component built on budget() inherits exact timing: 3.879 s of
    # audio is 59 frames at 15 fps, whatever the segment count.
    params = get_component(name).examples()[1]
    assert _render_frames(make_component(name, params), 3.879, DRAFT, tmp_path) == 59
