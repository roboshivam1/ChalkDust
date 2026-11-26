"""Render worker: where Manim writes, what it leaves behind, and what the
tiers produce."""

from __future__ import annotations

import json
import subprocess

import pytest
from manim import tempconfig

from chalkdust.core.cache import Cache
from chalkdust.core.models import Beat, BeatSpec, BuildContext, Quality, Video, VideoSpec
from chalkdust.render.worker import render_beat, render_video
from chalkdust.scenes.components.raw_scene import USAGE_LOG_NAME
from chalkdust.validate.geometric import validate_beat


@pytest.fixture(autouse=True)
def _quiet_manim():
    with tempconfig({"verbosity": "WARNING", "progress_bar": "none"}):
        yield


def _beat() -> Beat:
    beat = Beat(spec=BeatSpec(id="b01", narration="placeholder narration",
                              component="TitleCard", params={"title": "Worker"}))
    beat.duration = 1.0
    return beat


def test_render_writes_only_to_work_dir_and_cleans_partials(tmp_path, monkeypatch):
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    cache, work = Cache(tmp_path / "cache"), tmp_path / "work"

    out = render_beat(_beat(), "default", BuildContext(), cache, work)

    assert out.is_file() and out.parent == cache.root / "beats"
    assert not any(cwd.iterdir()), f"render wrote into cwd: {list(cwd.iterdir())}"
    leftovers = [p for p in work.rglob("*") if p.is_file() and p.suffix == ".mp4"]
    assert not leftovers, f"partial movie files left behind: {leftovers}"


def test_draft_tier_sets_output_resolution_and_rate(tmp_path):
    cache, work = Cache(tmp_path / "cache"), tmp_path / "work"

    out = render_beat(_beat(), "default", BuildContext(quality=Quality.DRAFT),
                      cache, work)

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate", "-of", "json",
         str(out)],
        capture_output=True, text=True, check=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    # 480p15 is D-006's draft tier; the render must come out exactly that.
    assert (stream["width"], stream["height"], stream["r_frame_rate"]) == (854, 480, "15/1")


def test_render_applies_the_mechanical_repair_plan(tmp_path, off_edge_beat):
    # The beat's text sits off the right edge: the geometric probe flags it,
    # and repair_beat nudges it back (SCENE_SPEC.md §9 step 1). The render must
    # build with that plan -- its own strict settle checks raise LayoutError on
    # the unrepaired layout, so a clip coming out means the repair was applied.
    with tempconfig({"media_dir": str(tmp_path / "probe")}):
        report = validate_beat(off_edge_beat.spec, duration=1.0)
    assert report.kinds() == {"out_of_bounds"}

    out = render_beat(off_edge_beat, "default", BuildContext(),
                      Cache(tmp_path / "cache"), tmp_path / "work")

    assert out.is_file()


def _video(spec, duration: float = 1.0) -> Video:
    video = Video.from_spec(spec)
    for beat in video.beats:
        beat.duration = duration
    return video


def test_render_video_renders_carry_in_beat_keyed_on_its_producer(tmp_path,
                                                                  carry_in_video):
    # b02 builds only if its carried artifact is on screen (conftest
    # _Highlight), and its key must move when the producing beat is edited
    # (SCENE_SPEC.md §6). Before the worker resolved carry-ins this raised
    # ValueError from beat_render_key.
    cache, work = Cache(tmp_path / "cache"), tmp_path / "work"
    first = render_video(_video(carry_in_video("Bucket array")), BuildContext(),
                         cache, work, verbose=False)
    edited = render_video(_video(carry_in_video("Bucket list")), BuildContext(),
                          cache, work, verbose=False)

    assert all(b.render_path.is_file() for b in first.beats + edited.beats)
    assert first.beats[1].render_path != edited.beats[1].render_path


def _raw_beat() -> Beat:
    beat = Beat(spec=BeatSpec(id="b01", narration="A point traces the curve.",
                              component="RawScene",
                              params={"rationale": "needs os", "code": "import os"}))
    beat.duration = 1.0
    return beat


def test_render_beat_routes_raw_scene_and_degrades(tmp_path):
    # A RawScene beat used to go to plan_repair / RepairedScene, where
    # RawScene.build raises RawSceneError(kind="out_of_process").
    beat = _raw_beat()
    out = render_beat(beat, "default", BuildContext(), Cache(tmp_path / "cache"),
                      tmp_path / "work")
    assert out.is_file() and beat.degraded
    assert (tmp_path / "work" / USAGE_LOG_NAME).is_file()


def test_render_video_routes_raw_scene_and_degrades(tmp_path):
    video = _video(VideoSpec(video_id="v", beats=(_raw_beat().spec,)))
    render_video(video, BuildContext(), Cache(tmp_path / "cache"), tmp_path / "work",
                 verbose=False)
    assert video.beats[0].degraded and video.beats[0].render_path.is_file()
