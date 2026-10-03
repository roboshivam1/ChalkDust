"""Render worker: where Manim writes, what it leaves behind, and what the
tiers produce."""

from __future__ import annotations

import json
import subprocess

import pytest
from manim import tempconfig

from chalkdust.core.cache import Cache
from chalkdust.core.models import Beat, BeatSpec, BuildContext, Quality
from chalkdust.render.worker import render_beat
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
