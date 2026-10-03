"""StepTrace: behaviour the registry-wide layout tests do not pin.

test_layout.py already proves examples() validate clean and stress() fits or
refuses cleanly; test_snapshots.py pins what the examples build. This file
pins timing against the audio budget, the schema's refusals, the typed
overflow path, the muting rule that is the point of the component, and the
artifact a later beat carries in.
"""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import asdict

import pytest
from manim import tempconfig
from pydantic import ValidationError

from chalkdust.continuity import ArtifactRecipe, build_artifact, resolve_carry_in
from chalkdust.core.models import BeatSpec, Quality, Region, VideoSpec
from chalkdust.render.worker import TIERS
from chalkdust.scenes.base import ChalkdustScene
from chalkdust.scenes.components.step_trace import (
    FRAME_MIN_SECONDS,
    HEADER_MIN_SECONDS,
    HIGHLIGHT_OPACITY,
    HOLD_MIN_SECONDS,
    MAX_FRAMES,
    MAX_VARIABLES,
    StepTrace,
)
from chalkdust.scenes.regions import LayoutError, bbox, fit_to_region
from chalkdust.validate.geometric import LayoutProbe, validate_beat

BINARY_SEARCH = StepTrace.examples()[0]  # three frames, five segments
# Five frames, seven segments. Not the at-caps stress case: that one fits only
# ~2 pt above the legibility floor under taller-capped mono fonts (Cascadia
# Mono measured 23.6), so on a machine with the theme's real font it could
# refuse with overflow -- correct behaviour, but not what a timing test is for.
ACCUMULATOR = StepTrace.examples()[1]
OVERLOADED = StepTrace.stress()[1]  # both caps, long values -- must refuse
DRAFT = TIERS[Quality.DRAFT]  # 480p15 (D-006): the coarsest frame grid


def _probe(params: dict, tmp_path, duration: float = 8.0) -> LayoutProbe:
    probe = LayoutProbe(StepTrace(params), duration=duration,
                        media_dir=tmp_path)
    probe.construct()
    return probe


def _parts(table, prefix: str) -> list:
    return [m for m in table.submobjects
            if getattr(m, "_chalk_label", "").startswith(prefix)]


def _colours(row) -> list[str]:
    # Text keeps colour on its glyphs; the parent reports a default.
    return [cell[0].get_fill_color().to_hex().upper() for cell in row]


# --- timing (D-002) -----------------------------------------------------------


def _clocked_frames(params: dict, budget: float, tmp_path) -> tuple[int, int]:
    """(frames the render loop produced, the beat's frames) at draft fps.

    Runs Manim's real frame loop -- every play() iterates its time progression,
    every frozen wait() writes its frames -- and counts each frame handed to
    the renderer, without encoding a movie.
    """
    settings = {**asdict(DRAFT), "media_dir": str(tmp_path),
                "disable_caching": True, "write_to_movie": False,
                "save_last_frame": False, "progress_bar": "none",
                "verbosity": "WARNING"}
    with tempconfig(settings):
        scene = ChalkdustScene(StepTrace(params), duration=budget)
        counted: list[int] = []
        add_frame = scene.renderer.add_frame

        def counting(frame, num_frames: int = 1) -> None:
            counted.append(num_frames)
            add_frame(frame, num_frames)

        scene.renderer.add_frame = counting
        scene.render()
        return sum(counted), scene.beat_frames


@pytest.mark.parametrize("params", [BINARY_SEARCH, ACCUMULATOR],
                         ids=["binary-search", "accumulator"])
@pytest.mark.parametrize("factor", [0.5, 3.0], ids=["short", "long"])
def test_clocked_frames_equal_the_beat(params, factor, tmp_path):
    # Narration far shorter and far longer than the trace wants: either way
    # the beat lasts exactly ceil(audio * fps) frames.
    budget = factor * StepTrace(params).min_seconds()
    frames, beat_frames = _clocked_frames(params, budget, tmp_path)
    assert beat_frames == math.ceil(round(budget * DRAFT.frame_rate, 6))
    assert frames == beat_frames


def test_draft_render_is_exactly_the_beat(tmp_path):
    # A real 480p15 clip, frames counted by ffprobe. 3.879 s is not a whole
    # number of frames, so it must round UP to 59.
    duration = 3.879
    with tempconfig({**asdict(DRAFT), "media_dir": str(tmp_path),
                     "disable_caching": True, "progress_bar": "none",
                     "verbosity": "WARNING", "output_file": "step_trace"}):
        scene = ChalkdustScene(StepTrace(BINARY_SEARCH), duration=duration)
        scene.render()
        movie = scene.renderer.file_writer.movie_file_path
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames", "-of", "json", str(movie)],
        capture_output=True, text=True, check=True).stdout
    frames = int(json.loads(out)["streams"][0]["nb_read_frames"])
    assert frames == math.ceil(duration * DRAFT.frame_rate) == 59


def test_min_seconds_is_sum_of_segment_minimums():
    # Header, one segment per frame, final hold.
    n = len(ACCUMULATOR["frames"])
    expected = HEADER_MIN_SECONDS + n * FRAME_MIN_SECONDS + HOLD_MIN_SECONDS
    assert StepTrace(ACCUMULATOR).min_seconds() == pytest.approx(expected)


def test_takes_no_latex():
    # Even a value that looks like LaTeX is drawn as text (stress case 5).
    assert StepTrace(StepTrace.stress()[5]).latex_strings() == []


# --- schema ---------------------------------------------------------------------


@pytest.mark.parametrize("params", [
    {"variables": [], "frames": [[1]]},
    {"variables": ["x"], "frames": []},
    {"variables": ["x"], "frames": [[]]},
    {"variables": ["x", "y"], "frames": [[1, 2], [3]]},
    {"variables": ["x", "x"], "frames": [[1, 2]]},
    {"variables": ["  "], "frames": [[1]]},
    {"variables": ["x"], "frames": [[""]]},
    {"variables": ["x"], "frames": [["two\nlines"]]},
    {"variables": [f"v{i}" for i in range(MAX_VARIABLES + 1)],
     "frames": [list(range(MAX_VARIABLES + 1))]},
    {"variables": ["x"], "frames": [[i] for i in range(MAX_FRAMES + 1)]},
], ids=["no-variables", "no-frames", "empty-frame", "ragged",
        "duplicate-name", "blank-name", "empty-cell", "multiline-cell",
        "too-many-variables", "too-many-frames"])
def test_schema_rejects(params):
    with pytest.raises(ValidationError):
        StepTrace(params)


def test_minimal_trace_validates_clean(tmp_path):
    spec = BeatSpec(id="b01", narration="x starts undefined.",
                    component="StepTrace",
                    params={"variables": ["x"], "frames": [[None]]})
    assert validate_beat(spec, media_dir=tmp_path).ok


# --- typed refusals ---------------------------------------------------------------


def test_overloaded_trace_refuses_as_overflow(tmp_path):
    with pytest.raises(LayoutError) as exc:
        _probe(OVERLOADED, tmp_path)
    assert exc.value.kind == "overflow"


# --- muting -----------------------------------------------------------------------


def test_only_changed_cells_keep_full_contrast(tmp_path):
    params = {"variables": ["a", "b", "tmp"],
              "frames": [[3, 7, None], [3, 7, 3], [7, 7, 3]]}
    probe = _probe(params, tmp_path)
    pal = probe.theme.palette
    rows = _parts(probe.mobjects[0], "frame[")

    fg, muted = pal.fg.upper(), pal.muted.upper()
    # Frame 0: defined values full, undefined tmp muted.
    assert _colours(rows[0]) == [fg, fg, muted]
    # Frame 1: only tmp was assigned.
    assert _colours(rows[1]) == [muted, muted, fg]
    # Frame 2: only a changed.
    assert _colours(rows[2]) == [fg, muted, muted]


# --- continuity (SCENE_SPEC.md §6) ------------------------------------------------


def _recipe(params: dict) -> ArtifactRecipe:
    return ArtifactRecipe(name="trace", producer="StepTrace", params=params)


def test_artifact_is_the_settled_trace(tmp_path):
    # What a later beat carries in is the picture this beat ends on: every
    # row shown, colours as revealed, the highlight on the last frame only.
    probe = _probe(BINARY_SEARCH, tmp_path)
    artifact = build_artifact(_recipe(BINARY_SEARCH), probe.theme)
    settled = probe.mobjects[0]

    for prefix in ("header", "frame["):
        assert [_colours(r) for r in _parts(artifact, prefix)] == \
            [_colours(r) for r in _parts(settled, prefix)]
    assert [s.get_fill_opacity() for s in _parts(artifact, "highlight[")] == \
        pytest.approx([0.0, 0.0, HIGHLIGHT_OPACITY])
    assert all(m.get_fill_opacity() == pytest.approx(1.0)
               for row in _parts(artifact, "frame[")
               for cell in row for m in cell)

    # Same shape as the beat's own table: once both are fitted to STAGE they
    # occupy the same box.
    fit_to_region(artifact, Region.STAGE)
    a, s = bbox(artifact), bbox(settled)
    assert (a.x, a.y, a.width, a.height) == pytest.approx(
        (s.x, s.y, s.width, s.height), abs=1e-6)


def test_artifact_is_deterministic(tmp_path):
    # Same params, same mobject: what keeps the consumer's cache key honest.
    theme = _probe(BINARY_SEARCH, tmp_path).theme
    one = build_artifact(_recipe(BINARY_SEARCH), theme)
    two = build_artifact(_recipe(BINARY_SEARCH), theme)
    boxes = [[(b.x, b.y, b.width, b.height) for b in map(bbox, m.get_family())]
             for m in (one, two)]
    assert boxes[0] == boxes[1]


def test_trace_carries_into_a_later_beat(tmp_path):
    # A trace registered by one beat resolves as a carry-in for the next, and
    # the consuming beat validates clean with it on screen.
    video = VideoSpec(video_id="v", beats=(
        BeatSpec(id="b01", narration="placeholder narration",
                 component="StepTrace", params=BINARY_SEARCH, registers="trace"),
        BeatSpec(id="b02", narration="placeholder narration",
                 component="TitleCard", params={"title": "Found it"},
                 carry_in=["trace"]),
    ))
    recipes = resolve_carry_in(video)["b02"]
    assert [r.producer for r in recipes] == ["StepTrace"]
    report = validate_beat(video.beats[1], recipes=recipes, media_dir=tmp_path)
    assert report.ok, f"\n{report}"
