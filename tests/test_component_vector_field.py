"""VectorField: timing, expression safety, refusals, the magnitude clamp, and
the carry-in rebuild.

Layout validity of examples() and stress() is covered by tests/test_layout.py,
which walks the registry; this file pins what is specific to VectorField.
"""

from __future__ import annotations

import builtins
import json
import math
import subprocess
from dataclasses import asdict

import pytest
from manim import Wait, tempconfig
from pydantic import ValidationError

from chalkdust.continuity import build_artifact, resolve_carry_in
from chalkdust.core.models import BeatSpec, Quality, Region, VideoSpec
from chalkdust.render.worker import TIERS
from chalkdust.scenes.base import ChalkdustScene
from chalkdust.scenes.components.vector_field import (
    ARROW_FILL,
    MIN_ARROWS_SECONDS,
    MIN_HOLD_SECONDS,
    MIN_PLANE_SECONDS,
    VectorField,
    grid_shape,
)
from chalkdust.scenes.regions import DEFAULT_PADDING, bbox, fit_to_region, region_rect
from chalkdust.validate.geometric import LayoutProbe, validate_beat

ROTATION = {"field_fn": {"x": "-y", "y": "x"}}
DRAFT = TIERS[Quality.DRAFT]


def _probe(params: dict, duration: float = 8.0) -> LayoutProbe:
    probe = LayoutProbe(VectorField(params), duration=duration)
    probe.construct()
    return probe


def _labelled(scene, name: str):
    return next(m for m in scene.mobjects if getattr(m, "_chalk_label", "") == name)


def _field(**overrides) -> dict:
    return {**ROTATION, **overrides}


# --- timing (D-002) ---------------------------------------------------------


class _FrameClock(LayoutProbe):
    """A probe that also counts the frames every play() and wait() would
    render, using the base scene's own run-time rule (ChalkdustScene.
    get_run_time snaps each one to whole frames), without encoding a frame.
    A play() with no explicit run_time counts at its animations' default, so
    a forgotten run_time shows up as a frame mismatch."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.frames = 0

    def _count(self, animations) -> None:
        self.frames += int(self.get_run_time(animations) * self.fps)

    def play(self, *animations, **kwargs) -> None:  # type: ignore[override]
        # compile_animations applies play()'s kwargs (run_time) to each
        # animation, as Scene.play does before asking for the run time.
        self._count(self.compile_animations(*animations, **kwargs))
        super().play(*animations, **kwargs)

    def wait(self, duration: float = 1.0, *args, **kwargs) -> None:  # type: ignore[override]
        self._count([Wait(run_time=duration)])


def _render_frames(params: dict, duration: float, out_dir) -> int:
    """Render at the draft tier and count the clip's frames with ffprobe."""
    with tempconfig({**asdict(DRAFT), "media_dir": str(out_dir),
                     "disable_caching": True, "progress_bar": "none",
                     "verbosity": "WARNING", "output_file": "frames"}):
        scene = ChalkdustScene(VectorField(params), duration=duration)
        scene.render()
        movie = scene.renderer.file_writer.movie_file_path
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames", "-of", "json", str(movie)],
        capture_output=True, text=True, check=True).stdout
    return int(json.loads(out)["streams"][0]["nb_read_frames"])


@pytest.mark.parametrize("factor", [0.5, 3.0], ids=["short", "long"])
@pytest.mark.parametrize("example", range(len(VectorField.examples())))
def test_clocked_frames_equal_the_beat(example, factor):
    # Narration far shorter (half of min_seconds) and far longer (3x) than
    # the animation wants: either way every play and wait together take
    # exactly ceil(audio * fps) frames at the draft tier.
    params = VectorField.examples()[example]
    duration = VectorField(params).min_seconds() * factor + 0.0123  # off-frame
    with tempconfig({"frame_rate": DRAFT.frame_rate}):
        clock = _FrameClock(VectorField(params), duration=duration)
    clock.construct()
    assert clock.frames == clock.beat_frames == math.ceil(duration * DRAFT.frame_rate)


def test_draft_render_is_exactly_the_beat(tmp_path):
    # The real thing, counted by ffprobe: 5.479 s of audio is 83 frames at
    # 15 fps (82.185 rounded up).
    duration = 5.479
    assert _render_frames(ROTATION, duration, tmp_path) ==         math.ceil(duration * DRAFT.frame_rate) == 83


def test_min_seconds_gives_every_step_its_floor():
    vf = VectorField(ROTATION)
    assert vf.min_seconds() == MIN_PLANE_SECONDS + MIN_ARROWS_SECONDS + MIN_HOLD_SECONDS


def test_compiles_no_latex():
    assert VectorField(ROTATION).latex_strings() == []


# --- expression safety ------------------------------------------------------


@pytest.mark.parametrize("expr", [
    "__import__('os').system('echo pwned')",
    "x.real",
    "(lambda: 1)()",
    "x^2",                       # BitXor, the classic model mistake
    "foo(x)",
    "sin(x, y)",
    "sin(x=1)",
    "'a'",
    "True",
    "[x]",
    "x if y else 1",
    "a_sixty_character_identifier_that_no_parser_should_accept_ok",
    "https://example.com/fields/a-sixty-character-url/with/no/spaces",
    "x +",
    "",
    "x" * 121,                   # over MAX_EXPR_CHARS
])
def test_rejects_expression(expr):
    with pytest.raises(ValidationError):
        VectorField({"field_fn": {"x": expr, "y": "x"}})


def test_caret_error_points_at_the_fix():
    with pytest.raises(ValidationError, match=r"use \*\* for powers"):
        VectorField({"field_fn": {"x": "x^2", "y": "y"}})


def test_never_calls_eval(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("VectorField must not eval field expressions")

    monkeypatch.setattr(builtins, "eval", refuse)
    monkeypatch.setattr(builtins, "exec", refuse)
    _probe({"field_fn": {"x": "sin(y)*exp(-x**2/4)", "y": "cos(x)+pi*e"}})


# --- schema refusals --------------------------------------------------------


@pytest.mark.parametrize("overrides", [
    {"x_range": [2, 1]},             # decreasing
    {"x_range": [1, 1]},             # empty
    {"x_range": [-50, 50]},          # aspect 25: a strip, not a plane
    {"y_range": [-20, 20]},          # aspect 0.25
    {"x_range": [-1e9, 1e9]},        # outside the bound
    {"sample_density": 3},
    {"sample_density": 21},
], ids=lambda o: str(o))
def test_rejects_bad_domain_or_density(overrides):
    with pytest.raises(ValidationError):
        VectorField(_field(**overrides))


@pytest.mark.parametrize("params", [
    {},                                   # no field at all
    {"field_fn": {}},                     # a field with no components
    {"field_fn": {"x": "1"}},             # half a field
    {"field_fn": {"x": "   ", "y": "1"}},  # blank component
], ids=["missing", "empty", "half", "blank"])
def test_rejects_empty_field(params):
    # The model has no meaningful empty VectorField: no field, nothing to draw.
    with pytest.raises(ValidationError):
        VectorField(params)


@pytest.mark.parametrize("fn", [
    {"x": "0", "y": "0"},                      # zero everywhere
    {"x": "sqrt(-1-x**2)", "y": "log(-1-y**2)"},  # undefined everywhere
])
def test_rejects_field_with_nothing_to_draw(fn):
    with pytest.raises(ValidationError, match="nothing to draw"):
        VectorField({"field_fn": fn})


def test_minimal_field_renders():
    probe = _probe({"field_fn": {"x": "1", "y": "0"}, "sample_density": 4})
    assert len(_labelled(probe, "field arrows").submobjects) == 8  # 4 x 2 grid


# --- layout -----------------------------------------------------------------


def test_too_dense_for_the_plane_refuses_with_overflow():
    spec = BeatSpec(id="b01", narration="n", component="VectorField",
                    params=_field(sample_density=20, x_range=[-2, 2], y_range=[-2, 2]))
    assert validate_beat(spec).kinds() == {"overflow"}


# stress()[0] is the too-dense case, which refuses instead of building.
@pytest.mark.parametrize("params", [*VectorField.examples(), *VectorField.stress()[1:]])
def test_arrows_stay_inside_the_plane(params):
    probe = _probe(params)
    plane = bbox(_labelled(probe, "field plane"))
    for arrow in _labelled(probe, "field arrows").submobjects:
        assert plane.contains(bbox(arrow), tol=1e-6), arrow._chalk_label


def test_singular_field_is_clamped_and_skips_undefined_points():
    # 9x3 grid: the origin is a sample point, where the field is inf/nan.
    params = {"field_fn": {"x": "x/(x**2+y**2)**1.5", "y": "y/(x**2+y**2)**1.5"},
              "sample_density": 9, "x_range": [-3, 3], "y_range": [-1, 1]}
    probe = _probe(params)
    arrows = _labelled(probe, "field arrows").submobjects
    nx, ny = grid_shape(VectorField(params).params)
    assert len(arrows) < nx * ny
    assert not any(a._chalk_label == "arrow(0.00, 0.00)" for a in arrows)

    inner = region_rect(Region.STAGE).inset(DEFAULT_PADDING)
    unit = min(inner.width / 6, inner.height / 2)
    full = ARROW_FILL * unit * min(6 / nx, 2 / ny)
    lengths = [a.get_length() for a in arrows]
    assert max(lengths) == pytest.approx(full, rel=1e-6)


# --- carry-in (SCENE_SPEC.md §6) --------------------------------------------


def test_carried_field_is_the_settled_field():
    # A later beat carrying the field in sees exactly what this beat left on
    # screen: the same arrows, the same boxes once placed in STAGE.
    video = VideoSpec(video_id="v", beats=(
        BeatSpec(id="b01", narration="placeholder narration",
                 component="VectorField", params=ROTATION, registers="field"),
        BeatSpec(id="b02", narration="placeholder narration",
                 component="TitleCard", params={"title": "Curl"},
                 carry_in=["field"]),
    ))
    (recipe,) = resolve_carry_in(video)["b02"]
    probe = _probe(ROTATION)
    artifact = build_artifact(recipe, probe.theme)
    fit_to_region(artifact, Region.STAGE)

    plane, field = artifact.submobjects
    settled = _labelled(probe, "field arrows").submobjects
    assert [a._chalk_label for a in field.submobjects] == [a._chalk_label for a in settled]
    for got, want in [(plane, _labelled(probe, "field plane")),
                      *zip(field.submobjects, settled)]:
        g, w = bbox(got), bbox(want)
        assert (g.x, g.y, g.width, g.height) == pytest.approx(
            (w.x, w.y, w.width, w.height), abs=1e-6)
