"""FreeBodyDiagram: the behaviours the registry-wide layout tests cannot pin.

tests/test_layout.py already proves examples() validate clean and stress()
fits or refuses cleanly. This file pins what is specific to this component:
timing in whole frames, the schema's refusals, the typed LaTeX failure, the
magnitude clamp, side-by-side clustering, the promise that a label never
touches another force's arrow, and the carry-in artifact.

These tests compile LaTeX, so `latex` and `dvisvgm` must be on PATH.
"""

from __future__ import annotations

import json
import math
import subprocess

import numpy as np
import pytest
from manim import Arrow, MathTex, tempconfig
from pydantic import ValidationError

from chalkdust.continuity import (
    ArtifactRecipe,
    beat_component,
    build_artifact,
    carried,
    resolve_carry_in,
)
from chalkdust.core.models import BeatSpec, Region, VideoSpec
from chalkdust.scenes.base import ChalkdustScene
from chalkdust.scenes.components import make_component
from chalkdust.scenes.components.free_body_diagram import (
    ARROW_MIN_FRACTION,
    BODY_S,
    FORCE_S,
    HOLD_S,
    FreeBodyDiagram,
    _segment_hits_rect,
)
from chalkdust.scenes.regions import LayoutError, bbox, region_rect
from chalkdust.scenes.theme import DEFAULT
from chalkdust.validate.geometric import LayoutProbe, validate_beat

NAME = "FreeBodyDiagram"
EXAMPLES = FreeBodyDiagram.examples()
DRAFT_FPS = 15

# The invalid-LaTeX labels stress() carries (as forces[1]); they refuse rather
# than draw, so the drawing tests below leave those cases out.
BAD_LATEX = (r"\frac{m", r"\notacommand{g}", r"\,")


def _is_bad_latex(params: dict) -> bool:
    return any(f["label"] in BAD_LATEX for f in params["forces"])


def _spec(params: dict) -> BeatSpec:
    return BeatSpec(id="b01", narration="placeholder narration",
                    component=NAME, params=params)


def _probe(params: dict, duration: float = 8.0) -> LayoutProbe:
    probe = LayoutProbe(make_component(NAME, params), duration=duration)
    probe.construct()
    return probe


def _parts(scene) -> tuple[list[Arrow], list[MathTex]]:
    """The force arrows and their labels, in force order."""
    arrows = [m for m in scene.mobjects if isinstance(m, Arrow)]
    labels = [m for m in scene.mobjects if isinstance(m, MathTex)]
    return arrows, labels


# --- timing (D-002) -----------------------------------------------------------


def _draft(tmp_path, **extra) -> dict:
    return {"pixel_width": 854, "pixel_height": 480, "frame_rate": DRAFT_FPS,
            "media_dir": str(tmp_path), "disable_caching": True,
            "verbosity": "WARNING", "progress_bar": "none", **extra}


@pytest.mark.parametrize("factor", [0.5, 3.0], ids=["short", "long"])
@pytest.mark.parametrize("params", EXAMPLES, ids=[f"ex{i}" for i in range(len(EXAMPLES))])
def test_timing_renders_exactly_the_beats_frames(params, factor, tmp_path):
    # Narration far shorter and far longer than the animation wants: either
    # way the beat lasts exactly ceil(audio * fps) frames (D-002). Counted on
    # the renderer's own clock, which advances one frame time per frame it
    # renders, at draft frame rate (dry run: nothing is encoded).
    budget = make_component(NAME, params).min_seconds() * factor
    with tempconfig(_draft(tmp_path, dry_run=True)):
        scene = ChalkdustScene(make_component(NAME, params), duration=budget)
        scene.render()
    frames = scene.renderer.time * DRAFT_FPS
    assert frames == pytest.approx(round(frames), abs=1e-6)
    assert round(frames) == scene.beat_frames == math.ceil(round(budget * DRAFT_FPS, 6))


def test_draft_render_has_exactly_the_beats_frames(tmp_path):
    # One real encode, counted by ffprobe: 5.03 s of audio at 15 fps is
    # ceil(75.45) = 76 frames, however the six segments split them.
    with tempconfig(_draft(tmp_path, output_file="fbd_frames")):
        scene = ChalkdustScene(make_component(NAME, EXAMPLES[0]), duration=5.03)
        scene.render()
        movie = scene.renderer.file_writer.movie_file_path
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames", "-of", "json", str(movie)],
        capture_output=True, text=True, check=True).stdout
    assert int(json.loads(out)["streams"][0]["nb_read_frames"]) == 76


def test_min_seconds_is_sum_of_segment_minimums():
    comp = make_component(NAME, EXAMPLES[0])   # four forces
    assert comp.min_seconds() == pytest.approx(BODY_S + 4 * FORCE_S + HOLD_S)


def test_latex_strings_lists_every_label_in_order():
    comp = make_component(NAME, EXAMPLES[0])
    assert comp.latex_strings() == [f["label"] for f in EXAMPLES[0]["forces"]]


# --- schema ---------------------------------------------------------------------


def _one(**force) -> dict:
    return {"body": "m", "forces": [{"label": "F", "angle": 0, **force}]}


@pytest.mark.parametrize("params", [
    {"body": "m", "forces": []},
    {"body": "   ", "forces": [{"label": "F", "angle": 0}]},
    {"body": "m", "forces": [{"label": " ", "angle": 0}]},
    {"body": "m", "forces": [{"label": "F", "angle": 0}] * 13},
    {"body": "m", "forces": [{"label": "F"}]},
    _one(magnitude=0),
    _one(magnitude=-3),
    _one(magnitude=float("inf")),
    _one(magnitude=float("nan")),
    _one(angle=400),
    _one(angle=float("nan")),
    _one(colour="red"),
], ids=["no-forces", "blank-body", "blank-label", "too-many-forces",
        "no-angle", "zero-magnitude", "negative-magnitude", "inf-magnitude",
        "nan-magnitude", "angle-out-of-range", "nan-angle", "unknown-force-param"])
def test_schema_rejects(params):
    with pytest.raises(ValidationError):
        FreeBodyDiagram(params)


def test_magnitude_defaults_to_equal_arrows():
    probe = _probe({"body": "m", "forces": [{"label": "A", "angle": 0},
                                             {"label": "B", "angle": 180}]})
    a, b = _parts(probe)[0]
    assert a.get_length() == pytest.approx(b.get_length())


# --- typed LaTeX failure -----------------------------------------------------------


@pytest.mark.parametrize(
    "params", [p for p in FreeBodyDiagram.stress() if _is_bad_latex(p)],
    ids=["unclosed-brace", "undefined-command", "draws-nothing"])
def test_invalid_latex_is_a_typed_refusal(params):
    # theme.math is the one LaTeX path: a label that does not compile, or
    # compiles to nothing, refuses as invalid_latex naming its spec field.
    with pytest.raises(LayoutError) as exc:
        _probe(params)
    assert exc.value.kind == "invalid_latex"
    assert "forces[1].label" in str(exc.value)
    # And the probe reports it as that kind, not as a crash.
    assert validate_beat(_spec(params)).kinds() == {"invalid_latex"}


# --- arrow sizing ----------------------------------------------------------------


def test_lengths_proportional_above_the_clamp():
    # Example 0: mg = N = 19.6 (the largest), T = 10, f_k = 4.
    arrows, _ = _parts(_probe(EXAMPLES[0]))
    mg, n, t, fk = (a.get_length() for a in arrows)
    assert n == pytest.approx(mg)
    assert t / mg == pytest.approx(10 / 19.6)
    # 4 / 19.6 = 0.20 is under the floor, so f_k is held there.
    assert fk / mg == pytest.approx(ARROW_MIN_FRACTION)


def test_tiny_force_held_at_floor_and_huge_force_at_ceiling():
    params = {"body": "grain", "forces": [{"label": "a", "angle": 270, "magnitude": 1e-9},
                                          {"label": "b", "angle": 90, "magnitude": 1e9}]}
    tiny, huge = (a.get_length() for a in _parts(_probe(params))[0])
    assert tiny / huge == pytest.approx(ARROW_MIN_FRACTION)


def test_every_arrowhead_is_the_same_size():
    # Manim's default thins short arrows; length must be the only encoding.
    arrows, _ = _parts(_probe(EXAMPLES[0]))
    tips = [a.get_tip().height for a in arrows if abs(a.get_unit_vector()[0]) < 1e-6]
    sides = [a.get_tip().width for a in arrows if abs(a.get_unit_vector()[1]) < 1e-6]
    assert tips and sides
    assert tips + sides == pytest.approx([tips[0]] * len(tips + sides))
    assert {a.get_stroke_width() for a in arrows} == {arrows[0].get_stroke_width()}


# --- clustering and label placement -------------------------------------------------


def test_parallel_forces_drawn_side_by_side_from_the_face():
    params = {"body": "crate",
              "forces": [{"label": f"F_{i}", "angle": 270} for i in range(4)]}
    probe = _probe(params)
    arrows, _ = _parts(probe)
    body = next(m for m in probe.mobjects if getattr(m, "_chalk_label", "") == "body")
    xs = sorted(a.get_start()[0] for a in arrows)
    assert all(b - a > 0.1 for a, b in zip(xs, xs[1:]))
    box = bbox(body)
    for a in arrows:
        assert a.get_start()[1] == pytest.approx(box.bottom, abs=1e-6)
        assert box.left < a.get_start()[0] < box.right


def test_directions_either_side_of_zero_cluster_together():
    # 355 and 5 degrees are 10 degrees apart, not 350.
    params = {"body": "puck", "forces": [{"label": "a", "angle": 355},
                                         {"label": "b", "angle": 5}]}
    a, b = _parts(_probe(params))[0]
    # Side by side, in angle order: 355 below, 5 above, never crossing.
    assert a.get_start()[1] < b.get_start()[1]
    assert a.get_end()[1] < b.get_end()[1]


def _fixtures_that_fit():
    out = [pytest.param(p, id=f"ex{i}") for i, p in enumerate(EXAMPLES)]
    for i, p in enumerate(FreeBodyDiagram.stress()):
        if not _is_bad_latex(p):
            out.append(pytest.param(p, id=f"stress{i}"))
    return out


@pytest.mark.parametrize("params", _fixtures_that_fit())
def test_labels_clear_every_other_arrow_and_each_other(params):
    try:
        probe = _probe(params)
    except LayoutError as exc:
        assert exc.kind == "overflow"   # refused cleanly; nothing was drawn
        return
    arrows, labels = _parts(probe)
    assert len(arrows) == len(labels) == len(params["forces"])
    for i, tex in enumerate(labels):
        box = bbox(tex)
        for j, arrow in enumerate(arrows):
            if j != i:
                assert not _segment_hits_rect(arrow.get_start(), arrow.get_end(), box), (
                    f"label[{i}] touches force[{j}]")
        for j, other in enumerate(labels[i + 1:], start=i + 1):
            assert not box.intersects(bbox(other)), f"label[{i}] overlaps label[{j}]"


def test_triple_volume_refuses_as_overflow():
    # stress()[0] is 12 long labels round a long-named body: too dense to stay
    # legible, and it must say so as overflow (split the beat), not crash.
    with pytest.raises(LayoutError) as exc:
        _probe(FreeBodyDiagram.stress()[0])
    assert exc.value.kind == "overflow"


# --- carry-in artifact (SCENE_SPEC.md §6) ---------------------------------------------


def _recipe(params: dict) -> ArtifactRecipe:
    return ArtifactRecipe(name="fbd", producer=NAME, params=params)


def test_artifact_rebuild_is_deterministic():
    a, b = (build_artifact(_recipe(EXAMPLES[0]), DEFAULT) for _ in range(2))
    pa = [m.points for m in a.family_members_with_points()]
    pb = [m.points for m in b.family_members_with_points()]
    assert len(pa) == len(pb) > 0
    assert all(np.array_equal(x, y) for x, y in zip(pa, pb))


def test_carried_artifact_is_the_settled_diagram(hold_consumer):
    # A later beat carrying the diagram in sees the picture this beat ended
    # on: the same body, arrows and labels, in the same place on the stage.
    # Carried into _Hold, a consumer: a beat that carries an artifact in
    # without acting on it must leave it a free STAGE region (D-G4c-1).
    params = EXAMPLES[2]
    video = VideoSpec(video_id="v", beats=(
        BeatSpec(id="b01", narration="placeholder narration", component=NAME,
                 params=params, registers="fbd"),
        BeatSpec(id="b02", narration="placeholder narration",
                 component=hold_consumer, params={"target_id": "fbd"},
                 carry_in=["fbd"]),
    ))
    recipes = resolve_carry_in(video)["b02"]
    assert recipes == (_recipe(params),)

    consumer = LayoutProbe(beat_component(video.beats[1], recipes), duration=4.0)
    consumer.construct()
    artifact = carried(consumer, "fbd")
    assert region_rect(Region.STAGE).contains(bbox(artifact))

    settled = {m._chalk_label: m for m in _probe(params).mobjects
               if any(sm.has_points() for sm in m.get_family())}
    parts = {m._chalk_label: m for m in artifact.submobjects}
    assert parts.keys() == settled.keys()
    for name, want in settled.items():
        g, w = bbox(parts[name]), bbox(want)
        assert (g.x, g.y, g.width, g.height) == pytest.approx(
            (w.x, w.y, w.width, w.height), abs=1e-6), name

    # And the consuming beat validates clean with it on screen.
    report = validate_beat(video.beats[1], duration=4.0, recipes=recipes)
    assert report.ok, f"\n{report}"
