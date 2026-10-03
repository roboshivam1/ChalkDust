"""GeometryConstruct: timing, schema refusals, intersection semantics, layout.

test_layout.py already runs every example and stress fixture through the
probe; these pin the behaviour specific to this component.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import asdict

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
from chalkdust.core.models import BeatSpec, Quality, Region, VideoSpec
from chalkdust.render.worker import TIERS
from chalkdust.scenes.base import ChalkdustScene, frames_covering
from chalkdust.scenes.components.geometry_construct import (
    HOLD_MIN,
    INTRO_MIN,
    STEP_MIN,
    GeometryConstruct,
    GeometryConstructParams,
    _dist_to_segment,
    _resolve,
)
from chalkdust.scenes.regions import bbox, region_rect
from chalkdust.scenes.theme import DEFAULT, resolve_fonts
from chalkdust.validate.geometric import LayoutProbe, validate_beat

DRAFT = TIERS[Quality.DRAFT]
FPS = DRAFT.frame_rate


def _pt(name, x, y, **kw):
    return {"kind": "point", "name": name, "at": [x, y], **kw}


def _euclid(scale=1.0, offset=0.0):
    """Euclid I.1 with its local frame scaled and shifted."""
    return {
        "shapes": [_pt("A", offset, offset), _pt("B", offset + scale, offset),
                   {"kind": "segment", "ends": ["A", "B"]}],
        "construction": [
            {"kind": "circle", "center": "A", "through": "B", "id": "cA"},
            {"kind": "circle", "center": "B", "through": "A", "id": "cB"},
            {"kind": "intersect", "of": ["cA", "cB"], "names": ["C", "D"]},
        ],
    }


def _validate(params):
    return validate_beat(BeatSpec(id="b01", narration="placeholder",
                                  component="GeometryConstruct", params=params))


# --- timing (D-002) ------------------------------------------------------------


class _FrameClock(ChalkdustScene):
    """The real scene with rendering skipped, counting the frames each play()
    and wait() would write. wait() is a play() of a Wait, and the base hands
    Manim (n + 0.5) / fps for an n-frame run time (scenes/base.py), so a
    play writes int(duration * fps) frames."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.frames = 0

    def play(self, *args, **kwargs) -> None:
        super().play(*args, **kwargs)
        self.frames += int(self.duration * self.fps)


def _clocked(params, duration, tmp_path) -> _FrameClock:
    with tempconfig({"dry_run": True, "media_dir": str(tmp_path),
                     "frame_rate": FPS, "verbosity": "WARNING"}):
        scene = _FrameClock(GeometryConstruct(params), duration=duration,
                            skip_animations=True)
        scene.render()
    return scene


@pytest.mark.parametrize("factor", [0.5, 3.0], ids=["short", "long"])
@pytest.mark.parametrize("index", range(len(GeometryConstruct.examples())))
def test_frames_equal_beat_frames(index, factor, tmp_path):
    # Narration far shorter and far longer than the build wants: either way
    # the beat lasts exactly ceil(audio * fps) frames (frames_covering: float
    # noise like 33.8 * 15 = 507.00000000000006 is not a frame), all budget()ed.
    params = GeometryConstruct.examples()[index]
    budget = GeometryConstruct(params).min_seconds() * factor
    scene = _clocked(params, budget, tmp_path)
    assert scene.frames == scene.beat_frames == frames_covering(budget, FPS)


def test_frames_equal_beat_frames_for_a_lone_point(tmp_path):
    # Minimal input: no given figure, one step -- no intro play.
    scene = _clocked({"construction": [_pt("A", 0, 0)]}, 0.4, tmp_path)
    assert scene.frames == scene.beat_frames == frames_covering(0.4, FPS)


def test_draft_render_is_exactly_the_beat(tmp_path):
    # One real 480p15 encode, counted by ffprobe: the clip is the audio
    # rounded up to whole frames, not a frame per play() more or less.
    params = GeometryConstruct.examples()[1]
    audio = GeometryConstruct(params).min_seconds() * 1.3
    with tempconfig({**asdict(DRAFT), "media_dir": str(tmp_path),
                     "disable_caching": True, "progress_bar": "none",
                     "verbosity": "WARNING", "output_file": "frames"}):
        scene = ChalkdustScene(GeometryConstruct(params), duration=audio)
        scene.render()
        movie = scene.renderer.file_writer.movie_file_path
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames", "-of", "json", str(movie)],
        capture_output=True, text=True, check=True).stdout
    frames = int(json.loads(out)["streams"][0]["nb_read_frames"])
    assert frames == frames_covering(audio, FPS)


def test_min_seconds_is_sum_of_step_minimums():
    six_words = "one two three four five six"
    comp = GeometryConstruct({
        "shapes": [_pt("A", 0, 0), _pt("B", 1, 0)],
        "construction": [
            {"kind": "circle", "center": "A", "through": "B", "note": six_words},
            {"kind": "segment", "ends": ["A", "B"]},
        ],
    })
    # A captioned step lasts long enough to read its caption (3 words/s).
    expected = INTRO_MIN + max(STEP_MIN["circle"], 2.0) + STEP_MIN["segment"] + HOLD_MIN
    assert comp.min_seconds() == pytest.approx(expected)


def test_compiles_no_latex():
    for params in GeometryConstruct.examples() + GeometryConstruct.stress():
        assert GeometryConstruct(params).latex_strings() == []


# --- regions -------------------------------------------------------------------


def test_lower_third_claimed_only_with_notes():
    bare = _euclid()
    noted = {**bare, "construction": [
        {**bare["construction"][0], "note": "Compass on A"},
        *bare["construction"][1:]]}
    assert GeometryConstruct(bare).regions() == {Region.STAGE}
    assert GeometryConstruct(noted).regions() == {Region.STAGE, Region.LOWER_THIRD}


def test_blank_note_is_no_note():
    # Whitespace is no caption: it must not claim LOWER_THIRD, weigh on the
    # timing, or build an empty Text over the figure (verify1 must-fix).
    bare = _euclid()
    blank = {**bare, "construction": [
        {**bare["construction"][0], "note": "   "},
        *bare["construction"][1:]]}
    assert GeometryConstructParams(**blank).construction[0].note is None
    component = GeometryConstruct(blank)
    assert component.regions() == {Region.STAGE}
    assert component.min_seconds() == GeometryConstruct(bare).min_seconds()
    report = _validate(blank)
    assert report.ok, f"\n{report}"


# --- schema refusals (rung 1) ----------------------------------------------------


@pytest.mark.parametrize("params,needle", [
    ({}, "at least one shape or step"),
    ({"shapes": [], "construction": []}, "at least one shape or step"),
    ({"shapes": [{"kind": "segment", "ends": ["A", "B"]}]}, "not defined yet"),
    ({"shapes": [_pt("A", 0, 0), _pt("A", 1, 0)]}, "already defined"),
    ({"shapes": [_pt("A", 0, 0), _pt("B", 0, 0),
                 {"kind": "segment", "ends": ["A", "B"]}]}, "zero length"),
    ({"shapes": [_pt("A", 0, 0), _pt("B", 1, 0), _pt("C", 2, 0),
                 {"kind": "polygon", "vertices": ["A", "B", "C"]}]}, "collinear"),
    # Two names on one spot make a zero-length polygon side, like a segment;
    # it used to reach build() and crash placing labels (rc2 must-fix).
    ({"shapes": [_pt("A", 0, 0), _pt("B", 0, 0), _pt("C", 1, 0), _pt("D", 0, 1),
                 {"kind": "polygon", "vertices": ["A", "B", "C", "D"]}]},
     "polygon side AB has zero length"),
    # ... including the closing side, last vertex back to the first.
    ({"shapes": [_pt("A", 0, 0), _pt("B", 1, 0), _pt("C", 0, 1), _pt("D", 0, 0),
                 {"kind": "polygon", "vertices": ["A", "B", "C", "D"]}]},
     "polygon side DA has zero length"),
    # (d) Notes are plain text and nothing here compiles maths: TeX in a
    # caption would draw as its source, so it is refused, malformed or not.
    ({"shapes": [_pt("A", 0, 0), _pt("B", 1, 0)],
      "construction": [{"kind": "circle", "center": "A", "through": "B",
                        "note": r"Radius $\frac{AB}{2}$ \unknown{"}]},
     "contains LaTeX markup"),
    ({"shapes": [_pt("A", 0, 0), _pt("B", 1, 0)],
      "construction": [{"kind": "segment", "ends": ["A", "B"],
                        "note": r"AB = \sqrt{2}"}]},
     "contains LaTeX markup"),
    ({"shapes": [_pt("A", 0, 0, note="given")]}, "notes belong to construction"),
    ({"shapes": [_pt("A", 0, 0)],
      "construction": [{"kind": "intersect", "of": ["x", "y"], "names": ["P"]}]},
     "no segment or circle with id 'x'"),
    # Circles of radius 1 centred 5 apart never meet.
    ({"shapes": [_pt("A", 0, 0), _pt("B", 1, 0), _pt("C", 5, 0), _pt("D", 6, 0)],
      "construction": [
          {"kind": "circle", "center": "A", "through": "B", "id": "c1"},
          {"kind": "circle", "center": "C", "through": "D", "id": "c2"},
          {"kind": "intersect", "of": ["c1", "c2"], "names": ["P"]}]},
     "meet in 0 new point(s)"),
])
def test_rejects_unresolvable_spec(params, needle):
    with pytest.raises(ValidationError, match=re.escape(needle)):
        GeometryConstructParams.model_validate(params)


@pytest.mark.parametrize("params", [
    # A 60-char point name: names are labels, so they are short by schema.
    {"shapes": [_pt("P" + "x" * 59, 0, 0)]},
    {"shapes": [_pt("A", float("inf"), 0)]},
    {"shapes": [_pt("A", 0, 0, colour="red")]},          # nested extra key
    {"shapes": [{"kind": "ellipse", "center": "A"}]},     # unknown kind
    {"construction": [_pt(f"P{i}", i, 0) for i in range(17)]},  # over the cap
])
def test_rejects_malformed_elements(params):
    with pytest.raises(ValidationError):
        GeometryConstructParams.model_validate(params)


def test_error_names_the_failing_step():
    params = _euclid()
    params["construction"][2] = {"kind": "intersect", "of": ["cA", "cX"],
                                 "names": ["C"]}
    with pytest.raises(ValidationError, match=r"construction\[2\] \(intersect\)"):
        GeometryConstructParams.model_validate(params)


# --- intersection semantics -------------------------------------------------------


def test_intersection_names_are_assigned_top_first():
    pts = _resolve(GeometryConstructParams.model_validate(_euclid()))
    assert pts["C"][1] > 0 > pts["D"][1]


def test_intersection_skips_already_named_points():
    # Angle bisector: circles at R and S meet at O and T. O is named, so the
    # single name goes to T even though O sorts first (lower-left vs upper).
    angle = GeometryConstruct.examples()[2]
    pts = _resolve(GeometryConstructParams.model_validate(angle))
    assert pts["T"] == pytest.approx([2.25, 1.299038], abs=1e-6)


# --- layout ------------------------------------------------------------------------


def _figure_box(params):
    probe = LayoutProbe(GeometryConstruct(params), duration=8.0)
    probe.construct()
    (fig,) = [m for m in probe.mobjects if getattr(m, "_chalk_label", "") == "figure"]
    return bbox(fig)


def test_local_frame_is_unitless():
    # The model picks the shape; the component picks the size and place.
    small = _figure_box(_euclid(scale=1e-3))
    large = _figure_box(_euclid(scale=1e3, offset=1e6))
    for attr in ("x", "y", "width", "height"):
        assert getattr(small, attr) == pytest.approx(getattr(large, attr), abs=1e-6)


@pytest.mark.parametrize("params,kind", [
    # Two points 1e-4 apart beside a segment of length 10: one visible dot.
    ({"shapes": [_pt("A", 0, 0), _pt("B", 1e-4, 0), _pt("C", 10, 0),
                 {"kind": "segment", "ends": ["A", "C"]}]}, "illegible"),
    # A circle that is a speck beside a long segment.
    ({"shapes": [_pt("A", 0, 0), _pt("B", 1000, 0),
                 {"kind": "segment", "ends": ["A", "B"]},
                 _pt("C", 500, 300), _pt("D", 500.01, 300)],
      "construction": [{"kind": "circle", "center": "C", "through": "D"}]},
     "illegible"),
    # Captions three lines deep cannot sit in LOWER_THIRD legibly.
    (GeometryConstruct.stress()[0], "overflow"),
])
def test_refuses_unreadable_figures(params, kind):
    report = _validate(params)
    assert report.kinds() == {kind}, f"\n{report}"


def test_distance_to_a_degenerate_segment_is_distance_to_its_point():
    # Label placement measures clearance from every side; a side of zero
    # length must read as a point, not nan (nan scores pick no direction).
    a = np.array([1.0, 1.0, 0.0])
    assert _dist_to_segment(np.array([4.0, 5.0, 0.0]), a, a.copy()) == 5.0


# --- continuity (SCENE_SPEC.md §6) -------------------------------------------------


EUCLID = GeometryConstruct.examples()[0]
# As a scene holds it: fonts resolved to what is installed, as the probe does.
THEME = resolve_fonts(DEFAULT)


def _carry_video() -> VideoSpec:
    return VideoSpec(video_id="v", beats=(
        BeatSpec(id="b01", narration="placeholder narration",
                 component="GeometryConstruct", params=EUCLID,
                 registers="euclid"),
        BeatSpec(id="b02", narration="placeholder narration",
                 component="BulletReveal", params={"items": ["All sides equal"]},
                 carry_in=["euclid"]),
    ))


def _recipe() -> ArtifactRecipe:
    return ArtifactRecipe(name="euclid", producer="GeometryConstruct",
                          params=EUCLID)


def _pieces(group) -> dict[str, list[np.ndarray]]:
    return {p._chalk_label: [m.points for m in p.family_members_with_points()]
            for p in group.submobjects}


def test_artifact_builder_is_registered():
    (recipe,) = resolve_carry_in(_carry_video())["b02"]
    assert recipe == _recipe()


def test_artifact_is_the_figure_the_beat_ends_on():
    # Same pieces at the same points as the producing beat's last frame;
    # the step captions are not part of the figure.
    probe = LayoutProbe(GeometryConstruct(EUCLID), duration=8.0)
    probe.construct()
    (figure,) = [m for m in probe.mobjects
                 if getattr(m, "_chalk_label", "") == "figure"]
    want, got = _pieces(figure), _pieces(build_artifact(_recipe(), THEME))
    assert sorted(got) == sorted(want)
    for name, points in want.items():
        assert len(got[name]) == len(points), name
        assert all(np.allclose(g, w) for g, w in zip(got[name], points)), name


def test_artifact_rebuild_is_deterministic():
    a, b = build_artifact(_recipe(), THEME), build_artifact(_recipe(), THEME)
    pa = [m.points for m in a.family_members_with_points()]
    pb = [m.points for m in b.family_members_with_points()]
    assert len(pa) == len(pb) > 0
    assert all(np.array_equal(x, y) for x, y in zip(pa, pb))


def test_artifact_layers_by_order_not_z_index():
    # z-index is scene-wide: a carried point at z=2 would draw over the
    # consuming beat's content. Fill, strokes, points -- by submobject order.
    art = build_artifact(_recipe(), THEME)
    assert {m.z_index for m in art.get_family()} == {0}
    kinds = [p._chalk_label.split()[0] for p in art.submobjects]
    rank = {"polygon": 0, "segment": 1, "circle": 1, "point": 2}
    assert [rank[k] for k in kinds] == sorted(rank[k] for k in kinds)


def test_carry_in_beat_builds_with_the_figure():
    video = _carry_video()
    recipes = resolve_carry_in(video)["b02"]
    report = validate_beat(video.beats[1], recipes=recipes)
    assert report.ok, f"\n{report}"
    probe = LayoutProbe(beat_component(video.beats[1], recipes), duration=4.0,
                        strict=False)
    probe.construct()
    figure = carried(probe, "euclid")
    assert region_rect(Region.STAGE).contains(bbox(figure))
    assert probe.layout_warnings == []
