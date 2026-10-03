"""ZoomHighlight: the behaviours the registry-wide layout tests cannot pin.

ZoomHighlight acts on a carried artifact (SCENE_SPEC.md §6). Its examples()
and stress() name a carried BulletReveal list; fixture_carry_in() gives the
registry walks (test_layout, test_snapshots, the semantic TestLibrary) that
list's recipe. Here the cases are built the way the compiler builds them --
b01 registers the list, b02 carries it in, continuity.beat_component wraps
the zoom -- with BulletReveal's own artifact builder, and the pipeline
renders one end to end. No LaTeX is involved.
"""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
from manim import Group, tempconfig
from manim.animation.animation import prepare_animation
from pydantic import ValidationError

from chalkdust import continuity, pipeline
from chalkdust.continuity import beat_component, resolve_carry_in
from chalkdust.core.models import BeatSpec, CarryInError, Quality, Region, VideoSpec, VoiceConfig
from chalkdust.render.worker import TIERS
from chalkdust.scenes.base import ChalkdustScene, frames_covering
from chalkdust.scenes.components import get_component, make_component
from chalkdust.scenes.components.zoom_highlight import (
    MAX_ZOOM,
    MIN_ZOOM,
    ZoomHighlight,
    _zoom_factor,
)
from chalkdust.scenes.regions import LayoutError, bbox, region_rect, smallest_font_size
from chalkdust.speech import tts
from chalkdust.speech.base import run
from chalkdust.validate.geometric import LayoutProbe
from chalkdust.validate.semantic import validate_semantic

NAME = "ZoomHighlight"
EXAMPLES = ZoomHighlight.examples()
STRESS = ZoomHighlight.stress()
CLEAN_REFUSALS = {"overflow", "illegible", "invalid_latex",
                  "unrenderable_text"}  # as in tests/test_layout.py
# stress() (d): callouts the theme font cannot draw (emoji, RTL, CJK, zero-width).
UNDRAWABLE = [c for c in STRESS if not c["callout"].isascii()]
DRAFT = TIERS[Quality.DRAFT]
DRAFT_FPS = DRAFT.frame_rate
ARTIFACT = "chain_causes"  # EXAMPLES[0]'s and EXAMPLES[2]'s target


def _video(case: dict, carry_in: tuple[str, ...] | None = None,
           **zoom_params) -> VideoSpec:
    """b01 registers the case's carried target; b02 zooms into it."""
    (recipe,) = ZoomHighlight.fixture_carry_in(case)
    return VideoSpec(video_id="v", beats=[
        BeatSpec(id="b01", narration="placeholder narration", component=recipe.producer,
                 params=recipe.params, registers=recipe.name),
        BeatSpec(id="b02", narration="placeholder narration", component=NAME,
                 params={**case, **zoom_params},
                 carry_in=list((recipe.name,) if carry_in is None else carry_in)),
    ])


def _component(case: dict, **zoom_params):
    video = _video(case, **zoom_params)
    return beat_component(video.beats[1], resolve_carry_in(video)["b02"])


def _probe(case: dict, duration: float = 8.0, **zoom_params) -> LayoutProbe:
    probe = LayoutProbe(_component(case, **zoom_params), duration=duration, strict=False)
    probe.construct()
    return probe


def _by_label(scene) -> dict[str, object]:
    return {getattr(m, "_chalk_label", None): m for m in scene.mobjects}


def _ids(cases, prefix):
    return [f"{prefix}{i}" for i in range(len(cases))]


# --- fixtures -----------------------------------------------------------------


@pytest.mark.parametrize("case", EXAMPLES + STRESS,
                         ids=_ids(EXAMPLES, "ex") + _ids(STRESS, "stress"))
def test_every_fixture_carries_its_target_in(case):
    # The registry walks build each case with exactly this recipe carried in;
    # it must be the target the case zooms into, rebuilt by BulletReveal's own
    # registered builder (nothing lent), and resolvable as a real 2-beat spec.
    (recipe,) = ZoomHighlight.fixture_carry_in(case)
    assert (recipe.name, recipe.producer) == (case["target_id"], "BulletReveal")
    assert continuity._BUILDERS["BulletReveal"].__module__ ==         "chalkdust.scenes.components.bullet_reveal"
    assert resolve_carry_in(_video(case))["b02"] == (recipe,)


def test_target_sits_where_the_producing_beat_left_it():
    # No jump at the b01 -> b02 cut: every row of the carried list (at full
    # strength) is where b01's BulletReveal settled it, so the focus comes up
    # in place and the lens grows out of the row the viewer was reading.
    video = _video(EXAMPLES[0])
    produced = LayoutProbe(make_component("BulletReveal", video.beats[0].params),
                           duration=8.0, strict=False)
    produced.construct()
    shown = _by_label(produced)["bullets"]
    target = _by_label(_probe(EXAMPLES[0]))[f"carried[{ARTIFACT}]"]
    assert len(target.submobjects) == len(shown.submobjects)
    for row, before in zip(target.saved_state.submobjects, shown.submobjects):
        np.testing.assert_allclose(row.get_center(), before.get_center(), atol=1e-6)


# --- layout -------------------------------------------------------------------


@pytest.mark.parametrize("case", EXAMPLES, ids=_ids(EXAMPLES, "ex"))
def test_examples_validate_clean(case):
    assert _probe(case).layout_warnings == []


@pytest.mark.parametrize("case", STRESS, ids=_ids(STRESS, "stress"))
def test_stress_fits_or_refuses_cleanly(case):
    try:
        probe = _probe(case)
    except LayoutError as exc:
        assert exc.kind in CLEAN_REFUSALS, exc
        return
    assert {kind for kind, _ in probe.layout_warnings} <= CLEAN_REFUSALS


def test_callout_at_three_times_volume_refuses_with_overflow():
    with pytest.raises(LayoutError) as exc_info:
        _probe(STRESS[0])
    assert exc_info.value.kind == "overflow"


@pytest.mark.parametrize("case", UNDRAWABLE, ids=["emoji", "rtl", "cjk", "zero-width"])
def test_undrawable_callout_refuses_as_unrenderable_text_naming_the_callout(case):
    # The theme's glyph guard fires in body_text, before fit_to_region or the
    # lens exist. Before it, an inkless callout was a zero-size box at the
    # origin -- inside the lens -- and refused as a misleading "overlap".
    with pytest.raises(LayoutError, match=r"^callout ") as exc_info:
        _probe(case)
    assert exc_info.value.kind == "unrenderable_text"


def test_private_use_callout_is_refused_at_rung_one():
    # U+E000 draws differently on every machine, so the spec is refused
    # before ZoomHighlight (or any fixture walk) sees it.
    with pytest.raises(ValidationError, match=r"private-use code point U\+E000"):
        _video(EXAMPLES[0], callout="key \ue000 here")


def test_lens_magnifies_focus_and_recedes_the_rest():
    # EXAMPLES[1]: two short rows of three, magnified on a lens.
    probe = _probe(EXAMPLES[1])
    mobs = _by_label(probe)
    target, lens, callout = mobs["carried[hash_steps]"], mobs["zoom lens"], mobs["callout"]

    def peak(m):
        return max(x.get_fill_opacity() for x in m.family_members_with_points())

    assert peak(target[1]) == pytest.approx(1.0)
    assert peak(target[2]) == pytest.approx(1.0)
    assert peak(target[0]) < 1 - continuity.DIM_DARKNESS
    # As far as the lens may magnify the two rows (BulletReveal's pitch makes
    # their height, not MAX_ZOOM, the limit), and magnified text is tracked at
    # its magnified size, so legibility is judged on what the viewer sees.
    zoom = _zoom_factor(bbox(Group(target[1], target[2])), region_rect(Region.STAGE))
    assert MIN_ZOOM <= zoom <= MAX_ZOOM
    assert lens[1].height == pytest.approx(bbox(Group(target[1], target[2])).height * zoom)
    assert smallest_font_size(lens) == pytest.approx(smallest_font_size(target) * zoom)
    assert region_rect(Region.STAGE).contains(bbox(lens))
    assert region_rect(Region.LOWER_THIRD).contains(bbox(callout))


@pytest.mark.parametrize("case", [EXAMPLES[1], STRESS[2]], ids=["ex1", "stress2"])
def test_lens_settles_beside_a_narrow_target_hiding_no_unfocused_part(case):
    # EXAMPLES[1]: over its focus (rows 1-2 of three short rows) the lens card
    # is taller than the list and hid "hash(key)" whole. STRESS[2]: over row 0
    # of two it hid half of "mod 8". STAGE has room beside the narrow list, so
    # the lens settles there, clear of the whole target (never half over the
    # rows it magnifies), level with its focus, still magnifying.
    mobs = _by_label(_probe(case))
    target, lens = mobs[f"carried[{case['target_id']}]"], mobs["zoom lens"]
    focus = Group(*(target[i] for i in case["parts"]))
    assert not bbox(lens).intersects(bbox(target))
    assert bbox(lens).left > bbox(target).right
    assert bbox(lens).y == pytest.approx(bbox(focus).y)
    assert lens[1].height >= bbox(focus).height * MIN_ZOOM
    assert region_rect(Region.STAGE).contains(bbox(lens))


def test_lens_stays_over_the_focus_when_that_hides_nothing():
    # EXAMPLES[0]: a full-width middle row. Its lens clears both neighbours
    # in place, and there is no room beside the list, so it magnifies in place.
    mobs = _by_label(_probe(EXAMPLES[0]))
    target, lens = mobs[f"carried[{ARTIFACT}]"], mobs["zoom lens"]
    np.testing.assert_allclose(lens.get_center(), target[1].get_center(), atol=1e-6)
    assert all(bbox(lens).overlap_area(bbox(target[i])) == 0.0 for i in (0, 2))


def test_whole_target_is_restored_and_framed():
    # EXAMPLES[2]: no parts -- the target fills STAGE's width, so it is
    # framed rather than magnified, and comes back to full strength.
    mobs = _by_label(_probe(EXAMPLES[2]))
    target = mobs[f"carried[{ARTIFACT}]"]
    assert "zoom lens" not in mobs
    assert bbox(mobs["focus frame"]).contains(bbox(target))
    assert max(x.get_fill_opacity() for x in target.family_members_with_points()) \
        == pytest.approx(1.0)


# --- hooks ----------------------------------------------------------------------


def test_semantic_hooks():
    comp = make_component(NAME, EXAMPLES[0])
    assert comp.min_seconds() == pytest.approx(3.5)  # 7 weight units x 0.5 s
    assert comp.latex_strings() == []                # the callout is plain Text
    assert comp.carried_targets() == [ARTIFACT]


# --- timing (D-002) -----------------------------------------------------------


class _Clock(LayoutProbe):
    """A probe that also adds up the time every play() and wait() asks for. A
    play() with no explicit run_time counts at its animations' own default,
    so a forgotten run_time shows up as a frame mismatch."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.segments: list[float] = []

    def play(self, *animations, **kwargs) -> None:  # type: ignore[override]
        run_time = kwargs.get("run_time")
        if run_time is None:
            run_time = max(prepare_animation(a).run_time for a in animations)
        self.segments.append(run_time)
        super().play(*animations, **kwargs)

    def wait(self, duration: float = 1.0, *args, **kwargs) -> None:  # type: ignore[override]
        self.segments.append(duration)


@pytest.mark.parametrize("factor", [0.5, 3.0], ids=["short", "long"])
@pytest.mark.parametrize("case", EXAMPLES, ids=_ids(EXAMPLES, "ex"))
def test_clocked_frames_equal_beat_frames(case, factor, tmp_path):
    """Narration far shorter and far longer than the zoom wants: every play
    and wait is a whole number of draft frames, and together they are exactly
    the beat's ceil(audio * fps) frames. The carried target appears at t=0
    without animation, so it costs nothing."""
    budget = make_component(NAME, case).min_seconds() * factor
    with tempconfig({"frame_rate": DRAFT_FPS, "media_dir": str(tmp_path)}):
        clock = _Clock(_component(case), duration=budget, strict=False)
    clock.construct()
    frames = [t * DRAFT_FPS for t in clock.segments]
    assert all(f == pytest.approx(round(f), abs=1e-9) for f in frames)
    assert sum(round(f) for f in frames) == clock.beat_frames == \
        math.ceil(budget * DRAFT_FPS)


def _count_frames(movie: Path) -> int:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames", "-of", "json", str(movie)],
        capture_output=True, text=True, check=True).stdout
    return int(json.loads(out)["streams"][0]["nb_read_frames"])


@pytest.mark.parametrize("factor", [0.5, 3.0], ids=["short", "long"])
def test_draft_render_frames_equal_audio_rounded_up(tmp_path, factor):
    """A real 480p15 render of the lens case (the reveal is a Transform from
    a scaled copy), frames counted by ffprobe."""
    case = EXAMPLES[1]
    budget = make_component(NAME, case).min_seconds() * factor
    with tempconfig({**asdict(DRAFT), "media_dir": str(tmp_path),
                     "disable_caching": True, "progress_bar": "none",
                     "verbosity": "WARNING", "output_file": "zoom"}):
        scene = ChalkdustScene(_component(case), duration=budget)
        scene.render()
        movie = scene.renderer.file_writer.movie_file_path
    assert _count_frames(movie) == math.ceil(budget * DRAFT_FPS)


# --- typed refusals -------------------------------------------------------------


def test_target_not_carried_is_typed_error_at_build():
    # The beat carries chain_causes but zooms into something else.
    with pytest.raises(CarryInError) as exc_info:
        _probe(EXAMPLES[0], target_id="bucket_array")
    assert exc_info.value.name == "bucket_array"


def test_target_not_carried_is_refused_by_the_semantic_rung():
    # Registered by b01, but b02 does not carry it in: refused before build.
    video = _video(EXAMPLES[0], carry_in=())
    report = validate_semantic(video.beats[1], registered_artifacts={ARTIFACT},
                               duration=8.0)
    assert report.kinds() == {"carry_in"}


def test_unregistered_target_fails_spec_validation():
    with pytest.raises(ValidationError) as exc_info:
        _video(EXAMPLES[0], carry_in=("nope",), target_id="nope")
    err = exc_info.value.errors()[0]["ctx"]["error"]
    assert isinstance(err, CarryInError) and err.name == "nope"


def test_part_out_of_range_is_typed_error():
    # Three rows: indices 0..2. Only the rebuilt artifact knows that.
    with pytest.raises(CarryInError, match=r"part\(s\) \[3\]") as exc_info:
        _probe(EXAMPLES[0], parts=[1, 3])
    assert exc_info.value.name == ARTIFACT


@pytest.mark.parametrize("params", [
    {"target_id": ARTIFACT},                                   # no callout
    {"target_id": ARTIFACT, "callout": ""},
    {"target_id": ARTIFACT, "callout": "   "},
    {"target_id": "", "callout": "x"},
    {"callout": "x"},                                          # no target
    {"target_id": ARTIFACT, "callout": "x", "parts": []},
    {"target_id": ARTIFACT, "callout": "x", "parts": [-1]},
    {"target_id": ARTIFACT, "callout": "x", "parts": [1, 1]},
    {"target_id": ARTIFACT, "callout": "x", "definitely_not_a_real_param": 1},
], ids=["no-callout", "empty-callout", "blank-callout", "empty-target", "no-target",
        "empty-parts", "negative-part", "duplicate-parts", "unknown-param"])
def test_schema_rejects(params):
    with pytest.raises(ValidationError):
        get_component(NAME)(params)


# --- through the pipeline (SCENE_SPEC.md §6, D-004) ---------------------------


class _ToneTTS:
    """A tone whose length follows the narration's character count; needs only
    ffmpeg, so the pipeline test runs on any OS (as tests/test_pipeline.py)."""

    name = "zoom_tone"
    raw_format = "wav"
    default_voice = "tone"

    def synthesize(self, text: str, voice: VoiceConfig, out_path: Path) -> None:
        seconds = max(1.0, 0.06 * len(text))
        run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
             "-i", f"sine=frequency=220:duration={seconds:.3f}", str(out_path)])


def _spec_file(root: Path, video: VideoSpec) -> Path:
    data = video.model_dump(mode="json")
    data["voice"] = {"backend": _ToneTTS.name, "voice_id": "tone"}
    for beat, narration in zip(data["beats"], (
            "Three things make a hash table chain grow long.",
            "Look at the load factor: past three quarters full, chains grow "
            "faster than the table, and every lookup pays for it.")):
        beat["narration"] = narration
    path = root / "spec.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.fixture
def tone_tts(monkeypatch):
    monkeypatch.setitem(tts.BACKENDS, _ToneTTS.name, _ToneTTS())


def test_pipeline_renders_a_zoom_into_an_earlier_beats_artifact(tmp_path, tone_tts):
    """b01 (BulletReveal) registers chain_causes; b02 zooms into its second
    row. ZoomHighlight's build() raises unless the artifact is on screen, so a
    rendered b02 proves the render rebuilt b01's artifact for it. Each beat's
    clip is exactly ceil(audio * fps) frames."""
    path = _spec_file(tmp_path, _video(EXAMPLES[0]))
    result = pipeline.render(path, Quality.DRAFT, tmp_path / "zoom.mp4",
                             cache_dir=tmp_path / "cache", work_dir=tmp_path / "work")
    assert result.rebuilt == ["b01", "b02"]
    assert result.output.exists()
    clips = sorted((tmp_path / "cache" / "beats").glob("*.mp4"))
    assert sorted(_count_frames(c) for c in clips) == \
        sorted(frames_covering(b.duration, DRAFT_FPS) for b in result.beats)


def test_pipeline_refuses_a_target_the_beat_does_not_carry(tmp_path, tone_tts):
    path = _spec_file(tmp_path, _video(EXAMPLES[0], carry_in=()))
    with pytest.raises(pipeline.SemanticRefused) as exc_info:
        pipeline.validate(path, work_dir=tmp_path / "work")
    (report,) = exc_info.value.reports
    assert (report.beat_id, report.kinds()) == ("b02", {"carry_in"})


def test_pipeline_refuses_an_unregistered_target_as_spec_invalid(tmp_path, tone_tts):
    video = _video(EXAMPLES[0]).model_dump(mode="json")
    video["beats"][1]["params"]["target_id"] = "nope"
    video["beats"][1]["carry_in"] = ["nope"]
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(video), encoding="utf-8")
    with pytest.raises(pipeline.SpecInvalid, match="nope"):
        pipeline.validate(path, work_dir=tmp_path / "work")
