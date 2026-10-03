"""Callout: annotate a carried artifact (SCENE_SPEC.md §5, §6).

Callout's examples() and stress() each act on a carried BulletReveal list;
Callout.fixture_carry_in() gives the registry walks (test_layout,
test_snapshots, the semantic TestLibrary) the recipe for each case, so the
shared "validates clean / fits or refuses cleanly / never crashes" assertions
cover Callout there. This file pins what is specific to the component: the
fixture wiring, timing in whole frames, schema refusals, the typed carry-in
errors (at build, at rung 2 and through the pipeline), the label's place, and
the arrow-crossing refusal.
"""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import asdict
from pathlib import Path

import pytest
from manim import tempconfig
from pydantic import ValidationError

from chalkdust import continuity, pipeline
from chalkdust.continuity import ArtifactRecipe, beat_component, carried
from chalkdust.core.models import BeatSpec, CarryInError, Quality, Region, VideoSpec
from chalkdust.render.worker import TIERS
from chalkdust.scenes.base import ChalkdustScene
from chalkdust.scenes.components import make_component
from chalkdust.scenes.components.base import wrap_scale
from chalkdust.scenes.components.callout import Callout
from chalkdust.scenes.regions import LayoutError, bbox, region_rect, safe_area
from chalkdust.scenes.theme import DEFAULT, resolve_fonts
from chalkdust.validate.geometric import LayoutProbe, validate_beat
from chalkdust.validate.semantic import validate_semantic
from test_pipeline import FakeTTS, write_spec

NAME = "Callout"
EXAMPLES = Callout.examples()
STRESS = Callout.stress()
DRAFT = TIERS[Quality.DRAFT]


def _spec(params: dict, carry: list[str], bid: str = "b02") -> BeatSpec:
    return BeatSpec(id=bid, narration="placeholder narration",
                    component=NAME, params=params, carry_in=carry)


def _recipes(params: dict) -> list[ArtifactRecipe]:
    recipes = Callout.fixture_carry_in(params)
    if not recipes:  # an ad-hoc target: a three-row list named by target_id
        recipes = [ArtifactRecipe(name=params["target_id"], producer="BulletReveal",
                                  params={"items": ["Hash the key", "Find the bucket",
                                                    "Walk the chain"]})]
    return recipes


def _component(params: dict, carry: list[ArtifactRecipe] | None = None):
    recipes = _recipes(params) if carry is None else carry
    return beat_component(_spec(params, [r.name for r in recipes]), recipes)


def _probe(params: dict, duration: float = 8.0, **kw) -> LayoutProbe:
    probe = LayoutProbe(_component(params, **kw), duration=duration)
    probe.construct()
    return probe


def _labelled(probe, text: str):
    return next(m for m in probe.mobjects if getattr(m, "_chalk_label", None) == text)


class _FrameClock(LayoutProbe):
    """A probe that counts the frames every play() and wait() asks for, at
    the scene's fps -- the clip's length without encoding it. Each run time
    must already be a whole number of frames (budget() hands out nothing
    else); a private rounding or a hardcoded run time shows up here."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.frames: list[float] = []

    def play(self, *animations, **kwargs) -> None:  # type: ignore[override]
        self.frames.append(kwargs["run_time"] * self.fps)
        super().play(*animations, **kwargs)

    def wait(self, duration: float = 1.0, *args, **kwargs) -> None:  # type: ignore[override]
        self.frames.append(duration * self.fps)


# --- fixtures reach the registry walks ---------------------------------------


@pytest.mark.parametrize("params", EXAMPLES + STRESS,
                         ids=[f"ex{i}" for i in range(len(EXAMPLES))]
                         + [f"stress{i}" for i in range(len(STRESS))])
def test_every_fixture_carries_its_target_in(params):
    # test_layout builds each case with exactly these recipes; a case whose
    # target were missing would be a build_error there, not a layout result.
    (recipe,) = Callout.fixture_carry_in(params)
    assert recipe.name == params["target_id"]
    assert recipe.producer == "BulletReveal"
    assert continuity.build_artifact(recipe, DEFAULT).submobjects


def test_label_past_capacity_refuses_as_overflow():
    # Stress case 1 is far past what a callout band can hold legibly; it must
    # refuse rather than shrink the text below the floor.
    spec = _spec(STRESS[1], ["crowded"])
    report = validate_beat(spec, recipes=Callout.fixture_carry_in(STRESS[1]))
    assert report.kinds() == {"overflow"}, f"\n{report}"


# --- timing (D-002): whole frames, exactly the beat --------------------------


@pytest.mark.parametrize("factor", [0.5, 3.0], ids=["short", "long"])
@pytest.mark.parametrize("params", EXAMPLES, ids=[f"ex{i}" for i in range(len(EXAMPLES))])
def test_clocked_frames_equal_beat_frames(params, factor):
    budget = make_component(NAME, params).min_seconds() * factor
    with tempconfig({"frame_rate": DRAFT.frame_rate}):
        clock = _FrameClock(_component(params), duration=budget)
    clock.construct()
    assert all(f == pytest.approx(round(f), abs=1e-9) for f in clock.frames)
    assert sum(round(f) for f in clock.frames) == clock.beat_frames
    assert clock.beat_frames == math.ceil(budget * DRAFT.frame_rate)


def test_draft_render_is_exactly_the_beat(tmp_path):
    # 3.879 s of audio is ceil(58.185) = 59 frames at 15 fps.
    params = EXAMPLES[0]
    with tempconfig({**asdict(DRAFT), "media_dir": str(tmp_path),
                     "disable_caching": True, "progress_bar": "none",
                     "verbosity": "WARNING", "output_file": "callout"}):
        scene = ChalkdustScene(_component(params), duration=3.879)
        scene.render()
        movie = scene.renderer.file_writer.movie_file_path
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames", "-of", "json", str(movie)],
        capture_output=True, text=True, check=True).stdout
    assert int(json.loads(out)["streams"][0]["nb_read_frames"]) == \
        math.ceil(3.879 * DRAFT.frame_rate)


def test_semantic_hooks():
    comp = make_component(NAME, EXAMPLES[0])
    assert comp.min_seconds() == pytest.approx(1.5)  # make room, point, read
    assert comp.latex_strings() == []  # plain text: no maths to compile
    assert comp.carried_targets() == ["causes"]


# --- schema -----------------------------------------------------------------


@pytest.mark.parametrize("params", [
    {"target_id": "t", "text": ""},
    {"target_id": "t", "text": "   "},
    {"target_id": "", "text": "note"},
    {"target_id": "t"},
    {"target_id": "t", "text": "note", "side": "top"},
    {"target_id": "t", "text": "note", "part": -1},
    {"target_id": "t", "text": "note", "definitely_not_a_real_param": 1},
], ids=["empty-text", "blank-text", "empty-target", "no-text", "bad-side",
        "negative-part", "unknown-param"])
def test_schema_rejects(params):
    with pytest.raises(ValidationError):
        make_component(NAME, params)


# --- typed carry-in errors (spec bugs, not layout) --------------------------


def test_unregistered_target_fails_spec_validation():
    producer = BeatSpec(id="b01", narration="placeholder narration",
                        component="BulletReveal", params={"items": ["a"]},
                        registers="causes")
    with pytest.raises(ValidationError) as exc_info:
        VideoSpec(video_id="v", beats=(producer, _spec(
            {"target_id": "nope", "text": "note"}, ["nope"])))
    err = exc_info.value.errors()[0]["ctx"]["error"]
    assert isinstance(err, CarryInError)
    assert (err.beat_id, err.name) == ("b02", "nope")


def test_target_not_carried_in_is_typed_error_at_build():
    # carry_in names one artifact, target_id another.
    other = _recipes({"target_id": "t"})
    with pytest.raises(CarryInError, match="something_else"):
        _probe({"target_id": "something_else", "text": "note"}, carry=other)


def test_target_not_carried_in_is_refused_at_rung_2():
    # The same spec bug, caught before anything is built, by name.
    spec = _spec({"target_id": "something_else", "text": "note"}, ["t"])
    report = validate_semantic(spec, registered_artifacts={"t"}, duration=4.0)
    assert report.kinds() == {"carry_in"}
    assert "'something_else'" in str(report)


def test_part_out_of_range_is_typed_error():
    with pytest.raises(CarryInError, match="part 3"):
        _probe({"target_id": "t", "text": "note", "part": 3})


# --- through the pipeline ----------------------------------------------------

NARRATION = "Look again at the last of these three causes of collisions."


def _video(target_id: str = "causes", items: list[str] | None = None) -> dict:
    items = items or ["A weak hash function", "A load factor left too high",
                      "Adversarial keys chosen to collide"]
    return {"video_id": "callout_e2e", "beats": [
        {"id": "b01", "narration": NARRATION, "component": "BulletReveal",
         "params": {"items": items}, "registers": "causes"},
        {"id": "b02", "narration": NARRATION, "component": NAME,
         "params": {"target_id": target_id, "part": 2, "side": "right",
                    "text": "The one an attacker controls"},
         "carry_in": ["causes"]},
    ]}


@pytest.fixture
def fake_voice(monkeypatch) -> FakeTTS:
    from chalkdust.speech import tts

    fake = FakeTTS()
    monkeypatch.setitem(tts.BACKENDS, fake.name, fake)
    return fake


def _render(root: Path, spec: dict, out: str) -> pipeline.RunResult:
    path = write_spec(root / "spec.json", spec)
    return pipeline.render(path, Quality.DRAFT, root / out,
                           cache_dir=root / "cache", work_dir=root / "work")


def test_pipeline_renders_a_carried_target_and_keys_on_it(tmp_path, fake_voice):
    # Callout.build raises CarryInError unless b01's list is on screen, so a
    # rendered b02 proves the render rebuilt it from b01's params; editing
    # those params must re-render b02 although b02 itself did not change.
    first = _render(tmp_path, _video(), "first.mp4")
    assert first.rebuilt == ["b01", "b02"]
    assert first.output.exists()
    edited = _render(tmp_path, _video(items=["A weak hash function",
                                             "A load factor left too high",
                                             "Keys an attacker picked"]), "edited.mp4")
    assert edited.rebuilt == ["b01", "b02"]
    assert all(b.speech_cached for b in edited.beats)


def test_pipeline_refuses_unknown_target_id_before_speech(tmp_path, fake_voice):
    with pytest.raises(pipeline.SemanticRefused) as exc:
        _render(tmp_path, _video(target_id="no_such_list"), "never.mp4")
    assert [(r.beat_id, sorted(r.kinds())) for r in exc.value.reports] == \
        [("b02", ["carry_in"])]
    assert "'no_such_list'" in str(exc.value)
    assert fake_voice.calls == []


# --- layout behaviour -------------------------------------------------------


@pytest.mark.parametrize("side", ["left", "right", "above", "below"])
def test_label_sits_on_requested_side(side):
    probe = _probe({"target_id": "t", "text": "Every lookup pays", "side": side})
    target = bbox(carried(probe, "t"))
    text = bbox(_labelled(probe, "callout"))
    assert not target.intersects(text)
    assert {"left": text.right <= target.left, "right": text.left >= target.right,
            "above": text.bottom >= target.top, "below": text.top <= target.bottom}[side]


LABELS = {
    "short": "Constant time",
    "token": "x" * 60,
    "url": "https://example.com/" + "a" * 60,
    "long": STRESS[0]["text"],
}


@pytest.mark.parametrize("scale", [0.7, 1.0, 1.5], ids=["wrap0.7", "wrap1", "wrap1.5"])
@pytest.mark.parametrize("label_text", LABELS.values(), ids=LABELS.keys())
@pytest.mark.parametrize("side", ["left", "right", "above", "below"])
@pytest.mark.parametrize("part", [None, 0, 5], ids=["whole", "first", "last"])
def test_label_and_arrow_never_leave_the_stage(side, label_text, scale, part):
    # Whatever `side` says, whichever part is pointed at, at every wrap width
    # mechanical repair may rebuild with: the label and its arrow stay inside
    # STAGE (so inside the safe area), or the beat refuses cleanly.
    params = {"target_id": "crowded", "text": label_text, "side": side, "part": part}
    stage = region_rect(Region.STAGE)
    try:
        with wrap_scale(scale):
            probe = _probe(params)
    except LayoutError as exc:
        # overflow: too much label for the band; overlap: the arrow to an
        # inner part would cross its siblings from this side.
        assert exc.kind in {"overflow", "overlap"}, exc
        return
    for name in ("callout", "callout arrow"):
        box = bbox(_labelled(probe, name))
        assert stage.contains(box), f"{name} at {box} leaves STAGE"
        assert safe_area().contains(box)


def test_pointed_part_restored_rest_stays_dim():
    probe = _probe({"target_id": "t", "text": "This one", "part": 1, "side": "right"})
    rows = carried(probe, "t").submobjects
    full = [max(m.get_fill_opacity() for m in r.family_members_with_points()) for r in rows]
    assert full[1] == pytest.approx(1.0)
    assert full[0] == full[2] == pytest.approx(1 - continuity.DIM_DARKNESS)


def test_shrunk_target_keeps_honest_font_sizes():
    # Making room shrinks the carried list by animation; its tracked font
    # sizes must shrink with it, or the legibility check would judge the
    # text at its pre-move size.
    items = ["This bullet carries a long point that wraps onto a second line"] * 3
    recipes = [ArtifactRecipe(name="t", producer="BulletReveal", params={"items": items})]
    params = {"target_id": "t", "text": "A note on every one of these", "side": "right"}
    target = carried(_probe(params, carry=recipes), "t")
    text = next(m for m in target.get_family() if hasattr(m, "_chalk_font_size"))
    theme = resolve_fonts(DEFAULT, warn=False)  # the fonts the probe drew with
    rebuilt = continuity.build_artifact(recipes[0], theme)
    ref = next(m for m in rebuilt.get_family() if hasattr(m, "_chalk_font_size"))
    assert text._chalk_font_size < ref._chalk_font_size  # it was shrunk
    assert text.height / ref.height == pytest.approx(
        text._chalk_font_size / ref._chalk_font_size, rel=1e-3)


def test_clamped_label_still_gets_a_straight_arrow():
    # Stress 0: a nine-line label beside the LAST of six rows is clamped up to
    # stay in the stage, so it is not level with its part. The arrow must
    # still run straight across the gap, not slant off the label's middle.
    probe = _probe(STRESS[0])
    assert bbox(_labelled(probe, "callout arrow")).height < 0.3  # tip only


def test_arrow_through_sibling_parts_refuses():
    # Below a three-row list, an arrow up to the FIRST row crosses rows 1-2.
    with pytest.raises(LayoutError) as exc_info:
        _probe({"target_id": "t", "text": "First", "part": 0, "side": "below"})
    assert exc_info.value.kind == "overlap"
