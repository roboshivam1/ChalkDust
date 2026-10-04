"""Where a carried artifact goes when the beat's component does not act on it
(register D-G4c-1; SCENE_SPEC.md §4, §6, §11 rule 1).

Only a consumer (Callout, ZoomHighlight: a non-empty carried_targets()) lays
its frame out around what it carries in. Any other component does not know
the artifact is there, so CarryIn draws it dimmed in the STAGE region the
component leaves free -- STAGE_RIGHT beside a STAGE_LEFT component. A
component claiming STAGE, or both halves, leaves none, and the semantic rung
refuses the beat as a region_conflict, before any speech or render. Every
library component claims one of those today, so every such beat is refused;
conftest's _LeftNote, which claims STAGE_LEFT only, stands in for the free
case.
"""

from __future__ import annotations

import json

import pytest
from manim import tempconfig
from test_layout import CLEAN_REFUSALS

from chalkdust import cli, continuity, pipeline
from chalkdust.continuity import (
    ArtifactRecipe,
    CarryIn,
    beat_component,
    carried,
    carry_in_fingerprint,
    carry_region,
    fixture_beat,
    is_consumer,
    resolve_carry_in,
)
from chalkdust.core.models import BeatSpec, Region, VideoSpec
from chalkdust.scenes.components import get_component, make_component, registered_names
from chalkdust.scenes.regions import (
    DEFAULT_PADDING,
    MIN_FONT_SIZE,
    LayoutError,
    bbox,
    region_rect,
    smallest_font_size,
)
from chalkdust.validate.geometric import Finding, LayoutProbe, Report, validate_beat
from chalkdust.validate.semantic import check_carry_placement, validate_semantic

LIST = {"items": ["hash(key)", "mod 8", "bucket 4"]}
# A twelve-cell array: legible across STAGE, too wide for half of it.
WIDE_ARRAY = {"kind": "array", "initial": list(range(10, 22))}


@pytest.fixture(autouse=True)
def _manim_scratch_in_tmp(tmp_path):
    with tempconfig({"media_dir": str(tmp_path / "manim"), "verbosity": "WARNING"}):
        yield


def _video(component: str, params: dict, *producers: tuple[str, dict]) -> VideoSpec:
    """b01.. register t0, t1, ...; the last beat carries them all in."""
    beats = [BeatSpec(id=f"b0{i + 1}", narration="placeholder narration",
                      component=name, params=p, registers=f"t{i}")
             for i, (name, p) in enumerate(producers)]
    beats.append(BeatSpec(id="b09", narration="placeholder narration",
                          component=component, params=params,
                          carry_in=[f"t{i}" for i in range(len(producers))]))
    return VideoSpec(video_id="v", beats=tuple(beats))


def _last(video: VideoSpec) -> tuple[BeatSpec, tuple[ArtifactRecipe, ...]]:
    beat = video.beats[-1]
    return beat, resolve_carry_in(video)[beat.id]


def _probe(beat: BeatSpec, recipes, strict: bool = False) -> LayoutProbe:
    probe = LayoutProbe(beat_component(beat, recipes), duration=4.0, strict=strict)
    probe.construct()
    return probe


# --- the rule ---------------------------------------------------------------


@pytest.mark.parametrize("claimed,free", [
    ({Region.STAGE}, None),
    ({Region.STAGE, Region.TITLE_BAR}, None),
    ({Region.STAGE_LEFT, Region.STAGE_RIGHT}, None),
    ({Region.STAGE_LEFT, Region.STAGE_RIGHT, Region.LOWER_THIRD}, None),
    ({Region.STAGE_LEFT}, Region.STAGE_RIGHT),
    ({Region.STAGE_LEFT, Region.LOWER_THIRD}, Region.STAGE_RIGHT),
    ({Region.STAGE_RIGHT, Region.TITLE_BAR}, Region.STAGE_LEFT),
    ({Region.TITLE_BAR}, Region.STAGE),
    ({Region.LOWER_THIRD}, Region.STAGE),
])
def test_free_region_is_the_stage_the_component_does_not_claim(claimed, free):
    assert carry_region(claimed) is free


# --- rung 2: refused, typed, before anything is built -------------------------

# The gate-45c repros (D-G4c-1): a list or an array carried into each of
# these, none of which acts on it. All claim STAGE or both halves.
NON_CONSUMERS = {
    "SplitCompare": {"left": {"title": "Open", "body": "probe next slot"},
                     "right": {"title": "Chain", "body": "list per bucket"}},
    "TitleCard": {"title": "Hash tables"},
    "BulletReveal": {"items": ["one", "two"]},
    "EquationDerivation": get_component("EquationDerivation").examples()[0],
    "AnswerBox": {"value": "4"},
    "CodeWalk": {"language": "python", "source": "def f(x):\n    return x * 2",
                 "highlights": [{"start": 2}]},
}


@pytest.mark.parametrize("name", sorted(NON_CONSUMERS))
def test_non_consumer_claiming_the_stage_is_a_region_conflict(name):
    component = make_component(name, NON_CONSUMERS[name])
    [finding] = check_carry_placement(component, ["tgt"])
    assert finding.kind == "region_conflict"
    # Readable: names the component, the artifact and the regions.
    regions = sorted(r.value for r in component.regions())
    assert name in finding.message and "'tgt'" in finding.message
    assert ", ".join(regions) in finding.message
    assert "carry_in" in finding.message


@pytest.mark.parametrize("name", [n for n in registered_names()
                                  if get_component(n).examples()])
def test_every_library_component_is_placed_or_refused(name):
    # Registry walk: each component either consumes what it carries in, has
    # a free STAGE region for it, or is refused at rung 2 -- never neither.
    # (RawScene ships no examples; a RawScene beat with a carry_in degrades
    # before rung 2 sees it, pipeline.raw_scene_refusal.)
    cls = get_component(name)
    params = cls.examples()[0]
    component = make_component(name, params)
    findings = check_carry_placement(component, ["tgt"])
    if is_consumer(component):
        assert findings == []
    elif carry_region(component.regions()) is None:
        assert [f.kind for f in findings] == ["region_conflict"]
    else:
        assert findings == []


def test_semantic_rung_refuses_the_beat(tmp_path):
    beat, recipes = _last(_video("AnswerBox", {"value": "4"}, ("BulletReveal", LIST)))
    report = validate_semantic(beat, {"t0"}, duration=6.0, recipes=recipes,
                               media_dir=tmp_path)
    assert report.kinds() == {"region_conflict"}, report


def test_cli_refuses_it_as_spec_invalid_before_speech(tmp_path, monkeypatch, capsys):
    # g5-nc-bul-answer: exit 3, the finding typed, and no beat synthesised.
    spec = {"video_id": "nc", "theme": "default", "beats": [
        {"id": "b01", "narration": "we now walk through this idea slowly",
         "component": "BulletReveal", "params": LIST, "registers": "tgt"},
        {"id": "b02", "narration": "we now walk through this idea slowly",
         "component": "AnswerBox", "params": {"value": "4"}, "carry_in": ["tgt"]},
    ]}
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec), encoding="utf-8")

    def no_speech(*args, **kwargs):
        raise AssertionError("speech ran for a beat rung 2 refuses")

    monkeypatch.setattr(pipeline, "synthesize_beat", no_speech)
    for verb in (["validate"], ["render", "--cache-dir", str(tmp_path / "cache")]):
        rc = cli.main([*verb, str(path), "--work-dir", str(tmp_path / "work")])
        err = capsys.readouterr().err
        assert rc == cli.EXIT_SPEC_INVALID, err
        assert err.startswith("chalkdust: spec invalid: rung 2 (semantic: carry-in)")
        assert "[region_conflict] AnswerBox carries in 'tgt'" in err


def test_region_conflict_is_a_carry_in_refusal():
    # Fixed by the beat's carry_in, never by its content: spec invalid, as a
    # broken carry_in reference is (exit 3), not semantic refused (exit 8).
    report = Report("b02", [Finding("region_conflict", "x carries in 'tgt'")])
    with pytest.raises(pipeline.CarryInRefused):
        pipeline.refuse_unfit([report], pipeline.SemanticRefused)


# --- rung 3 backstop -----------------------------------------------------------


def test_probe_refuses_typed_when_rung_2_was_skipped():
    beat, recipes = _last(_video("TitleCard", {"title": "Next"}, ("BulletReveal", LIST)))
    report = validate_beat(beat, recipes=recipes)
    assert report.kinds() == {"region_conflict"}, report
    with pytest.raises(LayoutError) as info:
        _probe(beat, recipes, strict=True)
    assert info.value.kind == "region_conflict"


# --- placed beside a component that leaves room -------------------------------


def test_artifact_goes_dimmed_into_the_free_half(left_note, tmp_path):
    beat, recipes = _last(_video(left_note, {"text": "Chaining"}, ("BulletReveal", LIST)))
    assert check_carry_placement(make_component(left_note, beat.params), beat.carry_in) == []
    assert validate_semantic(beat, {"t0"}, recipes=recipes, media_dir=tmp_path).ok

    report = validate_beat(beat, recipes=recipes)
    assert report.ok, f"\n{report}"
    probe = _probe(beat, recipes, strict=True)
    art = carried(probe, "t0")
    assert region_rect(Region.STAGE_RIGHT).inset(DEFAULT_PADDING).contains(bbox(art))
    assert not bbox(art).intersects(bbox(probe.mobjects[-1]))  # the note
    opacities = [m.get_fill_opacity() for m in art.family_members_with_points()]
    assert max(opacities) == pytest.approx(1 - continuity.DIM_DARKNESS)
    assert smallest_font_size(art) >= MIN_FONT_SIZE
    assert beat_component(beat, recipes).regions() == {Region.STAGE_LEFT,
                                                       Region.STAGE_RIGHT}


def test_several_artifacts_share_the_free_half_in_carry_in_order(left_note):
    beat, recipes = _last(_video(left_note, {"text": "Chaining"},
                                 ("BulletReveal", LIST),
                                 ("BulletReveal", {"items": ["probe", "rehash"]})))
    report = validate_beat(beat, recipes=recipes)
    assert report.ok, f"\n{report}"
    probe = _probe(beat, recipes)
    first, second = bbox(carried(probe, "t0")), bbox(carried(probe, "t1"))
    half = region_rect(Region.STAGE_RIGHT)
    assert half.contains(first) and half.contains(second)
    assert not first.intersects(second)
    assert first.y > second.y  # carry_in order, top to bottom


def test_artifact_too_dense_for_the_free_half_refuses_cleanly(left_note):
    # Legible across STAGE, not in half of it: refused, never shrunk below
    # the floor, by a kind the repair loop treats as a clean refusal.
    beat, recipes = _last(_video(left_note, {"text": "Arrays"},
                                 ("DataStructureViz", WIDE_ARRAY)))
    report = validate_beat(beat, recipes=recipes)
    assert report.kinds() and report.kinds() <= CLEAN_REFUSALS, report
    message = report.findings[0].message
    assert "'t0'" in message and "stage_right" in message and "_LeftNote" in message


def test_probe_sees_the_component_drawing_over_it(left_note):
    # _LeftNote(spill=True) centres its text across STAGE, past its claim.
    # The component never marked the artifact exclusive -- it does not know
    # it is there -- so only the probe's own check can catch this.
    beat, recipes = _last(_video(left_note, {"text": "Chaining, probing and more",
                                             "spill": True},
                                 ("BulletReveal", LIST)))
    report = validate_beat(beat, recipes=recipes)
    assert report.kinds() == {"overlap"}, report
    assert "note is drawn over carried[t0]" in report.findings[0].message
    with pytest.raises(LayoutError) as info:
        _probe(beat, recipes, strict=True)
    assert info.value.kind == "overlap"


# --- consumers are unchanged -----------------------------------------------------


@pytest.mark.parametrize("name", ["Callout", "ZoomHighlight"])
def test_consumers_keep_their_artifacts_in_stage(name):
    spec, recipes = fixture_beat(name, get_component(name).examples()[0])
    wrapped = beat_component(spec, recipes)
    assert isinstance(wrapped, CarryIn) and is_consumer(wrapped.component)
    assert wrapped.placement() is Region.STAGE
    assert wrapped.regions() == wrapped.component.regions() | {Region.STAGE}
    assert check_carry_placement(wrapped.component, spec.carry_in) == []
    probe = _probe(spec, recipes)
    for art in probe._chalk_carried.values():
        assert not getattr(art, "_chalk_carry_clear", False)


# --- render key -----------------------------------------------------------------


def test_placement_rule_is_in_the_carried_fingerprint(monkeypatch):
    recipes = (ArtifactRecipe(name="t0", producer="BulletReveal", params=LIST),)
    before = carry_in_fingerprint(recipes)
    monkeypatch.setattr(continuity, "PLACEMENT_RULE", "something else")
    assert carry_in_fingerprint(recipes) != before


def test_placement_follows_the_beats_own_spec(left_note):
    # The rest of what decides placement -- the consuming component and its
    # params -- is the beat's own spec, which beat_render_key hashes.
    a = BeatSpec(id="b02", narration="n", component=left_note,
                 params={"text": "x"}, carry_in=["t0"])
    b = BeatSpec(id="b02", narration="n", component="TitleCard",
                 params={"title": "x"}, carry_in=["t0"])
    assert CarryIn(make_component(a.component, a.params), ()).placement() \
        is Region.STAGE_RIGHT
    assert CarryIn(make_component(b.component, b.params), ()).placement() is None
