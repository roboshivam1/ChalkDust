"""ZoomHighlight: the behaviours the registry-wide layout tests cannot pin.

ZoomHighlight acts on a carried artifact (SCENE_SPEC.md §6), and
tests/test_layout.py probes through validate_beat, which builds without
carry-ins -- so the class leaves examples()/stress() empty and this file runs
its carried_examples()/carried_stress() through beat_component instead, the
way the compiler will.

No shipped component registers an artifact builder yet, so BulletReveal is
lent one for each test (monkeypatched, never left in the registry) that
rebuilds its rows: one part per bullet. No LaTeX is involved.
"""

from __future__ import annotations

import pytest
from manim import DOWN, LEFT, RIGHT, Dot, VGroup, tempconfig
from pydantic import ValidationError

from chalkdust import continuity
from chalkdust.continuity import beat_component, resolve_carry_in
from chalkdust.core.models import BeatSpec, CarryInError, Region, VideoSpec
from chalkdust.scenes.base import ChalkdustScene
from chalkdust.scenes.components import get_component, make_component
from chalkdust.scenes.components.base import wrap
from chalkdust.scenes.components.zoom_highlight import MAX_ZOOM, ZoomHighlight
from chalkdust.scenes.regions import LayoutError, bbox, region_rect, smallest_font_size
from chalkdust.scenes.theme import body_text
from chalkdust.validate.geometric import LayoutProbe

NAME = "ZoomHighlight"
EXAMPLES = ZoomHighlight.carried_examples()
STRESS = ZoomHighlight.carried_stress()
CLEAN_REFUSALS = {"overflow", "illegible"}  # as in tests/test_layout.py
DRAFT_FPS = 15
ARTIFACT = "chain_causes"


def _rows_artifact(params, theme):
    rows = [VGroup(Dot(radius=0.07, color=theme.palette.accent),
                   body_text(wrap(item, 46), theme)).arrange(RIGHT, buff=0.28)
            for item in params.items]
    return VGroup(*rows).arrange(DOWN, buff=0.35, aligned_edge=LEFT)


@pytest.fixture(autouse=True)
def bullets_builder(monkeypatch):
    monkeypatch.setitem(continuity._BUILDERS, "BulletReveal", _rows_artifact)


def _video(producer: str, producer_params: dict, zoom_params: dict,
           carry_in: tuple[str, ...] = (ARTIFACT,)) -> VideoSpec:
    return VideoSpec(video_id="v", beats=[
        BeatSpec(id="b01", narration="placeholder narration", component=producer,
                 params=producer_params, registers=ARTIFACT),
        BeatSpec(id="b02", narration="placeholder narration", component=NAME,
                 params=zoom_params, carry_in=list(carry_in)),
    ])


def _component(case):
    video = _video(*case)
    return beat_component(video.beats[1], resolve_carry_in(video)["b02"])


def _probe(case, duration: float = 8.0) -> LayoutProbe:
    probe = LayoutProbe(_component(case), duration=duration, strict=False)
    probe.construct()
    return probe


def _by_label(scene) -> dict[str, object]:
    return {getattr(m, "_chalk_label", None): m for m in scene.mobjects}


def _ids(cases, prefix):
    return [f"{prefix}{i}" for i in range(len(cases))]


# --- layout -------------------------------------------------------------------


@pytest.mark.parametrize("case", EXAMPLES, ids=_ids(EXAMPLES, "ex"))
def test_carried_examples_validate_clean(case):
    assert _probe(case).layout_warnings == []


@pytest.mark.parametrize("case", STRESS, ids=_ids(STRESS, "stress"))
def test_carried_stress_fits_or_refuses_cleanly(case):
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


def test_lens_magnifies_focus_and_recedes_the_rest():
    # EXAMPLES[1]: two short rows of three, magnified on a lens.
    probe = _probe(EXAMPLES[1])
    mobs = _by_label(probe)
    target, lens, callout = mobs[f"carried[{ARTIFACT}]"], mobs["zoom lens"], mobs["callout"]

    def peak(m):
        return max(x.get_fill_opacity() for x in m.family_members_with_points())

    assert peak(target[1]) == pytest.approx(1.0)
    assert peak(target[2]) == pytest.approx(1.0)
    assert peak(target[0]) < 1 - continuity.DIM_DARKNESS
    # Magnified text is tracked at its magnified size, so legibility is
    # judged on what the viewer sees.
    assert smallest_font_size(lens) == pytest.approx(smallest_font_size(target) * MAX_ZOOM)
    assert region_rect(Region.STAGE).contains(bbox(lens))
    assert region_rect(Region.LOWER_THIRD).contains(bbox(callout))


def test_whole_target_is_restored_and_framed():
    # EXAMPLES[2]: no parts -- the target fills STAGE's width, so it is
    # framed rather than magnified, and comes back to full strength.
    mobs = _by_label(_probe(EXAMPLES[2]))
    target = mobs[f"carried[{ARTIFACT}]"]
    assert "zoom lens" not in mobs
    assert bbox(mobs["focus frame"]).contains(bbox(target))
    assert max(x.get_fill_opacity() for x in target.family_members_with_points()) \
        == pytest.approx(1.0)


# --- timing (D-002) -----------------------------------------------------------


def test_min_seconds_and_latex_strings():
    comp = make_component(NAME, EXAMPLES[0][2])
    assert comp.min_seconds() == pytest.approx(3.5)  # 7 weight units x 0.5 s
    assert comp.latex_strings() == []


@pytest.mark.parametrize("factor", [0.5, 3.0], ids=["short", "long"])
@pytest.mark.parametrize("case", EXAMPLES, ids=_ids(EXAMPLES, "ex"))
def test_real_render_consumes_budget_within_one_frame(tmp_path, case, factor):
    """Narration far shorter and far longer than the zoom wants: renders for
    real at 480p15 and reads the time Manim actually wrote. The carried
    target is added at t=0 without animation, so it must cost nothing."""
    budget = make_component(NAME, case[2]).min_seconds() * factor
    with tempconfig({"media_dir": str(tmp_path), "pixel_width": 854,
                     "pixel_height": 480, "frame_rate": DRAFT_FPS,
                     "disable_caching": True, "progress_bar": "none",
                     "verbosity": "WARNING", "output_file": f"zoom_{factor}"}):
        scene = ChalkdustScene(_component(case), duration=budget)
        scene.render()
    assert abs(scene.renderer.time - budget) <= 1 / DRAFT_FPS


# --- typed refusals -------------------------------------------------------------


def test_target_not_carried_is_typed_error():
    # The beat carries chain_causes but zooms into something else.
    producer, pp, zp = EXAMPLES[0]
    case = (producer, pp, {**zp, "target_id": "bucket_array"})
    with pytest.raises(CarryInError) as exc_info:
        _probe(case)
    assert exc_info.value.name == "bucket_array"


def test_unregistered_target_fails_spec_validation():
    producer, pp, zp = EXAMPLES[0]
    with pytest.raises(ValidationError) as exc_info:
        _video(producer, pp, {**zp, "target_id": "nope"}, carry_in=("nope",))
    err = exc_info.value.errors()[0]["ctx"]["error"]
    assert isinstance(err, CarryInError) and err.name == "nope"


def test_part_out_of_range_is_typed_error():
    # Three rows: indices 0..2. Only the rebuilt artifact knows that.
    producer, pp, zp = EXAMPLES[0]
    with pytest.raises(CarryInError, match=r"part\(s\) \[3\]") as exc_info:
        _probe((producer, pp, {**zp, "parts": [1, 3]}))
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
