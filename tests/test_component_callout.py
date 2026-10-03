"""Callout: annotate a carried artifact (SCENE_SPEC.md §5, §6).

tests/test_layout.py cannot cover Callout yet: it validates bare components,
and a Callout without its carried target is a spec bug by definition
(register N-4). So Callout keeps its fixtures in carried_examples() /
carried_stress(), and this file walks them through CarryIn with exactly the
assertions test_layout makes, plus what is specific to the component: timing,
schema refusals, the typed carry-in errors, and the arrow-crossing refusal.

No shipped component registers an artifact builder yet, so BulletReveal is
lent one here for every test (monkeypatched, never left in the registry).
"""

from __future__ import annotations

import pytest
from manim import DOWN, LEFT, Dot, VGroup
from manim.animation.animation import prepare_animation
from pydantic import ValidationError

from chalkdust import continuity
from chalkdust.continuity import ArtifactRecipe, beat_component, carried
from chalkdust.core.models import BeatSpec, CarryInError, VideoSpec
from chalkdust.scenes.components import make_component
from chalkdust.scenes.components.callout import Callout
from chalkdust.scenes.components.base import label, wrap
from chalkdust.scenes.components.bullet_reveal import WRAP_WIDTH
from chalkdust.scenes.regions import LayoutError, bbox
from chalkdust.scenes.theme import body_text
from chalkdust.validate.geometric import Finding, LayoutProbe, Report

NAME = "Callout"
EXAMPLES = Callout.carried_examples()
STRESS = Callout.carried_stress()
CLEAN_REFUSALS = {"overflow", "illegible"}  # as tests/test_layout.py
DRAFT_FPS = 15


def _list_artifact(params, theme):
    """A BulletReveal-shaped artifact: one part per item, dot + text rows,
    wrapped at BulletReveal's width."""
    rows = []
    for i, item in enumerate(params.items):
        text = body_text(wrap(item, WRAP_WIDTH), theme)
        dot = Dot(radius=0.07, color=theme.palette.accent).next_to(text, LEFT, buff=0.28)
        rows.append(label(VGroup(dot, text), f"item[{i}]"))
    return VGroup(*rows).arrange(DOWN, aligned_edge=LEFT, buff=0.35)


@pytest.fixture(autouse=True)
def lent_builder(monkeypatch):
    monkeypatch.setitem(continuity._BUILDERS, "BulletReveal", _list_artifact)


def _spec(params: dict, carry: list[str]) -> BeatSpec:
    return BeatSpec(id="b02", narration="placeholder narration",
                    component=NAME, params=params, carry_in=carry)


def _component(case: dict):
    recipes = [ArtifactRecipe(**r) for r in case["carry_in"]]
    spec = _spec(case["params"], [r.name for r in recipes])
    return beat_component(spec, recipes)


def _validate(case: dict, duration: float = 8.0) -> Report:
    """validate_beat() for a carried beat: same probe, same error mapping."""
    report = Report("b02")
    probe = LayoutProbe(_component(case), duration=duration, strict=False)
    try:
        probe.construct()
    except LayoutError as exc:
        report.findings.append(Finding(exc.kind, str(exc)))
        return report
    except Exception as exc:
        report.findings.append(Finding("build_error", f"{type(exc).__name__}: {exc}"))
        return report
    report.findings.extend(Finding(k, m) for k, m in probe.layout_warnings)
    return report


def _probe(case: dict, duration: float = 8.0) -> LayoutProbe:
    probe = LayoutProbe(_component(case), duration=duration)
    probe.construct()
    return probe


def _case(params: dict, items: list[str] | None = None) -> dict:
    items = items or ["Hash the key", "Find the bucket", "Walk the chain"]
    return {"carry_in": [{"name": "t", "producer": "BulletReveal",
                          "params": {"items": items}}],
            "params": {"target_id": "t", **params}}


class _Clock(LayoutProbe):
    """A probe that adds up the time every play() and wait() asks for -- the
    scene's elapsed time, without encoding a frame. A play() with no explicit
    run_time counts at its animations' own default, so a forgotten run_time
    shows up as a budget mismatch. (Pattern from EquationDerivation's tests.)"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.elapsed = 0.0

    def play(self, *animations, **kwargs) -> None:  # type: ignore[override]
        run_time = kwargs.get("run_time")
        if run_time is None:
            run_time = max(prepare_animation(a).run_time for a in animations)
        self.elapsed += run_time
        super().play(*animations, **kwargs)

    def wait(self, duration: float = 1.0, *args, **kwargs) -> None:  # type: ignore[override]
        self.elapsed += duration


# --- the registry-walk assertions, for carried fixtures ---------------------


@pytest.mark.parametrize("case", EXAMPLES, ids=[f"ex{i}" for i in range(len(EXAMPLES))])
def test_examples_validate_clean(case):
    report = _validate(case)
    assert report.ok, f"\n{report}"


@pytest.mark.parametrize("case", STRESS, ids=[f"stress{i}" for i in range(len(STRESS))])
def test_stress_fits_or_refuses_cleanly(case):
    report = _validate(case)
    assert report.kinds() <= CLEAN_REFUSALS, f"\n{report}"


def test_label_past_capacity_refuses_as_overflow():
    # Stress case 1 is far past what a callout band can hold legibly; it must
    # refuse rather than shrink the text below the floor.
    assert _validate(STRESS[1]).kinds() == {"overflow"}


def test_registry_walk_fixtures_are_empty():
    # Pins the deviation: test_layout's walk has no carry-in path (N-4), so a
    # non-empty examples()/stress() would fail there with CarryInError.
    assert Callout.examples() == [] and Callout.stress() == []


# --- timing (D-002) ---------------------------------------------------------


@pytest.mark.parametrize("factor", [0.5, 3.0], ids=["short", "long"])
@pytest.mark.parametrize("case", EXAMPLES, ids=[f"ex{i}" for i in range(len(EXAMPLES))])
def test_timing_consumes_budget_exactly(case, factor):
    budget = make_component(NAME, case["params"]).min_seconds() * factor
    clock = _Clock(_component(case), duration=budget)
    clock.construct()
    assert clock.elapsed == pytest.approx(budget, abs=1 / DRAFT_FPS)


def test_semantic_hooks():
    comp = make_component(NAME, EXAMPLES[0]["params"])
    assert comp.min_seconds() == pytest.approx(1.5)  # make room, point, read
    assert comp.latex_strings() == []


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


def test_target_not_carried_in_is_typed_error():
    # carry_in names one artifact, target_id another.
    case = _case({"text": "note"})
    case["params"]["target_id"] = "something_else"
    with pytest.raises(CarryInError, match="something_else"):
        _probe(case)


def test_part_out_of_range_is_typed_error():
    with pytest.raises(CarryInError, match="part 3"):
        _probe(_case({"text": "note", "part": 3}))


# --- layout behaviour -------------------------------------------------------


@pytest.mark.parametrize("side", ["left", "right", "above", "below"])
def test_label_sits_on_requested_side(side):
    probe = _probe(_case({"text": "Every lookup pays", "side": side}))
    target = bbox(carried(probe, "t"))
    text = bbox(next(m for m in probe.mobjects
                     if getattr(m, "_chalk_label", None) == "callout"))
    assert not target.intersects(text)
    assert {"left": text.right <= target.left, "right": text.left >= target.right,
            "above": text.bottom >= target.top, "below": text.top <= target.bottom}[side]


def test_pointed_part_restored_rest_stays_dim():
    probe = _probe(_case({"text": "This one", "part": 1, "side": "right"}))
    rows = carried(probe, "t").submobjects
    full = [max(m.get_fill_opacity() for m in r.family_members_with_points()) for r in rows]
    assert full[1] == pytest.approx(1.0)
    assert full[0] == full[2] == pytest.approx(1 - continuity.DIM_DARKNESS)


def test_arrow_through_sibling_parts_refuses():
    # Below a three-row list, an arrow up to the FIRST row crosses rows 1-2.
    with pytest.raises(LayoutError) as exc_info:
        _probe(_case({"text": "First", "part": 0, "side": "below"}))
    assert exc_info.value.kind == "overlap"
