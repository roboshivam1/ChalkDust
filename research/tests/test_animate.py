"""The harness's view of the animation components (research/animate.py)."""

import pytest

from research import animate
from research.generate import PROMPT_HASH, PROMPT_HASH_ANIM, _anim_context, _plan_schema


def test_every_registered_animation_has_a_card():
    # A new Animation without a card is invisible to the model: this is the
    # check that the harness picks up new components.
    assert sorted(animate.animation_names()) == sorted(animate.CARDS)


@pytest.mark.parametrize("name", sorted(animate.CARDS))
def test_a_card_describes_exactly_the_components_phases(name):
    assert list(animate.CARDS[name][3]) == animate.phases(name)


def test_an_animation_item_expands_to_one_beat_per_phase_in_order():
    plan = {"beats": [{"purpose": "hook", "idea": "why slopes", "component": "TitleCard"},
                      {"purpose": "intuition", "idea": "see it", "component": "SecantToTangent"},
                      {"purpose": "intuition", "idea": "again", "component": "SecantToTangent"},
                      {"purpose": "answer", "idea": "answer", "component": "AnswerBox"}]}
    out = animate.expand_plan(plan)["beats"]
    assert [b.get("anim_phase") for b in out] == [None, "pose", "secant", "shrink", "name", None]
    assert [b.get("anim_first") for b in out[1:5]] == [True, False, False, False]


def test_at_most_two_animations_are_kept():
    plan = {"beats": [{"purpose": "intuition", "idea": "x" * 10, "component": c}
                      for c in ("LimitAtHole", "RiemannSum", "SecantToTangent")]}
    out = animate.expand_plan(plan)
    assert out["anim_dropped"] == ["SecantToTangent"]
    assert {b["component"] for b in out["beats"]} == {"LimitAtHole", "RiemannSum"}


def test_later_phases_get_the_carried_setup_and_only_their_moments_as_cues():
    s = animate.beat_schema("SecantToTangent", "shrink", False)["properties"]["params"]
    assert set(s["properties"]) == {"cues"}
    assert set(s["properties"]["cues"]["properties"]) == {"slide", "tangent"}
    setup = {"function": "x^2/4 + 1", "x_range": [-1, 4], "a": 1, "h": 2}
    p = animate.merge_params("SecantToTangent", "shrink",
                             {"cues": {"slide": "slide it"}, "a": 99, "phase": "pose"}, setup)
    assert p == {**setup, "phase": "shrink", "cues": {"slide": "slide it"}}


def test_the_facts_are_the_numbers_the_screen_shows():
    fx, head = animate.facts("RiemannSum", {"function": "x^2/4 + 1", "x_range": [-1, 4], "a": 0,
                                            "b": 3, "rule": "left", "n0": 4})
    assert head == pytest.approx(5.25)
    assert "4.48, 4.85, 5.04, 5.15, 5.20" in fx["refine"][0]
    fx, head = animate.facts("SecantToTangent", {"function": "x^2/4 + 1", "x_range": [-1, 4],
                                                 "a": 1, "h": 2})
    assert head == pytest.approx(0.5) and "slope = 1.00" in fx["secant"][0]
    ctx, text = _anim_context({"component": "LimitAtHole", "anim_phase": "approach",
                               "anim_first": False, "anim_id": 1},
                              {1: {"function": "(x^2 - 1)/(x - 1)", "x_range": [-1, 3], "a": 1}})
    assert ctx["phase"] == "approach" and "end on 2.00" in text


@pytest.mark.parametrize("expr, want", [
    ("derivative(x^2/4 + 1, 1)", 0.5),
    ("limit((x^2 - 1)/(x - 1), 1)", 2.0),
    ("limit(sin(x)/x, 0)", 1.0),
    ("integral(x^2/4 + 1, 0, 3)", 5.25),
    ("15*cos(40*deg)", 11.4906666),
])
def test_the_calculator_computes_calculus_answers(expr, want):
    assert animate.calc_value(expr) == pytest.approx(want, rel=1e-6)


@pytest.mark.parametrize("expr", ["limit(abs(x)/x, 0)", "derivative(x, 1, 2)", "integral(y, 0, 1)",
                                  "__import__('os')", "derivative(x^2, x)", ""])
def test_the_calculator_refuses_what_is_not_one_number(expr):
    assert animate.calc_value(expr) is None


def test_the_decided_answer_is_written_into_the_answer_box():
    assert animate.answerbox_value("f'(1) = 0.6", 0.5) == "f'(1) = 0.5"
    assert animate.answerbox_value("10.4", 8 * 1.2992829841302609) == r"\approx 10.39"
    assert animate.says("so the slope is zero point five", 0.5)
    assert not animate.says("so the slope is zero point six", 0.5)


def test_the_animation_method_leaves_the_other_methods_prompts_alone():
    assert PROMPT_HASH == "e7709251718c" and PROMPT_HASH_ANIM != PROMPT_HASH
    enum = _plan_schema(True, False)["properties"]["beats"]["items"]["properties"]["component"]["enum"]
    assert not set(enum) & set(animate.CARDS)


def test_the_rules_give_the_model_no_example_number_it_could_copy():
    # R3-anim (2026-11-26): the rule's example "five point two five" (the
    # integral lesson's answer) turned up as the derivative lesson's slope.
    import re
    from research.generate import BEAT_ANIM_RULES, PLAN_ANIM_RULES
    from research.content import spoken_numbers
    quoted = re.findall(r'"([^"]*)"', BEAT_ANIM_RULES + PLAN_ANIM_RULES)
    assert all(spoken_numbers(q) == [] for q in quoted), quoted


def test_riemann_facts_give_the_red_area_the_screen_shows():
    # writer-riemann, round 2: under the mid rule the red slivers lie on both
    # sides of the curve, so the gap on screen (their area) is not |integral - sum|.
    fx, _ = animate.facts("RiemannSum", {"function": "sin(2*x) + 1.5", "x_range": [0, 5.5], "a": 0.3,
                                         "b": 5, "rule": "mid", "n0": 4})
    assert "total area 1.70" in fx["boxes"][0]
