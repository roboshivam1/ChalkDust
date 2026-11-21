"""Pinned behaviours of the generation and training plumbing (no model, no GPU)."""

import json

from research.build_sft import _is_plan, video_ok
from research.generate import _plan_schema
from research.reward_server import reward_of
from research.score import BeatScore
from research.train_topics import TRAIN, EXCLUDED_NEAR_TWINS
from research.topics import FAMILIES


def test_working_and_calc_come_before_final_answer():
    # grammar-constrained decoding writes properties in schema order
    keys = list(_plan_schema(calc=True, work=True)["properties"])
    assert keys.index("working") < keys.index("answer_calc") < keys.index("final_answer") < keys.index("beats")
    assert list(_plan_schema(False, False)["properties"]) == [
        "title", "objective", "key_insight", "misconception", "final_answer", "beats"]


def test_plan_schema_required_matches_properties():
    s = _plan_schema(calc=True, work=True)
    assert set(s["required"]) == set(s["properties"])


def _sc(level, flags=(), chars=200, component="TitleCard"):
    return BeatScore("b01", component, level, flags=list(flags), text_chars=chars)


def test_reward_levels_and_floor():
    long = {"narration": " ".join(["word"] * 40)}
    short = {"narration": "too short"}
    assert reward_of(_sc(3), long)[0] == 1.0
    assert reward_of(_sc(3, ["narration_latex"]), long)[0] == 0.5
    assert reward_of(_sc(2), long)[0] == 0.1
    assert reward_of(_sc(0), long)[0] == 0.0
    assert reward_of(_sc(3), short)[0] == 0.5            # content floor halves it
    assert reward_of(_sc(3, chars=5, component="AnswerBox"), long)[0] == 0.5   # params with almost no text
    # (training review, 2026-11-21) the text floor is selection_key's: 12 chars, TitleCard exempt
    assert reward_of(_sc(3, chars=15, component="AnswerBox"), long)[0] == 1.0
    assert reward_of(_sc(3, chars=5), long)[0] == 1.0     # a TitleCard is allowed little text


def test_is_plan_mirrors_make_plan_filter():
    ok = json.dumps({"beats": [{"purpose": "hook", "idea": "x", "component": "TitleCard"}]})
    assert _is_plan(ok)
    assert not _is_plan(json.dumps({"beats": [{"component": "RawScene"}]}))
    assert not _is_plan("not json")


def test_training_pool_is_disjoint_from_evaluation():
    fams = {t.family for t in TRAIN}
    assert not fams & FAMILIES
    assert not any(w in f for f in fams for w in EXCLUDED_NEAR_TWINS)
    assert len({t.id for t in TRAIN}) == len(TRAIN)


def test_video_ok_requires_correct_answer(tmp_path, monkeypatch):
    import research.analyze as an
    monkeypatch.setattr(an, "rescore", lambda rec: {})
    rec = {"answer": {"status": "correct"}, "answer_spoken": {"status": "not_found"}}
    assert not video_ok(rec, require_spoken=True)
    assert video_ok(rec, require_spoken=False)
    assert video_ok({"answer": {"status": "correct"}, "answer_spoken": {"status": "n/a"}})
    assert not video_ok({"answer": {"status": "wrong"}, "answer_spoken": {"status": "correct"}})
    assert video_ok({"answer": {"status": "n/a"}, "answer_spoken": {"status": "n/a"}})
