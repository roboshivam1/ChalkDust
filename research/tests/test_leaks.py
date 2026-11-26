"""Leaked TitleCard fields (research/leaks.py): detection, and the SFT kicker trim
that changes nothing but the kicker. The strings are real E1c/E2/T1 outputs."""

import json

import pytest

from research.build_sft import trim_titlecard
from research.leaks import beat_leaks, field_leaks, kicker_leaks, trim_kicker

MEASURED = [
    ('JEE Main - Calculus", ', "JEE Main - Calculus"),                       # Gemma, straight quote
    ("First-year CS”, ", "First-year CS"),                                # Qwen, curly quote
    ("Class 10 CBSE Mathematics, ", "Class 10 CBSE Mathematics"),             # bare trailing comma
    ("First-year CS - Algorithms\n\nSubtitle: Learn how to search", "First-year CS - Algorithms"),
    ("Class 10 CBSE - Physics\n  Subtitle: Calculating current", "Class 10 CBSE - Physics"),
    ("JEE Main - Mechanics\nCalculating motion with friction.", "JEE Main - Mechanics"),   # unlabelled subtitle
]


@pytest.mark.parametrize("kicker,trimmed", MEASURED)
def test_measured_kickers_leak_and_trim_to_first_line(kicker, trimmed):
    assert kicker_leaks(kicker)
    assert trim_kicker(kicker) == trimmed
    assert not kicker_leaks(trimmed)


@pytest.mark.parametrize("kicker", ["JEE Main - Mechanics", "Class 11 CBSE - Probability", "Part 2", None])
def test_clean_kickers_do_not_leak(kicker):
    assert not kicker_leaks(kicker)


def test_trim_changes_only_the_kicker_and_keeps_teacher_formatting():
    raw = ('{\n  "narration": "Why do two resistors in series add? Think of one road.",\n'
           '  "params": {"title": "Series Resistors", "kicker": "Class 10 CBSE - Physics\\n\\nSubtitle: x"}\n}')
    new, why = trim_titlecard(raw)
    assert why == "trimmed"
    assert new == raw.replace("Class 10 CBSE - Physics\\n\\nSubtitle: x", "Class 10 CBSE - Physics")
    v = json.loads(new)
    assert v["params"] == {"title": "Series Resistors", "kicker": "Class 10 CBSE - Physics"}


def test_clean_titlecard_is_returned_unchanged():
    raw = '{"narration": "n", "params": {"title": "T", "kicker": "JEE Main - Mechanics"}}'
    assert trim_titlecard(raw) == (raw, "")


def test_untrimmable_kicker_is_dropped():
    raw = json.dumps({"narration": "n", "params": {"title": "T", "kicker": '", Subtitle: x'}})
    new, why = trim_titlecard(raw)
    assert new is None and why == "kicker_untrimmable"


def test_code_params_are_exempt_and_other_text_is_checked():
    assert field_leaks({"language": "python", "source": 'def f(title: str):\n    g("a", "b")'}) == []
    assert field_leaks({"items": ["one", 'two", ']}) == ["items[1]"]
    assert beat_leaks({"component": "TitleCard", "params": {"title": "T", "kicker": 'X", '}})
    assert not beat_leaks({"component": "TitleCard", "params": {"title": "T", "kicker": "X"}})
