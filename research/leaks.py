"""Leaked fields: one param's text carrying another field, a field label or JSON
punctuation. The validator passes these, and they are drawn on screen.

Found 2026-11-21 in TitleCard. `beat_schema` lists its params in class order
(title, subtitle, kicker; both optional) while the prompt (`catalog.py`) describes
title, kicker, subtitle. Unconstrained, models write that order (6/6 E1 "free"
samples). Under the grammar, `subtitle` must come before `kicker`. Models skip it,
write the kicker, and then put the subtitle inside the kicker or end it with JSON
punctuation ('JEE Main - Calculus", '). Some loop inside the kicker until max_tokens.

Leaky clean TitleCard samples: Qwen3-4B (E1c) 40/40, Gemma 4 12B (E2) 24/25,
gpt-oss-20b (E2) 5/12. No other component leaked in about 800 samples.
Generation is frozen until the study ends. Analysis reports the rate as `leak`
(`analyze.rescore`), and `build_sft` trims the kicker in SFT targets."""

from __future__ import annotations

import re

LABEL = re.compile(r"(?im)(?:^|\n)\s*(?:sub)?title\s*:|\b(?:kicker|narration|params)\s*:")
PUNCT = re.compile("[\"“”]\\s*,\\s*$|,\\s*$|\"\\s*,\\s*\"|^\\s*[\"”],")
TAIL = re.compile("[\\s\"“”',_]+$")
CODE_KEYS = {"source", "code"}   # CodeWalk / RawScene code: `title: str` and f("a", "b") are legitimate


def kicker_leaks(kicker) -> bool:
    """A TitleCard kicker that is more than one small line."""
    if not isinstance(kicker, str):
        return False
    return bool(LABEL.search(kicker) or PUNCT.search(kicker) or "\n" in kicker.strip())


def trim_kicker(kicker: str) -> str | None:
    """The kicker's first line without the stray tail; None if nothing clean is left."""
    first = next((ln for ln in kicker.split("\n") if ln.strip()), "")
    t = TAIL.sub("", first).strip()
    return t if t and not kicker_leaks(t) else None


def field_leaks(params) -> list[str]:
    """Paths of string params (code excepted) with a field label or stray JSON punctuation."""
    out = []

    def walk(o, path):
        if path.rsplit(".", 1)[-1] in CODE_KEYS:
            return
        if isinstance(o, str):
            if LABEL.search(o) or PUNCT.search(o):
                out.append(path)
        elif isinstance(o, dict):
            for k, v in o.items():
                walk(v, f"{path}.{k}" if path else k)
        elif isinstance(o, list):
            for i, v in enumerate(o):
                walk(v, f"{path}[{i}]")

    walk(params, "")
    return out


def beat_leaks(beat: dict) -> bool:
    params = beat.get("params") if isinstance(beat.get("params"), dict) else {}
    if beat.get("component") == "TitleCard" and kicker_leaks(params.get("kicker")):
        return True
    return bool(field_leaks(params))
