"""Font resolution: once per process, not once per scene.

resolve_fonts runs on every scene construction -- the validation probe, the
repair probe and the render of every beat -- so its cost and its warnings
scale with the number of scenes unless they are held per process.
"""

from __future__ import annotations

import manimpango

from chalkdust.scenes import theme as theme_mod
from chalkdust.scenes.theme import get_theme, resolve_fonts


def test_substitution_warned_once_per_process(monkeypatch, capsys):
    monkeypatch.setattr(theme_mod, "_installed_fonts",
                        lambda: frozenset({"Arial", "Courier New"}))
    monkeypatch.setattr(theme_mod, "_WARNED", set())

    for _ in range(3):  # three scenes, e.g. probe, repair probe, render
        resolve_fonts(get_theme("default"))

    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 3, lines  # heading, body, mono -- once each
    assert all(line.startswith("[theme] ") for line in lines)


def test_fonts_enumerated_once_per_process(monkeypatch):
    calls = []

    def list_fonts():
        calls.append(1)
        return ["Arial", "Courier New"]

    monkeypatch.setattr(manimpango, "list_fonts", list_fonts)
    theme_mod._installed_fonts.cache_clear()
    try:
        for _ in range(4):
            resolve_fonts(get_theme("default"), warn=False)
    finally:
        theme_mod._installed_fonts.cache_clear()  # next caller sees real fonts

    assert len(calls) == 1
