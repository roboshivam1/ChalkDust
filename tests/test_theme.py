"""Theme constructors: the behaviours every component inherits from them.

The maths tests compile LaTeX, so `latex` and `dvisvgm` must be on PATH.
"""

from __future__ import annotations

import dataclasses

import pytest
from manim import tempconfig

from chalkdust.scenes import theme as theme_mod
from chalkdust.scenes.regions import INVALID_LATEX, LayoutError
from chalkdust.scenes.theme import DEFAULT, math, resolve_fonts


@pytest.mark.parametrize("bad", [r"\notacommand{x} = 1", r"\quad"],
                         ids=["compile-error", "renders-nothing"])
def test_math_refuses_invalid_latex_as_one_kind(bad, tmp_path):
    # Every maths constructor refuses through refuse_invalid_latex, so the
    # repair loop sees one kind whichever component built the maths.
    with tempconfig({"media_dir": str(tmp_path)}), pytest.raises(LayoutError) as exc:
        math(bad, DEFAULT, what="label[7]")
    assert exc.value.kind == INVALID_LATEX
    assert "label[7]" in str(exc.value)


def test_font_substitution_is_announced_once_per_process(capsys):
    if not theme_mod._installed_fonts():
        pytest.skip("manimpango cannot list fonts here; nothing is substituted")
    # A font no machine has, so the fallback (and its warning) always fires,
    # and a name no other test uses, so this test sees the first warning.
    missing = dataclasses.replace(
        DEFAULT, type=dataclasses.replace(DEFAULT.type, heading_font="ChalkDust Absent Sans"))
    first = resolve_fonts(missing)
    second = resolve_fonts(missing)
    out = capsys.readouterr().out
    assert out.count("'ChalkDust Absent Sans' missing") == 1, out
    assert first == second
