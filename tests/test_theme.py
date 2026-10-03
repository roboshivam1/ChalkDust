"""Theme constructors: the behaviours every component inherits from them.

The maths tests compile LaTeX, so `latex` and `dvisvgm` must be on PATH.
"""

from __future__ import annotations

import pytest

from chalkdust.scenes.regions import INVALID_LATEX, LayoutError
from chalkdust.scenes.theme import DEFAULT, math


@pytest.mark.parametrize("bad", [r"\notacommand{x} = 1", r"\quad"],
                         ids=["compile-error", "renders-nothing"])
def test_math_refuses_invalid_latex_as_one_kind(bad):
    # Every maths constructor refuses through refuse_invalid_latex, so the
    # repair loop sees one kind whichever component built the maths.
    with pytest.raises(LayoutError) as exc:
        math(bad, DEFAULT, what="label[7]")
    assert exc.value.kind == INVALID_LATEX
    assert "label[7]" in str(exc.value)

