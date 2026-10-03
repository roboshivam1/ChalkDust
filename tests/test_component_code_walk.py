"""CodeWalk-specific behaviour.

Layout of examples() and stress() is covered by test_layout.py walking the
registry; this file pins what that walk cannot see: the timing contract, the
schema's refusals, and where the highlight actually lands.
"""

from __future__ import annotations

import pytest
from manim import config, tempconfig
from pydantic import ValidationError

from chalkdust.scenes.base import ChalkdustScene
from chalkdust.scenes.components.code_walk import (
    DIM_OPACITY,
    MAX_HIGHLIGHTS,
    CodeWalk,
)
from chalkdust.scenes.regions import LayoutError
from chalkdust.validate.geometric import LayoutProbe

SOURCE = "def f(x):\n    y = x * 2\n    return y\n\nprint(f(3))"


def _elapsed(component: CodeWalk, budget: float, tmp_path) -> float:
    """Scene time after building at `budget`, without encoding frames.

    With skip_animations the renderer still advances its clock by every
    play() and wait() run time, so this is the beat's real elapsed time.
    """
    with tempconfig({"media_dir": str(tmp_path), "verbosity": "WARNING"}):
        scene = ChalkdustScene(component, duration=budget, skip_animations=True)
        scene.setup()
        scene.construct()
        return scene.renderer.time


def _probe(component: CodeWalk) -> LayoutProbe:
    probe = LayoutProbe(component, duration=10.0)
    probe.construct()
    return probe


class TestTiming:
    """Animation consumes exactly the audio budget (D-002), whether narration
    runs far shorter or far longer than the component would like."""

    @pytest.mark.parametrize("factor", [0.5, 3.0])
    @pytest.mark.parametrize("highlights", [[], [{"start": 2}, {"start": 3, "end": 5}]])
    def test_consumes_budget_exactly(self, tmp_path, factor, highlights):
        comp = CodeWalk({"language": "python", "source": SOURCE,
                         "highlights": highlights})
        budget = factor * comp.min_seconds()
        elapsed = _elapsed(comp, budget, tmp_path)
        assert abs(elapsed - budget) < 1 / config.frame_rate, (elapsed, budget)

    def test_min_seconds_grows_with_each_highlight(self):
        bare = CodeWalk({"language": "python", "source": SOURCE})
        walked = CodeWalk({"language": "python", "source": SOURCE,
                           "highlights": [{"start": 1}, {"start": 2}]})
        assert bare.min_seconds() > 0
        per_step = (walked.min_seconds() - bare.min_seconds()) / 2
        assert per_step > 0

    def test_takes_no_latex(self):
        assert CodeWalk({"language": "python", "source": SOURCE}).latex_strings() == []


class TestSchema:
    """Input the component cannot honour fails at validation, as a pydantic
    error -- never as a crash inside build()."""

    @pytest.mark.parametrize("params", [
        {"language": "python", "source": ""},
        {"language": "python", "source": "  \n\t\n   "},
        {"language": "klingon", "source": "x = 1"},
        {"language": "python", "source": "x = 1", "highlights": [{"start": 2}]},
        {"language": "python", "source": SOURCE, "highlights": [{"start": 0}]},
        {"language": "python", "source": SOURCE,
         "highlights": [{"start": 3, "end": 2}]},
        {"language": "python", "source": SOURCE,
         "highlights": [{"start": 1, "colour": "red"}]},
        {"language": "python", "source": SOURCE,
         "highlights": [{"start": 1}] * (MAX_HIGHLIGHTS + 1)},
    ], ids=["empty", "whitespace", "unknown-language", "past-last-line",
            "line-zero", "reversed-span", "unknown-span-field", "too-many-steps"])
    def test_rejects(self, params):
        with pytest.raises(ValidationError):
            CodeWalk(params)

    def test_blank_edges_dropped_so_numbers_match_screen(self):
        # Pygments strips leading newlines before lexing; if the spec kept
        # them, "line 3" in the spec would be line 1 on screen.
        comp = CodeWalk({"language": "python", "source": "\n\nx = 1\ny = 2  \n\n",
                         "highlights": [{"start": 2}]})
        assert comp.params.source == "x = 1\ny = 2"
        with pytest.raises(ValidationError):
            CodeWalk({"language": "python", "source": "\n\nx = 1\ny = 2\n\n",
                      "highlights": [{"start": 3}]})


class TestBuild:
    def test_too_wide_refuses_with_overflow(self):
        comp = CodeWalk({"language": "python",
                         "source": "# https://example.com/" + "segment-" * 20})
        with pytest.raises(LayoutError) as info:
            _probe(comp)
        assert info.value.kind == "overflow"
        assert "columns" in str(info.value)

    def test_minimal_listing_has_no_highlight_bar(self):
        probe = _probe(CodeWalk({"language": "c", "source": "x"}))
        assert [getattr(m, "_chalk_label", None) for m in probe.mobjects] == ["code"]

    def test_bar_covers_exactly_the_last_span_and_dims_the_rest(self):
        probe = _probe(CodeWalk({"language": "python", "source": SOURCE,
                                 "highlights": [{"start": 1}, {"start": 2, "end": 3}]}))
        code, bar = probe.mobjects
        assert getattr(bar, "_chalk_label") == "highlight"
        nums = code.line_numbers
        inside = {i for i, n in enumerate(nums)
                  if bar.get_bottom()[1] < n.get_y() < bar.get_top()[1]}
        assert inside == {1, 2}
        # Behind the text, or the wash tints the glyphs it is meant to frame.
        assert bar.z_index < code.code_lines.z_index
        lit = [nums[i].get_fill_opacity() for i in range(len(nums))]
        assert lit == pytest.approx([DIM_OPACITY, 1, 1, DIM_OPACITY, DIM_OPACITY])
