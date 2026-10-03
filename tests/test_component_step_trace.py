"""StepTrace: behaviour the registry-wide layout tests do not pin.

test_layout.py already proves examples() validate clean and stress() fits or
refuses cleanly. This file pins timing against the audio budget, the schema's
refusals, the typed overflow path, and the muting rule that is the point of
the component.
"""

from __future__ import annotations

import pytest
from manim import tempconfig
from pydantic import ValidationError

from chalkdust.core.models import BeatSpec
from chalkdust.scenes.base import ChalkdustScene
from chalkdust.scenes.components.step_trace import (
    MAX_FRAMES,
    MAX_VARIABLES,
    StepTrace,
)
from chalkdust.scenes.regions import LayoutError
from chalkdust.validate.geometric import LayoutProbe, validate_beat

BINARY_SEARCH = StepTrace.examples()[0]
AT_CAPS = StepTrace.stress()[0]   # both caps, short values -- fits
OVERLOADED = StepTrace.stress()[1]  # both caps, long values -- must refuse
FPS = 15  # draft (D-006); the coarsest frame grid we render at


def _elapsed(params: dict, budget: float, tmp_path) -> float:
    """Scene clock after a build, without encoding a frame.

    With skip_animations Manim advances its clock by each play's run_time
    exactly -- after bumping any run_time shorter than one frame up to one
    frame, which is precisely the overrun this test must catch.
    """
    settings = {"media_dir": str(tmp_path), "frame_rate": FPS,
                "disable_caching": True, "verbosity": "WARNING"}
    with tempconfig(settings):
        scene = ChalkdustScene(StepTrace(params), duration=budget,
                               skip_animations=True)
        scene.setup()
        scene.construct()
        return scene.renderer.time


class TestTiming:
    """Animation consumes exactly the beat's audio budget (D-002)."""

    @pytest.mark.parametrize("params", [BINARY_SEARCH, AT_CAPS],
                             ids=["example", "at-caps"])
    @pytest.mark.parametrize("factor", [0.5, 3.0], ids=["short", "long"])
    def test_consumes_budget_exactly(self, params, factor, tmp_path):
        budget = factor * StepTrace(params).min_seconds()
        assert _elapsed(params, budget, tmp_path) == pytest.approx(
            budget, abs=1 / FPS)

    def test_min_seconds_grows_with_frames(self):
        one = StepTrace({"variables": ["x"], "frames": [[1]]}).min_seconds()
        two = StepTrace({"variables": ["x"], "frames": [[1], [2]]}).min_seconds()
        assert 0 < one < two

    def test_takes_no_latex(self):
        assert StepTrace(BINARY_SEARCH).latex_strings() == []


class TestSchema:
    """Inputs the table cannot honour fail at validation, not at render."""

    @pytest.mark.parametrize("params", [
        {"variables": [], "frames": [[1]]},
        {"variables": ["x"], "frames": []},
        {"variables": ["x"], "frames": [[]]},
        {"variables": ["x", "y"], "frames": [[1, 2], [3]]},
        {"variables": ["x", "x"], "frames": [[1, 2]]},
        {"variables": ["  "], "frames": [[1]]},
        {"variables": ["x"], "frames": [[""]]},
        {"variables": ["x"], "frames": [["two\nlines"]]},
        {"variables": [f"v{i}" for i in range(MAX_VARIABLES + 1)],
         "frames": [list(range(MAX_VARIABLES + 1))]},
        {"variables": ["x"], "frames": [[i] for i in range(MAX_FRAMES + 1)]},
    ], ids=["no-variables", "no-frames", "empty-frame", "ragged",
            "duplicate-name", "blank-name", "empty-cell", "multiline-cell",
            "too-many-variables", "too-many-frames"])
    def test_rejects(self, params):
        with pytest.raises(ValidationError):
            StepTrace(params)

    def test_minimal_trace_validates_clean(self):
        spec = BeatSpec(id="b01", narration="x starts undefined.",
                        component="StepTrace",
                        params={"variables": ["x"], "frames": [[None]]})
        assert validate_beat(spec).ok


class TestOverflow:
    def test_overloaded_trace_raises_layout_error(self, tmp_path):
        with tempconfig({"media_dir": str(tmp_path)}):
            probe = LayoutProbe(StepTrace(OVERLOADED), duration=8.0)
            with pytest.raises(LayoutError) as exc:
                probe.construct()
        assert exc.value.kind == "overflow"


class TestMuting:
    def test_only_changed_cells_keep_full_contrast(self, tmp_path):
        params = {"variables": ["a", "b", "tmp"],
                  "frames": [[3, 7, None], [3, 7, 3], [7, 7, 3]]}
        with tempconfig({"media_dir": str(tmp_path)}):
            probe = LayoutProbe(StepTrace(params), duration=8.0)
            probe.construct()
        pal = probe.theme.palette
        table = probe.mobjects[0]
        rows = [m for m in table.submobjects
                if getattr(m, "_chalk_label", "").startswith("frame[")]

        def colours(row):
            # Text keeps colour on its glyphs; the parent reports a default.
            return [cell[0].get_fill_color().to_hex().upper() for cell in row]

        fg, muted = pal.fg.upper(), pal.muted.upper()
        # Frame 0: defined values full, undefined tmp muted.
        assert colours(rows[0]) == [fg, fg, muted]
        # Frame 1: only tmp was assigned.
        assert colours(rows[1]) == [muted, muted, fg]
        # Frame 2: only a changed.
        assert colours(rows[2]) == [fg, muted, muted]
