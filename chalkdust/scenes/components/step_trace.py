"""StepTrace: a variable-state table that fills in one frame at a time.

Columns are variables, rows are moments in the program's execution. Each
frame's row is revealed in turn under a highlight that marks "now", and a
value that did not change since the previous frame is muted so the eye goes
straight to what the step actually did.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from manim import Line, Mobject, Rectangle, Text, VGroup
from pydantic import Field, field_validator, model_validator

from chalkdust.continuity import artifact_builder
from chalkdust.core.models import Region
from chalkdust.scenes.base import ChalkdustScene
from chalkdust.scenes.components.base import (
    Component,
    ComponentParams,
    label,
    register,
)
from chalkdust.scenes.regions import fit_to_region
from chalkdust.scenes.theme import Theme, mono_text

# Density limits that fire at schema validation, like BulletReveal's item cap.
# Past six columns the values have no room to be read; past eight frames the
# trace no longer fits a 8-25 second beat at a legible pace (SCENE_SPEC.md §2).
# The fix in both cases is splitting the beat.
MAX_VARIABLES = 6
MAX_FRAMES = 8

# Vertical rhythm in cap heights of the mono font, so it scales with the theme.
# Rows are placed by BASELINE, not bounding box: "y" and "120" have different
# boxes, and box-aligned cells drift up and down across a row.
ROW_PITCH = 2.3    # data row to data row, baseline to baseline
HEADER_GAP = 2.9   # header baseline to first data baseline
RULE_DROP = 1.0    # header baseline to the rule beneath it
COL_GAP = 4.0      # space between adjacent columns
SLOT_PAD = 1.0     # how far the highlight overhangs the outer columns

HIGHLIGHT_OPACITY = 0.14
RULE_WIDTH = 2.0

# Shown for a variable that has no value yet at this frame (a `null` cell).
UNDEFINED = "—"

# Minimum legible duration of each timed segment, in seconds (SCENE_SPEC.md §8
# rung 2). These double as the budget weights: build() scales all of them by
# the same factor, so every segment gets at least its minimum exactly when the
# narration is at least min_seconds() long. A frame needs long enough for the
# viewer to find the changed cells.
HEADER_MIN_SECONDS = 0.5
FRAME_MIN_SECONDS = 0.8
HOLD_MIN_SECONDS = 1.0

# A cell is whatever scalar the trace shows. Values are displayed verbatim via
# str(); when exact spelling matters (JavaScript's `true`, a quoted string),
# pass a string.
Cell = str | int | float | bool | None


class StepTraceParams(ComponentParams):
    # Column headers, in order -- typically identifiers ("lo", "a[mid]").
    variables: list[str] = Field(min_length=1, max_length=MAX_VARIABLES)
    # One row per moment in time; each holds one value per variable, in the
    # same order as `variables`. null means "not yet defined".
    frames: list[list[Cell]] = Field(min_length=1, max_length=MAX_FRAMES)

    @field_validator("variables")
    @classmethod
    def _names_sane(cls, v: list[str]) -> list[str]:
        names = [_single_line(n, "variable name") for n in v]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f"duplicate variable names: {dupes}")
        return names

    @field_validator("frames")
    @classmethod
    def _cells_sane(cls, v: list[list[Cell]]) -> list[list[Cell]]:
        return [
            [_single_line(c, "cell value") if isinstance(c, str) else c
             for c in frame]
            for frame in v
        ]

    @model_validator(mode="after")
    def _frames_match_variables(self) -> StepTraceParams:
        n = len(self.variables)
        for i, frame in enumerate(self.frames):
            if len(frame) != n:
                raise ValueError(
                    f"frames[{i}] has {len(frame)} values but there are {n} "
                    f"variables; use null for a variable with no value yet"
                )
        return self


def _single_line(s: str, what: str) -> str:
    """A table cell is one line by construction -- a newline would silently
    break the constant row pitch -- and an empty one is unreadable."""
    s = s.strip()
    if not s:
        raise ValueError(
            f"{what} cannot be empty; use null for an undefined value, or "
            f"'\"\"' to show an empty string"
        )
    if "\n" in s:
        raise ValueError(f"{what} {s!r} must be a single line")
    return s


def _display(v: Cell) -> str:
    return UNDEFINED if v is None else str(v)


@lru_cache(maxsize=16)
def _mono_cap_height(font: str, size: float) -> float:
    """Cap height of the mono font: the layout unit for this table.

    theme.py only exposes the body font's cap height; this is its mono twin.
    """
    return float(Text("H", font=font, font_size=size).height)


def _cell(s: str, theme: Theme, color: str, x: float, baseline: float) -> Text:
    """A mono text centred on x and sitting on `baseline`.

    Manim keeps no baseline for a Text, so measure one: a throwaway copy with
    a leading "H" has a glyph whose bottom IS the baseline, and the glyphs
    after it are laid out exactly as the real text's (verified on Courier New
    and the theme's mono fonts -- no kerning across the prefix).
    """
    probe = mono_text("H" + s, theme)
    rest = VGroup(*probe.submobjects[1:])
    rise = rest.get_center()[1] - probe.submobjects[0].get_bottom()[1]

    text = mono_text(s, theme, color)
    text.set_x(x)
    text.set_y(baseline + rise)
    return text


@dataclass
class _Trace:
    """The table's parts, laid out in the table's own frame, all visible."""

    table: VGroup           # everything below; slots first, so they draw behind
    header: VGroup
    rule: Line
    rows: list[VGroup]      # one per frame
    slots: list[Rectangle]  # one highlight per frame, transparent


def _build_trace(p: StepTraceParams, theme: Theme) -> _Trace:
    """Lay the whole table out, unplaced.

    A pure function of params and theme: build() animates what it returns,
    and the continuity builder below rebuilds the same table for a later
    beat's carry_in (SCENE_SPEC.md §6).
    """
    pal = theme.palette
    cap = _mono_cap_height(theme.type.mono_font, theme.type.mono)

    shown = [[_display(v) for v in frame] for frame in p.frames]

    # Everything below is laid out in the table's own frame, origin at the
    # header baseline; the caller places the table as a whole with
    # fit_to_region, so no coordinate here survives into the frame
    # (SCENE_SPEC.md §4).
    #
    # Column widths from the widest thing in each column, measured on
    # throwaway texts before anything is positioned.
    widths = []
    for j, name in enumerate(p.variables):
        cells = [name] + [row[j] for row in shown]
        widths.append(max(mono_text(c, theme).width for c in cells))
    centres, cursor = [], 0.0
    for w in widths:
        centres.append(cursor + w / 2)
        cursor += w + COL_GAP * cap
    left, right = 0.0, cursor - COL_GAP * cap

    header = label(VGroup(*[
        _cell(name, theme, pal.accent, x, 0.0)
        for name, x in zip(p.variables, centres)
    ]), "header")

    rule_y = -RULE_DROP * cap
    rule = label(Line(
        [left - SLOT_PAD * cap, rule_y, 0.0],
        [right + SLOT_PAD * cap, rule_y, 0.0],
        color=pal.muted, stroke_width=RULE_WIDTH,
    ), "rule")

    rows, slots = [], []
    for i, values in enumerate(shown):
        baseline = -(HEADER_GAP + i * ROW_PITCH) * cap
        cells = []
        for j, (s, x) in enumerate(zip(values, centres)):
            # Muted: undefined, or unchanged since the previous frame.
            # Only the cells this step touched keep full contrast.
            quiet = p.frames[i][j] is None or (
                i > 0 and s == shown[i - 1][j])
            cells.append(_cell(s, theme, pal.muted if quiet else pal.fg,
                               x, baseline))
        rows.append(label(VGroup(*cells), f"frame[{i}]"))

        # One highlight per row, crossfaded rather than one bar moved:
        # every position is then fixed at build time and checked by the
        # probe, which applies a move in one step (validate/geometric.py).
        slot = Rectangle(
            width=right - left + 2 * SLOT_PAD * cap,
            height=ROW_PITCH * cap,
            stroke_width=0, fill_color=pal.accent, fill_opacity=0,
        )
        slot.move_to([(left + right) / 2, baseline + cap / 2, 0.0])
        slots.append(label(slot, f"highlight[{i}]"))

    table = label(VGroup(*slots, header, rule, *rows), "StepTrace")
    return _Trace(table, header, rule, rows, slots)


@artifact_builder("StepTrace")
def _artifact(params: StepTraceParams, theme: Theme) -> Mobject:
    """The settled trace, for a later beat that carries it in: every frame
    shown and the highlight on the last one -- the picture this beat ends on.
    Unplaced; continuity.CarryIn fits it to STAGE and dims it."""
    trace = _build_trace(params, theme)
    trace.slots[-1].set_fill(opacity=HIGHLIGHT_OPACITY)
    return trace.table


@register
class StepTrace(Component):
    name = "StepTrace"
    Params = StepTraceParams

    def regions(self) -> set[Region]:
        # A table this wide wants the full stage; there is no heading, the
        # narration names what is being traced.
        return {Region.STAGE}

    def build(self, scene: ChalkdustScene) -> None:
        trace = _build_trace(self.params, scene.theme)
        # One fit for the whole table. Raises LayoutError (overflow) when it
        # cannot stay legible: the fix is splitting the beat.
        fit_to_region(trace.table, Region.STAGE)

        # Add everything up front, hidden, so the scene holds one top-level
        # mobject; reveal by animating opacity (the BulletReveal pattern).
        for m in (trace.header, trace.rule, *trace.rows):
            m.set_opacity(0)
        scene.add(trace.table)

        # Whole-frame run times that sum to exactly the beat (D-002); the
        # base renders each as exactly that many frames.
        times = iter(scene.budget(*self._segments()))
        scene.play(trace.header.animate.set_opacity(1),
                   trace.rule.animate.set_opacity(1), run_time=next(times))
        for i, row in enumerate(trace.rows):
            anims = [row.animate.set_opacity(1),
                     trace.slots[i].animate.set_fill(opacity=HIGHLIGHT_OPACITY)]
            if i > 0:
                anims.append(trace.slots[i - 1].animate.set_fill(opacity=0))
            scene.play(*anims, run_time=next(times))

        scene.settle("trace complete")
        scene.wait(next(times))

    # --- semantic-rung hooks (SCENE_SPEC.md §8 rung 2) ------------------------

    def min_seconds(self) -> float:
        """Shortest narration at which every segment still gets its minimum."""
        return sum(self._segments())

    def _segments(self) -> list[float]:
        """Minimum duration of every timed segment, in the order build() plays
        them: header, one per frame, final hold. Shared by build() (as budget
        weights) and min_seconds() so the two cannot drift apart."""
        return [HEADER_MIN_SECONDS,
                *[FRAME_MIN_SECONDS] * len(self.params.frames),
                HOLD_MIN_SECONDS]

    def latex_strings(self) -> list[str]:
        # All text is mono Text drawn verbatim: a value that looks like LaTeX
        # or markup is shown exactly as typed, never compiled, so there is
        # nothing here that could refuse as invalid_latex.
        return []

    # --- fixtures -------------------------------------------------------------

    @classmethod
    def examples(cls):
        return [
            # Binary search for 23 in [2, 5, 8, 12, 16, 23, 38, 56, 72, 91].
            {"variables": ["lo", "hi", "mid", "a[mid]"],
             "frames": [[0, 9, 4, 16], [5, 9, 7, 56], [5, 6, 5, 23]]},
            # Accumulator loop.
            {"variables": ["i", "acc"],
             "frames": [[1, 1], [2, 2], [3, 6], [4, 24], [5, 120]]},
            # Swap through a temporary that starts undefined.
            {"variables": ["a", "b", "tmp"],
             "frames": [[3, 7, None], [3, 7, 3], [7, 7, 3], [7, 3, 3]]},
        ]

    @classmethod
    def stress(cls):
        long_id = "number_of_requests_still_waiting_in_the_retry_queue_total"
        url = "https://example.com/api/v2/resources/items?page=1&limit=100"
        return [
            # At both caps with short values: the densest realistic trace.
            {"variables": ["i", "j", "lo", "hi", "mid", "found"],
             "frames": [[i, i + 1, 0, 9 - i, 4, False]
                        for i in range(MAX_FRAMES)]},
            # At both caps with long values: ~3x realistic volume.
            {"variables": ["queue", "visited", "frontier", "parent",
                           "distance", "current_node"],
             "frames": [[f"[{i}, {i + 1}, {i + 2}, {i + 3}, {i + 4}]"] * 6
                        for i in range(MAX_FRAMES)]},
            # Unwrappable tokens: a 60-character identifier and a URL value.
            {"variables": [long_id], "frames": [[1], [2]]},
            {"variables": ["url"], "frames": [[url]]},
            # Minimal: one variable, one frame, and that frame undefined.
            {"variables": ["x"], "frames": [[None]]},
            # Values that look like LaTeX or Pango markup -- including LaTeX
            # that would not compile -- are text, drawn verbatim. StepTrace
            # takes no maths, so this must fit, not refuse as invalid_latex.
            {"variables": ["expr", "tag"],
             "frames": [[r"\frac{1}{", "<b>&amp;"],
                        [r"$x^2$", "</span>"]]},
        ]
