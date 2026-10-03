"""CodeWalk: a source listing walked through by highlighting lines.

Built on Manim's `Code` mobject as reworked in the 0.19 line (code_string=,
formatter_style=, paragraph_config=). Examples written against the pre-0.19
API (`file_name=`, `style=`, `insert_line_no=`) do not run on 0.21.

Code is not prose, so it is never wrapped. Breaking a line mid-token makes the
lexer colour the continuation wrongly (a wrapped comment loses its `#` and is
highlighted as code), and a wrapped Python line reads as a different
indentation. A listing that is too long or too wide scales down until it hits
the legibility floor and then refuses with a LayoutError -- the fix is a
shorter excerpt, which is a decision for the script, not the renderer
(SCENE_SPEC.md §4, §11 rule 1).
"""

from __future__ import annotations

from functools import lru_cache

from manim import (
    Code,
    FadeIn,
    ManimColor,
    Rectangle,
    Transform,
    VGroup,
    VMobject,
    interpolate_color,
)
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pygments.lexers import get_lexer_by_name
from pygments.style import Style
from pygments.token import Comment, Error, Keyword, Number, String, Token
from pygments.util import ClassNotFound

from chalkdust.continuity import artifact_builder
from chalkdust.core.models import Region
from chalkdust.scenes.base import ChalkdustScene
from chalkdust.scenes.components.base import (
    Component,
    ComponentParams,
    label,
    register,
)
from chalkdust.scenes.regions import LayoutError, fit_to_region, tag_font_size
from chalkdust.scenes.theme import Palette, Theme

# Schema-level density limit, like BulletReveal's six items: past eight stops
# the beat is a lecture, and the error points at the real fix (split the beat).
MAX_HIGHLIGHTS = 8

# Relative timing weights; scene.budget() turns them into seconds (D-002).
# The final hold is long because a listing with no highlights is mostly hold,
# and a reveal stretched over the whole beat reads as a stalled render.
REVEAL = 1
MOVE = 1    # highlight bar travels to the next span
DWELL = 3   # viewer reads the highlighted lines while narration explains them
HOLD = 3

# Below this many seconds per weight unit a step stops being readable: the
# 0.4 s bar move is a jump cut, the 1.2 s dwell is one line glanced at.
# min_seconds() is this times the total weight, i.e. the sum of every step's
# own minimum (SCENE_SPEC.md §8 rung 2).
MIN_SECONDS_PER_WEIGHT = 0.4

DIM_OPACITY = 0.35        # lines outside the highlighted span
HIGHLIGHT_OPACITY = 0.18  # accent wash behind the span; text stays on top
PANEL_LIFT = 0.06         # panel fill, this far from bg toward fg

# Line pitch over digit height for Code's default line_spacing. Only used for
# a one-line listing, where there is no second line number to measure from.
SINGLE_LINE_PITCH = 1.75


class LineSpan(BaseModel):
    """1-based, inclusive line range, numbered as the listing displays it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    start: int = Field(ge=1)
    end: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _ordered(self) -> LineSpan:
        if self.end is not None and self.end < self.start:
            raise ValueError(f"end ({self.end}) is before start ({self.start})")
        return self

    @property
    def last(self) -> int:
        return self.end if self.end is not None else self.start


class CodeWalkParams(ComponentParams):
    # A Pygments lexer alias ("python", "javascript", "c", "sql"). Required:
    # Pygments' guesser is unreliable on short excerpts.
    language: str
    source: str
    highlights: list[LineSpan] = Field(default_factory=list, max_length=MAX_HIGHLIGHTS)

    @field_validator("language")
    @classmethod
    def _known_language(cls, v: str) -> str:
        try:
            get_lexer_by_name(v)
        except ClassNotFound:
            raise ValueError(
                f"unknown language {v!r}; use a Pygments lexer alias such as "
                "'python', 'javascript', 'c' or 'sql'"
            ) from None
        return v

    @field_validator("source")
    @classmethod
    def _normalise_source(cls, v: str) -> str:
        """Drop trailing whitespace and blank lines at either end.

        Pygments strips leading and trailing newlines before lexing (its
        `stripnl` default), so a leading blank line would silently shift every
        displayed line number against the highlights. Normalising here means
        the numbers the spec validates against are the numbers on screen.
        """
        lines = [line.rstrip() for line in v.splitlines()]
        while lines and not lines[0]:
            lines.pop(0)
        while lines and not lines[-1]:
            lines.pop()
        if not lines:
            raise ValueError("source cannot be empty")
        return "\n".join(lines)

    @model_validator(mode="after")
    def _highlights_in_range(self) -> CodeWalkParams:
        n = self.source.count("\n") + 1
        for i, span in enumerate(self.highlights):
            if span.last > n:
                raise ValueError(
                    f"highlights[{i}] reaches line {span.last}; source has {n} lines"
                )
        return self


@register
class CodeWalk(Component):
    name = "CodeWalk"
    Params = CodeWalkParams

    def regions(self) -> set[Region]:
        return {Region.STAGE}

    def _weights(self) -> list[float]:
        return [REVEAL] + [MOVE, DWELL] * len(self.params.highlights) + [HOLD]

    def min_seconds(self) -> float:
        return MIN_SECONDS_PER_WEIGHT * sum(self._weights())

    def latex_strings(self) -> list[str]:
        return []

    def build(self, scene: ChalkdustScene) -> None:
        p: CodeWalkParams = self.params
        theme = scene.theme

        code = _listing(p, theme)
        try:
            fit_to_region(code, Region.STAGE)
        except LayoutError as exc:
            lines = p.source.split("\n")
            widest = max(len(line.expandtabs(4)) for line in lines)
            raise LayoutError(
                f"CodeWalk listing of {len(lines)} lines, widest {widest} "
                f"columns: {exc}",
                kind=exc.kind,
            ) from exc

        # What each source line's glyphs and number currently show at.
        opacity = [1.0] * len(code.line_numbers)

        times = scene.budget(*self._weights())
        scene.play(FadeIn(code), run_time=times[0])
        scene.settle("code revealed")

        bar = None
        for k, span in enumerate(p.highlights):
            move, dwell = times[1 + 2 * k], times[2 + 2 * k]
            # One bar travels span to span (Transform keeps the first mobject),
            # so its label carries no index; settle() names the step instead.
            target = label(_bar(code, span, theme.palette.accent), "highlight")
            if bar is None:
                bar = target
                anims = [FadeIn(bar)]
            else:
                anims = [Transform(bar, target)]

            # Animate the listing's own parts, never a wrapper group: Manim
            # adds any animated mobject that is not already on screen to the
            # scene, so a fresh VGroup per row would become a stray top-level
            # mobject on top of the listing.
            for i, want in enumerate(_row_opacities(code, span)):
                if want != opacity[i]:
                    anims += [m.animate.set_opacity(want) for m in _row(code, i)]
                    opacity[i] = want

            scene.play(*anims, run_time=move)
            scene.settle(f"highlight {k}")
            scene.wait(dwell)

        scene.wait(times[-1])

    @classmethod
    def examples(cls):
        return [
            {"language": "python",
             "source": "def bucket_index(key, capacity):\n"
                       "    h = hash(key)\n"
                       "    return h % capacity\n"
                       "\n"
                       "index = bucket_index(\"cat\", 8)",
             "highlights": [{"start": 2}, {"start": 3}, {"start": 5}]},
            {"language": "javascript",
             "source": "function binarySearch(arr, target) {\n"
                       "  let lo = 0, hi = arr.length - 1;\n"
                       "  while (lo <= hi) {\n"
                       "    const mid = (lo + hi) >> 1;\n"
                       "    if (arr[mid] === target) return mid;\n"
                       "    if (arr[mid] < target) lo = mid + 1;\n"
                       "    else hi = mid - 1;\n"
                       "  }\n"
                       "  return -1;\n"
                       "}",
             "highlights": [{"start": 2}, {"start": 3, "end": 4},
                            {"start": 5, "end": 7}, {"start": 9}]},
            {"language": "sql",
             "source": "SELECT name, COUNT(*) AS orders\n"
                       "FROM customers JOIN orders USING (customer_id)\n"
                       "GROUP BY name\n"
                       "HAVING COUNT(*) > 5;"},
        ]

    @classmethod
    def stress(cls):
        long_listing = "\n".join(
            f"    total_{i} = compute_partial_sum(values, start={i}, step=3)"
            for i in range(30)
        )
        return [
            # (a) 3x a realistic excerpt, every highlight slot used.
            {"language": "python",
             "source": "def accumulate(values):\n" + long_listing,
             "highlights": [{"start": 1}] + [
                 {"start": 4 * i + 2, "end": 4 * i + 4} for i in range(7)]},
            # (b) a 60-character identifier: no break point, has to scale.
            {"language": "python",
             "source": "rehashed_bucket_index_for_keys_that_collide_in_the_open_tabl"
                       " = 0\nprint(table)",
             "highlights": [{"start": 1}]},
            # (b) a URL in a comment, far wider than the stage at legible size.
            {"language": "python",
             "source": "# https://example.com/" + "a-path-segment-without-any-spaces/" * 3
                       + "index.html\nfetch()",
             "highlights": [{"start": 1}, {"start": 2}]},
            # (c) minimal: one character, no highlights.
            {"language": "c", "source": "x"},
            # (c) a highlight that lands on a blank line mid-listing.
            {"language": "python", "source": "a = 1\n\nb = 2",
             "highlights": [{"start": 2}, {"start": 1, "end": 3}]},
        ]


# --- private helpers --------------------------------------------------------


@lru_cache(maxsize=8)
def _code_style(palette: Palette) -> type[Style]:
    """A Pygments style built from the theme palette.

    Code takes its colours from a Pygments style, so this is how the theme
    reaches the glyphs -- no named Pygments style, no colour from the spec.
    Could be shared via theme.py if another component ever lexes code.
    """
    panel = interpolate_color(
        ManimColor(palette.bg), ManimColor(palette.fg), PANEL_LIFT
    ).to_hex()
    return type(
        "ChalkdustCodeStyle",
        (Style,),
        {
            "background_color": panel,
            "line_number_color": palette.muted,
            "styles": {
                Token: palette.fg,
                Comment: palette.muted,
                Keyword: palette.accent,
                String: palette.success,
                Number: palette.accent_alt,
                Error: palette.danger,
            },
        },
    )


def _listing(p: CodeWalkParams, theme: Theme) -> Code:
    """The listing at its natural size, unplaced: shared by build() and the
    carry-in artifact so the two cannot drift apart."""
    code = label(
        Code(
            code_string=p.source,
            language=p.language,
            formatter_style=_code_style(theme.palette),
            add_line_numbers=True,
            background="rectangle",
            background_config={"stroke_color": theme.palette.muted},
            paragraph_config={
                "font": theme.type.mono_font,
                "font_size": theme.type.mono,
            },
        ),
        "code",
    )
    # Code builds its Paragraphs itself, bypassing theme.mono_text, so tag
    # them here or the legibility floor never sees this text.
    tag_font_size(code.code_lines, theme.type.mono)
    tag_font_size(code.line_numbers, theme.type.mono)
    # Text above the highlight bar; the bar is added later and would
    # otherwise draw over the glyphs it is meant to sit behind.
    code.code_lines.set_z_index(1)
    code.line_numbers.set_z_index(1)
    return code


def _row(code: Code, i: int) -> list[VMobject]:
    """Source line i's glyphs and its line number. An empty source line is an
    empty group with no points; it has nothing to dim, so it is left out."""
    return [m for m in (code.code_lines[i], code.line_numbers[i])
            if m.has_points() or m.family_members_with_points()]


def _row_opacities(code: Code, span: LineSpan) -> list[float]:
    """Per source line: full strength inside `span`, dimmed outside."""
    return [1.0 if span.start <= i + 1 <= span.last else DIM_OPACITY
            for i in range(len(code.line_numbers))]


@artifact_builder("CodeWalk")
def _artifact(p: CodeWalkParams, theme: Theme) -> Code:
    """The settled last frame, for a later beat to carry in (SCENE_SPEC.md §6):
    the listing, and when the walk had highlights, its last span lit and the
    other lines dimmed -- the picture the viewer was left with.

    Returned as the Code mobject itself (unplaced), so a consumer acting on a
    carried listing (ZoomHighlight, Callout) can reach `code_lines[i]` and
    `line_numbers[i]` for line i + 1. The bar, when present, is the last
    submobject, labelled "highlight".
    """
    code = _listing(p, theme)
    if p.highlights:
        span = p.highlights[-1]
        code.add(label(_bar(code, span, theme.palette.accent), "highlight"))
        for i, want in enumerate(_row_opacities(code, span)):
            for m in _row(code, i):
                m.set_opacity(want)
    return code


def _bar(code: Code, span: LineSpan, color: str) -> Rectangle:
    """Highlight bar behind `span`, measured from the line numbers.

    Line numbers are digits only -- no descenders, one per line even when the
    code line is empty -- so their centres are a stable row grid, unlike the
    code glyphs whose bounding boxes change with content.
    """
    nums = code.line_numbers
    if len(nums) > 1:
        pitch = (nums[0].get_y() - nums[-1].get_y()) / (len(nums) - 1)
    else:
        pitch = nums[0].height * SINGLE_LINE_PITCH

    top = nums[span.start - 1].get_y()
    bottom = nums[span.last - 1].get_y()
    content = VGroup(code.line_numbers, code.code_lines)
    pad = (pitch - nums[0].height) / 2

    bar = Rectangle(
        width=content.width + 2 * pad,
        height=top - bottom + pitch,
        fill_color=color,
        fill_opacity=HIGHLIGHT_OPACITY,
        stroke_width=0,
    )
    bar.move_to(content).set_y((top + bottom) / 2)
    return bar
