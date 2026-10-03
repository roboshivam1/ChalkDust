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

import unicodedata
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
from chalkdust.scenes.regions import (
    DEFAULT_PADDING,
    MIN_FONT_SIZE,
    LayoutError,
    fit_to_region,
    region_rect,
    tag_font_size,
)
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

# Code checks that every non-space character became exactly one glyph and
# raises a bare ValueError when one did not. Two kinds of character never do:
# format characters (category Cf: zero-width space and joiner, bidi marks, soft
# hyphen) draw nothing, and emoji are drawn by Pango from a colour font that
# yields no outline. Both are refused at the schema, where the repair loop can
# read which character on which line to drop (SCENE_SPEC.md §8 rung 1).
# The BMP characters with Emoji_Presentation=Yes (Unicode emoji-data.txt);
# every other emoji lives in the pictograph blocks U+1F000-U+1FAFF.
_EMOJI_BMP = frozenset(
    [0x231A, 0x231B, *range(0x23E9, 0x23ED), 0x23F0, 0x23F3, 0x25FD, 0x25FE,
     0x2614, 0x2615, *range(0x2648, 0x2654), 0x267F, 0x2693, 0x26A1, 0x26AA,
     0x26AB, 0x26BD, 0x26BE, 0x26C4, 0x26C5, 0x26CE, 0x26D4, 0x26EA, 0x26F2,
     0x26F3, 0x26F5, 0x26FA, 0x26FD, 0x2705, 0x270A, 0x270B, 0x2728, 0x274C,
     0x274E, 0x2753, 0x2754, 0x2755, 0x2757, 0x2795, 0x2796, 0x2797, 0x27B0,
     0x27BF, 0x2B1B, 0x2B1C, 0x2B50, 0x2B55]
)
_PICTOGRAPHS = range(0x1F000, 0x1FB00)


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
        """Compose accents, refuse undrawable characters, and drop trailing
        whitespace and blank lines at either end.

        NFC first: a decomposed accent ("e" + U+0301) is two characters that
        Pango draws as one glyph, which Code refuses; the composed form looks
        the same and is one character, one glyph.

        Pygments strips leading and trailing newlines before lexing (its
        `stripnl` default), so a leading blank line would silently shift every
        displayed line number against the highlights. Normalising here means
        the numbers the spec validates against are the numbers on screen.
        """
        lines = [line.rstrip() for line in unicodedata.normalize("NFC", v).splitlines()]
        for n, line in enumerate(lines, start=1):
            for ch in line:
                if _undrawable(ch):
                    name = unicodedata.name(ch, "unnamed character")
                    raise ValueError(
                        f"source line {n} contains U+{ord(ch):04X} {name}, which "
                        "a code listing cannot draw (emoji and zero-width or "
                        "format characters); remove it or spell it in ASCII"
                    )
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
            # (a) a whole 200-line file pasted in: refused on size before
            # Code is built (Pango would drop glyphs past the frame).
            {"language": "python",
             "source": "\n".join(f"x{i} = {i}" for i in range(200)),
             "highlights": [{"start": 1}]},
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


def _undrawable(ch: str) -> bool:
    """A character Code cannot turn into exactly one glyph (see _EMOJI_BMP)."""
    cp = ord(ch)
    if unicodedata.category(ch) == "Cf":
        return True
    return cp in _EMOJI_BMP or (
        cp in _PICTOGRAPHS and unicodedata.category(ch) in ("So", "Sk")
    )


def _code(source: str, language: str, theme: Theme) -> Code:
    """Manim's Code in the theme's colours and mono type, at natural size."""
    try:
        return Code(
            code_string=source,
            language=language,
            formatter_style=_code_style(theme.palette),
            add_line_numbers=True,
            background="rectangle",
            background_config={"stroke_color": theme.palette.muted},
            paragraph_config={
                "font": theme.type.mono_font,
                "font_size": theme.type.mono,
            },
        )
    except ValueError as exc:
        # Backstop for a character the schema does not know the theme's mono
        # font cannot draw: Code checks one glyph per non-space character and
        # raises a bare ValueError. Only that check is translated; any other
        # ValueError is a bug and propagates as one.
        if "rendered fewer glyph" not in str(exc):
            raise
        raise LayoutError(
            f"CodeWalk listing has characters the {theme.type.mono_font} font "
            "cannot draw as one glyph each (an emoji, combining or invisible "
            "character); spell them in ASCII",
            kind="illegible",
        ) from exc


@lru_cache(maxsize=8)
def _cell(theme: Theme) -> tuple[float, float]:
    """(line pitch, column advance) of a listing at the theme's mono size,
    measured off a two-line ASCII probe rather than assumed from the font."""
    probe = _code("0000000000\n0000000000", "text", theme)
    nums, first = probe.line_numbers, probe.code_lines[0]
    pitch = nums[0].get_y() - nums[1].get_y()
    advance = (first[-1].get_x() - first[0].get_x()) / (len(first) - 1)
    return pitch, advance


def _cells(line: str) -> float:
    """A lower bound on how many mono cells `line` takes.

    Printable ASCII is one Courier cell by construction, and East Asian wide
    characters are square, so at least one. Anything else counts half a cell:
    the narrowest glyph measured across Latin, Greek, Cyrillic, Hebrew, Arabic,
    Indic, Thai, symbols and punctuation is 0.87 of a cell, and a script the
    mono font lacks falls back to wider glyphs. Combining marks and whitespace
    other than the space count nothing.
    """
    total = 0.0
    for c in line:
        if " " <= c <= "~" or unicodedata.east_asian_width(c) in ("W", "F"):
            total += 1
        elif not c.isspace() and unicodedata.category(c)[0] != "M":
            total += 0.5
    return total


def _refuse_oversized(p: CodeWalkParams, theme: Theme) -> None:
    """Refuse, before Code is built, a listing that cannot fit the STAGE even
    at the legibility floor.

    Code draws each Paragraph on a Pango surface the size of the output frame
    (config.pixel_width x pixel_height) and silently drops the glyphs that
    fall off it, which surfaces as Code's bare glyph-count ValueError -- for a
    pasted 200-line file, or one 3000-column line, before fit_to_region ever
    gets to say overflow. Both bounds here are lower bounds on the listing's
    size (line-number centres span (lines - 1) pitches; columns are counted
    by _cells), so this never refuses a listing fit_to_region would accept.
    """
    pitch, advance = _cell(theme)
    floor = MIN_FONT_SIZE / theme.type.mono  # the smallest scale fit allows
    room = region_rect(Region.STAGE).inset(DEFAULT_PADDING)
    lines = p.source.split("\n")
    columns = max(_cells(line.expandtabs(4)) for line in lines)
    tall = (len(lines) - 1) * pitch * floor > room.height
    wide = (columns - 1) * advance * floor > room.width
    if tall or wide:
        raise LayoutError(
            f"CodeWalk listing of {len(lines)} lines, at least {columns:.0f} columns, "
            f"cannot fit the stage at the legibility floor (font_size "
            f"{MIN_FONT_SIZE:.0f}). Show a shorter excerpt, or split the beat.",
            kind="overflow",
        )


def _listing(p: CodeWalkParams, theme: Theme) -> Code:
    """The listing at its natural size, unplaced: shared by build() and the
    carry-in artifact so the two cannot drift apart."""
    _refuse_oversized(p, theme)
    code = label(_code(p.source, p.language, theme), "code")
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
