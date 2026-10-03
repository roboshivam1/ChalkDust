"""ProblemStatement: the question, formatted (JEE branch, SCENE_SPEC.md §5).

    A block of mass m is released from rest on a rough plane      <- text
    inclined at theta to the horizontal ...
    GIVEN                          FIND
    m = 2 kg                       the acceleration a of the block
    theta = 30 deg

Every field is prose with inline maths written $...$. Prose is theme body
text and maths is theme LaTeX, set on one shared baseline and flowed into
lines here -- Manim's Text neither wraps nor knows where its baseline is, so
both are measured rather than assumed.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated, NamedTuple

from manim import LEFT, RIGHT, UP, Mobject, VGroup
from pydantic import AfterValidator, Field

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
    LayoutError,
    fit_to_region,
    region_rect,
)
from chalkdust.scenes.theme import (
    Theme,
    body_cap_height,
    body_text,
    caption_text,
    emphasize,
    math,
)

# The paragraph measure in average characters: a comfortable reading line,
# capped by the stage width for themes with wide type.
MEASURE_CHARS = 64

# Spacing in body cap heights, so it scales with the theme.
LINE_PITCH = 1.7     # baseline to baseline for an ordinary line of prose
LINE_CLEARANCE = 0.25  # minimum ink gap when tall inline maths needs more room
WORD_SPACE = 0.4     # between prose and an adjacent maths chunk
ITEM_GAP = 0.5       # extra space between two given items
HEADER_GAP = 0.5     # between a GIVEN/FIND header and its content
SECTION_GAP = 1.4    # between the statement and the given/find row
COLUMN_GAP = 1.6     # between the given column and the find column

# Prefix compiled ahead of every inline maths chunk. The H is a strut: its
# bottom is the TeX baseline, the one thing a MathTex bounding box cannot tell
# us. It is measured, then removed. \textstyle keeps fractions and limits at
# inline size -- MathTex sets display maths otherwise.
STRUT = r"\mathrm{H}\textstyle "

# Relative weights of each step's share of the beat (D-002). The statement's
# weight is split evenly over its lines, so the total -- and min_seconds() --
# is known from the params alone, before any text is measured.
SECONDS_PER_WEIGHT = 0.4
STATEMENT_WEIGHT = 3
GIVEN_WEIGHT = 1
FIND_WEIGHT = 2
HOLD_WEIGHT = 3


class LatexError(LayoutError):
    """Inline maths that does not compile, or compiles to nothing visible.

    A LayoutError subclass because that is the only exception the geometric
    probe preserves the kind of (validate/geometric.py) -- anything else is
    reported as a build_error crash, which this is not: it is bad content
    from the spec, and the repair loop should regenerate it (SCENE_SPEC.md §9).
    """

    def __init__(self, message: str) -> None:
        super().__init__(message, kind="latex")


def _inline(v: str) -> str:
    """Prose with $...$ maths. Rejected here rather than at build time,
    because an unbalanced $ would silently swap which parts are maths."""
    v = v.strip()
    if not v:
        raise ValueError("cannot be blank")
    if v.count("$") % 2:
        raise ValueError(
            "unbalanced $: inline maths is written $...$ and a literal "
            "dollar sign is not supported"
        )
    if any(not chunk.strip() for chunk in v.split("$")[1::2]):
        raise ValueError("empty inline maths ($$)")
    return v


InlineText = Annotated[str, AfterValidator(_inline)]


class ProblemStatementParams(ComponentParams):
    text: InlineText
    # Capped at 6, like BulletReveal: a problem with more givens than that is
    # a data table, and failing at the schema points at the real fix.
    given: list[InlineText] = Field(default_factory=list, max_length=6)
    find: InlineText


# --- inline flow ------------------------------------------------------------


class _Token(NamedTuple):
    text: str
    is_math: bool
    space_before: bool  # whether the source had whitespace before it


def _tokens(s: str) -> list[_Token]:
    """Split into prose words and maths chunks, remembering where the source
    had spaces: in "the $x$-axis" nothing separates the maths from "-axis"."""
    out: list[_Token] = []
    space = False
    for i, part in enumerate(s.split("$")):
        if i % 2:
            out.append(_Token(part.strip(), True, space))
            space = False
            continue
        words = part.split()
        if not words:
            space = space or bool(part)
            continue
        lead = space or part[0].isspace()
        out += [_Token(w, False, lead if j == 0 else True) for j, w in enumerate(words)]
        space = part[-1].isspace()
    return out


def _maths(chunk: str) -> str:
    """The exact LaTeX string compiled for one inline maths chunk."""
    return STRUT + chunk


@lru_cache(maxsize=16)
def _char_width(theme: Theme) -> float:
    """Average prose character width, for estimating where lines break. The
    estimate is then checked against real widths, so it only has to be close."""
    sample = "The quick brown fox jumps over the lazy dog"
    return float(body_text(sample, theme).width) / len(sample)


@lru_cache(maxsize=16)
def _math_size(theme: Theme) -> float:
    """The maths size whose capitals match the prose's. Computer Modern and
    the theme's sans differ in proportion, so at equal nominal sizes inline
    maths reads a size smaller than the words around it."""
    tex_cap = math(r"\mathrm{H}", theme, size=theme.type.body).height
    return theme.type.body * body_cap_height(theme) / tex_cap


def _baselined(mob: Mobject, strut: Mobject) -> Mobject:
    """Remove the strut glyph and put the baseline it marked at y=0."""
    baseline = strut.get_bottom()[1]
    for m in mob.get_family():
        if strut in m.submobjects:
            m.remove(strut)
            break
    return mob.shift(UP * -baseline)


class _Flow:
    """Lays one InlineText out into lines no wider than `width`."""

    def __init__(self, s: str, theme: Theme, width: float) -> None:
        self.theme, self.width = theme, width
        self.cap = body_cap_height(theme)
        self.tokens = _tokens(s)
        self.maths = {i: self._compile(t.text)
                      for i, t in enumerate(self.tokens) if t.is_math}

    def _compile(self, chunk: str) -> Mobject:
        source = _maths(chunk)
        try:
            m = math(source, self.theme, size=_math_size(self.theme))
        except ValueError as exc:
            # Manim raises ValueError when LaTeX rejects the source. Its
            # RuntimeError (no log file at all) is a broken installation, not
            # bad content, so it is deliberately left to propagate.
            raise LatexError(
                f"ProblemStatement maths failed to compile: ${chunk}$ ({exc})"
            ) from exc
        glyphs = m.family_members_with_points()
        if len(glyphs) < 2:
            raise LatexError(f"ProblemStatement maths ${chunk}$ draws nothing")
        return _baselined(m, glyphs[0])

    def _estimate(self, i: int) -> float:
        t = self.tokens[i]
        if t.is_math:
            return self.maths[i].width
        return len(t.text) * _char_width(self.theme)

    def _split_long_words(self) -> None:
        """Hard-break prose words wider than a line, as wrap() does for the
        other components. Maths is never broken; fit_to_region decides."""
        per_line = max(1, int(self.width / _char_width(self.theme)))
        out, maths = [], {}
        for i, t in enumerate(self.tokens):
            if t.is_math:
                maths[len(out)] = self.maths[i]
                out.append(t)
            elif len(t.text) > per_line:
                pieces = [t.text[k:k + per_line] for k in range(0, len(t.text), per_line)]
                out += [_Token(p, False, t.space_before if k == 0 else False)
                        for k, p in enumerate(pieces)]
            else:
                out.append(t)
        self.tokens, self.maths = out, maths

    def _line(self, idx: list[int]) -> VGroup:
        """Build one line, baseline at y=0, left edge at x=0."""
        pieces: list[tuple[Mobject, bool]] = []
        run: list[str] = []
        run_space = False

        def flush() -> None:
            if run:
                t = body_text("H" + "".join(run), self.theme)
                pieces.append((_baselined(t, t[0]), run_space))
                run.clear()

        for n, i in enumerate(idx):
            t = self.tokens[i]
            if t.is_math:
                flush()
                pieces.append((self.maths[i], t.space_before and n > 0))
            else:
                if not run:
                    run_space = t.space_before and n > 0
                    run.append(t.text)
                else:
                    run.append((" " if t.space_before else "") + t.text)
        flush()

        x = 0.0
        for mob, spaced in pieces:
            if spaced:
                x += WORD_SPACE * self.cap
            mob.shift(RIGHT * (x - mob.get_left()[0]))
            x = mob.get_right()[0]
        return VGroup(*(m for m, _ in pieces))

    def lines(self) -> list[VGroup]:
        """The lines, each with its baseline at y=0, for _stack()."""
        self._split_long_words()
        space = WORD_SPACE * self.cap
        # Greedy fill on estimated widths...
        plan: list[list[int]] = [[]]
        x = 0.0
        for i in range(len(self.tokens)):
            w = self._estimate(i)
            gap = space if plan[-1] and self.tokens[i].space_before else 0.0
            if plan[-1] and x + gap + w > self.width:
                plan.append([])
                x, gap = 0.0, 0.0
            plan[-1].append(i)
            x += gap + w
        # ...then corrected against real ones: an overlong line hands its last
        # token down until it fits. A lone token wider than the measure stays;
        # fit_to_region scales it or refuses.
        built: list[VGroup] = []
        k = 0
        while k < len(plan):
            line = self._line(plan[k])
            while line.width > self.width and len(plan[k]) > 1:
                if k + 1 == len(plan):
                    plan.append([])
                plan[k + 1].insert(0, plan[k].pop())
                line = self._line(plan[k])
            built.append(line)
            k += 1
        return built


def _stack(rows: list[list[VGroup]], cap: float) -> list[VGroup]:
    """Stack rows of lines, each built with its baseline at y=0, top to bottom.

    Baseline to baseline at a constant pitch, so a superscript or a
    descender never changes the rhythm -- more only where tall maths would
    otherwise touch the line above. Rows (given items) get ITEM_GAP extra.
    The first baseline stays at y=0; returns one group per row.
    """
    baseline, prev_descent = 0.0, None
    for lines in rows:
        for j, line in enumerate(lines):
            ascent, descent = line.get_top()[1], -line.get_bottom()[1]
            if prev_descent is not None:
                baseline -= (ITEM_GAP * cap if j == 0 else 0.0) + max(
                    LINE_PITCH * cap, prev_descent + LINE_CLEARANCE * cap + ascent)
            line.shift(UP * baseline)
            prev_descent = descent
    return [VGroup(*lines) for lines in rows]


# --- the component ----------------------------------------------------------


@register
class ProblemStatement(Component):
    name = "ProblemStatement"
    Params = ProblemStatementParams

    def regions(self) -> set[Region]:
        return {Region.STAGE}

    def _weights(self, statement_lines: int = 1) -> list[float]:
        n_given = len(self.params.given)
        return ([STATEMENT_WEIGHT / statement_lines] * statement_lines
                + [GIVEN_WEIGHT] * n_given + [FIND_WEIGHT, HOLD_WEIGHT])

    def min_seconds(self) -> float:
        return SECONDS_PER_WEIGHT * sum(self._weights())

    def latex_strings(self) -> list[str]:
        p: ProblemStatementParams = self.params
        return [_maths(t.text)
                for field in (p.text, *p.given, p.find)
                for t in _tokens(field) if t.is_math]

    def build(self, scene: ChalkdustScene) -> None:
        p: ProblemStatementParams = self.params
        theme = scene.theme
        cap = body_cap_height(theme)
        stage = region_rect(Region.STAGE).inset(DEFAULT_PADDING)
        measure = min(stage.width, MEASURE_CHARS * _char_width(theme))
        # GIVEN and FIND share the measure as two columns.
        column = (measure - COLUMN_GAP * cap) / 2

        lines = _Flow(p.text, theme, measure).lines()
        [statement] = _stack([lines], cap)
        label(statement, "statement")
        for i, line in enumerate(lines):
            label(line, f"statement[{i}]")

        def section(header: str, items: list[VGroup], x: float,
                    top: float) -> tuple[VGroup, Mobject]:
            """A captioned column, header over its items, top-left at (x, top).
            Both headers are caps-only captions, so side-by-side GIVEN and
            FIND columns line up without being aligned to each other."""
            head = caption_text(header, theme)
            head.next_to(items[0], UP, buff=HEADER_GAP * cap, aligned_edge=LEFT)
            block = VGroup(head, *items)
            block.shift(UP * (top - block.get_top()[1])
                        + RIGHT * (x - block.get_left()[0]))
            return block, head

        left = statement.get_left()[0]
        top = statement.get_bottom()[1] - SECTION_GAP * cap
        blocks = [statement]
        given_items = _stack([_Flow(g, theme, column).lines() for g in p.given], cap)
        for i, item in enumerate(given_items):
            label(item, f"given[{i}]")

        # FIND sits beside the givens, or under the statement when there are
        # none. A given whose maths is too wide to break overruns its column;
        # then FIND drops below the givens at the full measure, so the overrun
        # can never reach it.
        given_head = None
        find_x, find_top, find_width = left, top, measure
        if given_items:
            given_block, given_head = section("GIVEN", given_items, left, top)
            blocks.append(label(given_block, "given"))
            if given_block.get_right()[0] <= left + column + 1e-6:
                find_x, find_width = left + column + COLUMN_GAP * cap, column
            else:
                find_top = given_block.get_bottom()[1] - SECTION_GAP * cap
        [find_item] = _stack([_Flow(p.find, theme, find_width).lines()], cap)
        label(find_item, "find value")
        emphasize(find_item, theme)
        find_block, find_head = section("FIND", [find_item], find_x, find_top)
        blocks.append(label(find_block, "find"))

        fit_to_region(label(VGroup(*blocks), "ProblemStatement"), Region.STAGE)

        # Separate top-level mobjects so the settle check asserts the
        # statement and the columns never overlap; revealed by opacity, as
        # BulletReveal does, so the checks see the final geometry.
        for block in blocks:
            for mob in block.get_family():
                mob.set_opacity(0)
            scene.add(block)
        scene.exclusive(*blocks)

        times = iter(scene.budget(*self._weights(len(lines))))
        for line in lines:
            scene.play(line.animate.set_opacity(1), run_time=next(times))
        for i, item in enumerate(given_items):
            reveal = [given_head, item] if i == 0 else [item]
            scene.play(*(m.animate.set_opacity(1) for m in reveal), run_time=next(times))
        scene.play(find_head.animate.set_opacity(1), find_item.animate.set_opacity(1),
                   run_time=next(times))
        scene.settle("problem stated")
        scene.wait(next(times))

    @classmethod
    def examples(cls):
        return [
            {"text": r"A block of mass $m$ is released from rest at the top of a "
                     r"rough plane inclined at $\theta$ to the horizontal. The "
                     r"coefficient of kinetic friction between the block and the "
                     r"plane is $\mu$.",
             "given": [r"$m = 2\,\mathrm{kg}$", r"$\theta = 30^\circ$",
                       r"$\mu = 0.2$", r"$g = 10\,\mathrm{m\,s^{-2}}$"],
             "find": r"the acceleration $a$ of the block down the plane"},
            {"text": r"Evaluate the definite integral "
                     r"$\int_0^{\pi/2} \frac{\sin x}{\sin x + \cos x}\,dx$.",
             "find": r"the value of the integral $I$"},
            {"text": r"A particle moves along the $x$-axis so that its position "
                     r"at time $t$ seconds is $x(t) = t^3 - 6t^2 + 9t$ metres.",
             "given": [r"$x(t) = t^3 - 6t^2 + 9t$", r"$t \ge 0$"],
             "find": r"the times at which the particle is momentarily at rest, "
                     r"and the distance it travels between them"},
        ]

    @classmethod
    def stress(cls):
        incline = cls.examples()[0]
        return [
            # Three times a realistic statement, six long givens, a long find.
            {"text": " ".join([incline["text"]] * 3),
             "given": [r"the mass of the block on the incline is $m = 2\,\mathrm{kg}$ "
                       r"measured on a calibrated balance"] * 6,
             "find": "the acceleration of the block down the plane, the normal "
                     "reaction on it, and the work done against friction over "
                     r"the first $2\,\mathrm{m}$ of its slide"},
            # Unwrappable tokens: a 60-char identifier and a URL in prose, and
            # a 60-char maths chunk that cannot be broken at all.
            {"text": "The constant " + "k" * 60 + " is defined at "
                     "https://example.org/" + "a" * 40 + " for this problem.",
             "given": [r"$\mathrm{" + "X" * 60 + "}$"],
             "find": "the_value_of_" + "q" * 47},
            # Unbreakable givens wider than their column but not the stage:
            # FIND must drop below them, not collide.
            {"text": incline["text"],
             "given": [r"$v = \sqrt{2gh} = \sqrt{2 \times 10 \times 5} = "
                       r"10\,\mathrm{m\,s^{-1}}$",
                       r"$g = 10\,\mathrm{m\,s^{-2}}$"],
             "find": incline["find"]},
            # Minimal: one character each, no givens.
            {"text": "x", "find": "y"},
            {"text": "$x$", "given": ["$y$"], "find": "$z$"},
        ]
