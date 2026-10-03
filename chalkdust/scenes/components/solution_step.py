"""SolutionStep: one numbered step of a worked solution, with its work shown.

The JEE template is given -> approach -> derivation -> answer (PRD.md §6),
and this is one beat of the derivation. Each beat carries exactly one step,
read top to bottom the way a solution is written on a board:

    TITLE_BAR     (n) the claim -- what this step establishes
    STAGE         the work, one line at a time
    LOWER_THIRD   the justification -- why the work is allowed

The work is maths by default: each line is math-mode LaTeX, exactly as
EquationDerivation takes its steps. Some steps are reasoning rather than
algebra ("the block does not slip, so friction is static"), so `work_format`
switches the lines to plain prose. That flag is an addition to the key params
in SCENE_SPEC.md §5, and it is explicit on purpose: guessing whether a string
is LaTeX would make latex_strings() -- which the semantic rung compiles
standalone -- a guess too.
"""

from __future__ import annotations

from typing import Annotated, Literal

from manim import DOWN, LEFT, RIGHT, UP, Circle, FadeIn, VGroup, Write
from pydantic import Field, StringConstraints

from chalkdust.core.models import Region
from chalkdust.scenes.base import ChalkdustScene
from chalkdust.scenes.components.base import (
    Component,
    ComponentParams,
    label,
    register,
    wrap,
)
from chalkdust.scenes.regions import LayoutError, fit_to_region
from chalkdust.scenes.theme import (
    Theme,
    body_cap_height,
    body_text,
    heading_text,
    math,
)

# Blank strings are rejected at the schema rung: a blank claim is a step that
# says nothing, a blank work line is a silent gap, and a blank justification
# is a lower third with nothing in it.
NonBlank = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

# Characters per wrapped line. The claim sits beside the step badge in the
# title bar, so it wraps shorter than the full-width justification.
CLAIM_WRAP = 44
PROSE_WRAP = 56
JUSTIFY_WRAP = 64

# Spacing, in body cap heights so it scales with the theme (see bullet_reveal).
BADGE_PAD = 0.45    # ring clearance around the step number
BADGE_GAP = 1.1     # horizontal space between the badge and the claim
ROW_GAP = 1.6       # vertical space between one line of work and the next

# Minimum legible duration of each kind of segment, in seconds. These double
# as the budget weights: build() scales all of them by the same factor, so the
# beat stays legible exactly when the narration is at least min_seconds() long
# (SCENE_SPEC.md §8 rung 2 compares the two).
HEADER_S = 1.0      # badge and claim fading in
LINE_S = 1.0        # writing one line of work
READ_S = 0.5        # holding a fresh line still long enough to read it
JUSTIFY_S = 0.75    # fading in the justification
HOLD_S = 1.0        # the completed step, held for the end of the narration


class SolutionStepParams(ComponentParams):
    # The step's number in the solution. Two digits at most: the badge is sized
    # around the number, and a solution with 100 steps is not a video.
    n: int = Field(ge=1, le=99)
    # Plain text: what this step establishes ("Find the acceleration").
    claim: NonBlank
    # Lines of working, stacked downward. Capped at 6 -- three times a
    # realistic two-line step. More working than that is a derivation, and
    # EquationDerivation is the component built to pace one.
    work: list[NonBlank] = Field(min_length=1, max_length=6)
    # "math": each line is math-mode LaTeX. "text": each line is prose.
    work_format: Literal["math", "text"] = "math"
    # Why the work is valid ("Newton's second law along the incline").
    justification: NonBlank | None = None


@register
class SolutionStep(Component):
    name = "SolutionStep"
    Params = SolutionStepParams

    def regions(self) -> set[Region]:
        r = {Region.TITLE_BAR, Region.STAGE}
        if self.params.justification is not None:
            r.add(Region.LOWER_THIRD)
        return r

    # --- hooks for the semantic rung (SCENE_SPEC.md §8 rung 2) ---------------

    def latex_strings(self) -> list[str]:
        """Every LaTeX string build() compiles: the work lines, in maths mode
        only. Prose work and the claim are plain Text and compile nothing."""
        p: SolutionStepParams = self.params
        return list(p.work) if p.work_format == "math" else []

    def min_seconds(self) -> float:
        """Shortest narration at which every segment still gets its minimum."""
        return sum(self._segments())

    def _segments(self) -> list[float]:
        """Minimum durations of every timed segment, in the order build() plays
        them. Shared by build() and min_seconds() so they cannot drift apart."""
        p: SolutionStepParams = self.params
        out = [HEADER_S]
        for _ in p.work:
            out += [LINE_S, READ_S]
        if p.justification is not None:
            out.append(JUSTIFY_S)
        out.append(HOLD_S)
        return out

    # --- build ---------------------------------------------------------------

    def build(self, scene: ChalkdustScene) -> None:
        p: SolutionStepParams = self.params
        theme = scene.theme
        cap = body_cap_height(theme)

        # Header: the step number in an accent ring, then the claim. The ring
        # is sized from the number's own glyphs, so it follows the theme.
        number = heading_text(str(p.n), theme, theme.palette.accent)
        ring = Circle(radius=max(number.width, number.height) / 2 + cap * BADGE_PAD,
                      color=theme.palette.accent).move_to(number)
        badge = label(VGroup(ring, number), "badge")
        claim = label(heading_text(wrap(p.claim, CLAIM_WRAP), theme), "claim")
        claim.next_to(badge, RIGHT, buff=cap * BADGE_GAP)
        header = label(VGroup(badge, claim), "header")
        # Pinned left, like a heading in a written solution.
        fit_to_region(header, Region.TITLE_BAR, align=LEFT)

        # Work: one mobject per line, stacked at a constant gap. Maths lines
        # share a centre line, like a derivation on a board; prose lines share
        # a left edge, like a paragraph.
        lines = [_work_line(i, s, p.work_format, theme) for i, s in enumerate(p.work)]
        for prev, line in zip(lines, lines[1:]):
            line.next_to(prev, DOWN, buff=cap * ROW_GAP)
            if p.work_format == "math":
                line.match_x(lines[0])
            else:
                line.align_to(lines[0], LEFT)
        # One fit for every line so they scale together and keep their
        # alignment. Raises LayoutError (overflow) when the working is too long
        # or too wide to stay legible: the fix is splitting the step.
        work = label(VGroup(*lines), "work")
        fit_to_region(work, Region.STAGE)

        justification = None
        if p.justification is not None:
            justification = label(
                body_text(wrap(p.justification, JUSTIFY_WRAP), theme, theme.palette.muted),
                "justification",
            )
            fit_to_region(justification, Region.LOWER_THIRD)

        # Each work line is its own top-level scene mobject once revealed, so
        # the settle checks see each one -- including its tracked font size.
        scene.exclusive(header, *lines, *([justification] if justification else []))

        times = iter(scene.budget(*self._segments()))
        scene.play(FadeIn(header, shift=DOWN * cap), run_time=next(times))
        scene.settle("claim")

        for i, line in enumerate(lines):
            reveal = Write(line) if p.work_format == "math" else FadeIn(line, shift=UP * cap)
            scene.play(reveal, run_time=next(times))
            scene.settle(f"work[{i}]")
            scene.wait(next(times))

        if justification is not None:
            scene.play(FadeIn(justification, shift=UP * cap), run_time=next(times))
            scene.settle("justification")
        scene.wait(next(times))

    # --- fixtures ------------------------------------------------------------

    @classmethod
    def examples(cls):
        return [
            # Mechanics: a block on a smooth incline, with its justification.
            {"n": 2,
             "claim": "Find the acceleration of the block along the incline",
             "work": [r"mg\sin\theta = ma",
                      r"a = g\sin\theta = 9.8 \times \tfrac{1}{2} = 4.9\ \mathrm{m\,s^{-2}}"],
             "justification": "Newton's second law along the incline; the "
                              "surface is smooth, so no friction acts"},
            # Algebra: one line of work and no justification.
            {"n": 3,
             "claim": "Substitute the roots into the sum",
             "work": [r"\alpha + \beta = -\frac{b}{a} = -\frac{-7}{2} = \frac{7}{2}"]},
            # Reasoning step: the work is prose, not maths.
            {"n": 1,
             "claim": "Decide whether friction is static or kinetic",
             "work": ["The applied force is 8 N.",
                      "Limiting friction is mu N = 0.5 x 20 = 10 N.",
                      "8 N < 10 N, so the block stays at rest."],
             "work_format": "text",
             "justification": "Friction only becomes kinetic once the applied "
                              "force exceeds the limiting value"},
        ]

    @classmethod
    def stress(cls):
        # Invalid LaTeX is pinned in tests/test_component_solution_step.py
        # rather than here: it refuses with kind "invalid_latex", which the
        # registry-wide stress test does not (yet) count as a clean refusal.
        long_line = (r"\frac{m_1 u_1 + m_2 u_2}{m_1 + m_2} + "
                     r"\frac{m_1 m_2 (u_1 - u_2)^2}{2 (m_1 + m_2)} = v")
        long_text = ("Because both blocks move together the string stays taut, "
                     "so their accelerations are equal in magnitude throughout")
        token = "identifier" * 6
        return [
            # 3x realistic volume: the schema maximum of long maths lines, a
            # claim three sentences long, a justification at length.
            {"n": 12,
             "claim": "Apply conservation of momentum to the collision, then "
                      "use the energy equation to find the loss, then compare "
                      "it with the initial kinetic energy",
             "work": [long_line] * 6,
             "justification": long_text * 3},
            # 3x volume in prose mode.
            {"n": 7, "claim": long_text, "work": [long_text] * 6,
             "work_format": "text", "justification": long_text},
            # Unwrappable tokens everywhere: a 60-character identifier in the
            # claim and the maths, a space-free URL in the justification.
            {"n": 99, "claim": token,
             "work": [r"\mathrm{" + token + "} = 1"],
             "justification": "https://example.com/" + "a" * 60},
            {"n": 4, "claim": "Unwrappable prose", "work": [token],
             "work_format": "text"},
            # Minimal: one-character claim and one-character work.
            {"n": 1, "claim": "x", "work": ["x"]},
        ]


def _work_line(i: int, source: str, fmt: str, theme: Theme):
    """Build one line of work through the theme's constructors.

    Manim reports a LaTeX failure as a bare ValueError from deep inside its
    compile step. Converting it here gives the repair loop a typed refusal it
    can dispatch on (regenerate the spec) instead of a crash, and the probe
    reports it as a finding of that kind -- the same kind EquationDerivation
    raises.
    """
    if fmt == "text":
        return label(body_text(wrap(source, PROSE_WRAP), theme), f"work[{i}]")
    try:
        line = math(source, theme)
    except ValueError as exc:
        raise LayoutError(
            f"work[{i}] is not valid LaTeX: {source!r} ({exc})",
            kind="invalid_latex",
        ) from exc
    if line.width == 0 and line.height == 0:
        # Compiles, draws nothing (e.g. only spacing commands). Placing it would
        # leave a silent gap in the working.
        raise LayoutError(f"work[{i}] renders nothing: {source!r}", kind="invalid_latex")
    return label(line, f"work[{i}]")
