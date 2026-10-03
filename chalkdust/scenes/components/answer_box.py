"""AnswerBox: the final answer of a worked problem, boxed and unmistakable.

The last beat of a JEE solution (PRD.md §6: given -> approach -> derivation ->
answer). The answer is written large, its option letter (for MCQs) sits in a
badge beside it, and a box is drawn around both -- the board-work gesture for
"this is the result" -- under a small ANSWER label.

Three shapes of answer, all from the same three params:

    numerical:      value + units          T = 24 N
    MCQ:            option + value(+units) (B)  T = 24 N
    conceptual MCQ: option alone           (C)      -- the options are statements

Value and units are compiled as ONE MathTex, so the units sit on the value's
baseline whatever the value's height (a tall fraction, a square root).

The settled answer can be carried into a later beat (SCENE_SPEC.md §6): the
artifact builder below rebuilds it from the same params through the same
`_parts()` that build() animates, so the carried copy is the answer the viewer
saw.
"""

from __future__ import annotations

from typing import Annotated

from manim import (
    LEFT,
    UP,
    Circle,
    Create,
    FadeIn,
    GrowFromCenter,
    MathTex,
    Mobject,
    SurroundingRectangle,
    VGroup,
    Write,
)
from pydantic import StringConstraints, model_validator

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
    Rect,
    fit_to_region,
    region_rect,
)
from chalkdust.scenes.theme import (
    Theme,
    body_cap_height,
    caption_text,
    math,
    refuse_invalid_latex,
    title_text,
)

# Blank strings are rejected at the schema rung: an empty value compiles to
# nothing, and empty units are just absent units.
NonBlank = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

# One capital letter. JEE MCQs are four-option; anything longer than a letter
# would not fit the badge and is not an option label.
OptionLetter = Annotated[str, StringConstraints(pattern=r"^[A-D]$")]

# Spacing, in body cap heights so it scales with the theme (see bullet_reveal).
BOX_PAD = 1.4       # breathing room between the box and what it encloses
BADGE_GAP = 1.6     # horizontal space between the option badge and the value
LABEL_GAP = 0.9     # vertical space between the ANSWER label and the box
BADGE_RING = 0.8    # badge radius, as a multiple of the letter's larger side

# Minimum legible duration of each segment, in seconds. These double as the
# budget weights: build() scales all of them by the same factor, so the beat
# stays legible exactly when the narration is at least min_seconds() long
# (SCENE_SPEC.md §8 rung 2 compares the two).
REVEAL_S = 1.0      # writing the answer (or growing the badge, if it IS the answer)
OPTION_S = 0.5      # growing the option badge beside an already-written value
BOX_S = 0.75        # drawing the box and fading in the ANSWER label
DWELL_S = 1.0       # holding the finished answer still long enough to read it


class AnswerBoxParams(ComponentParams):
    # LaTeX maths, e.g. "T = 24" or r"\frac{mg}{k}". Optional only so that a
    # conceptual MCQ (option alone) is expressible; see the validator.
    value: NonBlank | None = None
    # LaTeX, set upright. Whitespace separates unit factors, as in SI notation:
    # "kg m s^{-2}" renders with thin spaces between the factors -- in raw maths
    # mode the spaces would vanish and "kgms" would be ambiguous.
    units: NonBlank | None = None
    # MCQ option letter. None for integer/numerical-type questions.
    option: OptionLetter | None = None

    @model_validator(mode="after")
    def _something_to_show(self) -> AnswerBoxParams:
        if self.value is None and self.option is None:
            raise ValueError("an answer needs a value, an option, or both")
        if self.units is not None and self.value is None:
            # Units with nothing to qualify: almost certainly a value the model
            # forgot, and boxing bare units would state a wrong answer.
            raise ValueError("units given without a value")
        return self


@register
class AnswerBox(Component):
    name = "AnswerBox"
    Params = AnswerBoxParams

    def regions(self) -> set[Region]:
        # The answer is the whole frame's point; it is centred on the stage.
        return {Region.STAGE}

    # --- hooks for the semantic rung (SCENE_SPEC.md §8 rung 2) ---------------

    def latex_strings(self) -> list[str]:
        """The one LaTeX string build() compiles -- value and units together,
        exactly as passed to MathTex -- or nothing for an option-only answer."""
        source = self._source()
        return [source] if source is not None else []

    def min_seconds(self) -> float:
        """Shortest narration at which every segment still gets its minimum."""
        return sum(self._segments())

    def _segments(self) -> list[float]:
        """Minimum durations of every timed segment, in the order build() plays
        them. Shared by build() and min_seconds() so they cannot drift apart."""
        p: AnswerBoxParams = self.params
        out = [REVEAL_S]  # the value, or the badge when the option is the answer
        if p.value is not None and p.option is not None:
            out.append(OPTION_S)
        out += [BOX_S, DWELL_S]
        return out

    def _source(self) -> str | None:
        return _source(self.params)

    # --- build ---------------------------------------------------------------

    def build(self, scene: ChalkdustScene) -> None:
        theme = scene.theme
        cap = body_cap_height(theme)
        tag, box, contents = _parts(self.params, theme)
        boxed = VGroup(box, VGroup(*contents))

        # The label stays OUT of the fitted group: it is caption-sized, so
        # scaling it with the answer would refuse at a ~0.9 scale -- long
        # before the answer itself is anywhere near illegible. Reserve a band
        # for it at the top of the stage and fit the box below.
        stage = region_rect(Region.STAGE)
        band = tag.height + cap * LABEL_GAP
        below = Rect(stage.x, stage.y - band / 2, stage.width, stage.height - band)
        # Raises LayoutError (overflow) when the answer cannot stay legible.
        fit_to_region(boxed, below)
        tag.next_to(box, UP, buff=cap * LABEL_GAP)

        # Centre the whole composition, label included, on the stage. It fits:
        # it is no taller than the band plus the padded box area.
        VGroup(tag, boxed).move_to(stage.center)

        # The box encloses the row by design, so only these must never touch.
        scene.exclusive(tag, *contents)

        # Whole-frame run times that sum to the beat exactly (scene.budget).
        times = iter(scene.budget(*self._segments()))
        parts = {m._chalk_label: m for m in contents}
        if "answer" in parts:
            scene.play(Write(parts["answer"]), run_time=next(times))
        if "option" in parts:
            scene.play(GrowFromCenter(parts["option"]), run_time=next(times))
        scene.play(Create(box), FadeIn(tag, shift=UP * cap / 2), run_time=next(times))
        scene.settle("answer boxed")
        scene.wait(next(times))

    # --- fixtures ------------------------------------------------------------

    @classmethod
    def examples(cls):
        return [
            # MCQ with a numerical value: the full form.
            {"value": "T = 24", "units": "N", "option": "B"},
            # Numerical-type answer; a tall root over a fraction, compound units.
            {"value": r"v = \sqrt{\frac{2gh}{1 + k^2/r^2}}", "units": "m s^{-1}"},
            # Conceptual MCQ: the option is the whole answer.
            {"option": "C"},
        ]

    @classmethod
    def stress(cls):
        long_value = (r"a = \frac{(m_1 - m_2)\, g \sin\theta - \mu (m_1 + m_2)\, g \cos\theta}"
                      r"{m_1 + m_2 + I / r^2}")
        return [
            # 3x realistic volume: three long expressions chained, verbose
            # compound units, and an option badge.
            {"value": " = ".join([long_value] * 3),
             "units": "kg m^{2} s^{-2} mol^{-1} K^{-1} A^{-1}",
             "option": "D"},
            # Unwrappable tokens: a 60-character identifier as the value, then
            # a space-free URL as the units.
            {"value": r"\mathrm{" + "identifier" * 6 + "}"},
            {"value": "x = 1", "units": "https://example.com/" + "a" * 40},
            # Minimal: one character, no units, no option.
            {"value": "0"},
            # Minimal MCQ: option alone.
            {"option": "A"},
            # Invalid LaTeX: each must refuse as "invalid_latex", never box an
            # answer nobody wrote.
            {"value": r"\notacommand{x} = 1"},             # does not compile
            {"value": r"\quad"},                           # compiles, draws nothing
            {"value": r"\frac{1}{"},                       # LaTeX silently recovers
            {"value": "x = 1", "units": r"m}\frac{1}{2"},   # escapes the \mathrm wrapper
            {"value": "50%", "units": "N"},                # % is a comment, not percent
        ]


def _source(params: AnswerBoxParams) -> str | None:
    """The one LaTeX string compiled for value and units, or None when the
    option is the whole answer."""
    if params.value is None:
        return None
    if params.units is None:
        return params.value
    units = r"\,".join(params.units.split())
    # A thick space before the units, as in typeset physics: "24 N".
    return rf"{params.value} \; \mathrm{{{units}}}"


def _parts(params: AnswerBoxParams, theme: Theme) -> tuple[Mobject, Mobject, list[Mobject]]:
    """(label, box, contents) at natural size: the settled answer, unplaced.

    Shared by build(), which fits and animates it, and the carry-in artifact
    builder, which returns it still -- so both draw the same answer.
    contents holds the option badge and the answer, whichever are present,
    badge first.
    """
    cap = body_cap_height(theme)
    # Value and badge share the title size, so one font size governs how far
    # the fit may shrink them before refusing.
    source = _source(params)
    answer = _answer_tex(source, params, theme) if source is not None else None
    badge = _badge(params.option, theme) if params.option is not None else None

    contents = [m for m in (badge, answer) if m is not None]
    if badge is not None and answer is not None:
        badge.next_to(answer, LEFT, buff=cap * BADGE_GAP)
    box = label(SurroundingRectangle(VGroup(*contents), color=theme.palette.accent,
                                     buff=cap * BOX_PAD, corner_radius=cap),
                "box")
    tag = label(caption_text("ANSWER", theme, theme.palette.accent), "label")
    return tag, box, contents


@artifact_builder("AnswerBox")
def _artifact(params: AnswerBoxParams, theme: Theme) -> Mobject:
    """The settled answer for a later beat's carry_in (SCENE_SPEC.md §6):
    label above the box, at natural size. CarryIn places and dims it."""
    tag, box, contents = _parts(params, theme)
    tag.next_to(box, UP, buff=body_cap_height(theme) * LABEL_GAP)
    return VGroup(tag, box, *contents)


def _answer_tex(source: str, params: AnswerBoxParams, theme: Theme) -> MathTex:
    """Compile the answer through theme.math, at title size.

    theme.math refuses LaTeX that does not compile or draws nothing (kind
    "invalid_latex"). Two faults are refused here first, through the same
    refuse_invalid_latex, because theme.math alone gets them wrong:

      - unbalanced braces COMPILE and draw the wrong answer: LaTeX recovers
        from "\\frac{1}{" by drawing a lone 1, and a stray "}" in the units
        closes the \\mathrm wrapper early;
      - an unescaped % comments out the rest of the line, closing brace and
        environment end included; the compile fails, but Manim's error blames
        dvisvgm. The repair loop needs the real fault: write \\% for percent.
    """
    for field, text in (("value", params.value), ("units", params.units)):
        if text is None:
            continue
        if not _braces_balance(text):
            raise refuse_invalid_latex(field, text, "has unbalanced braces")
        if _has_comment(text):
            raise refuse_invalid_latex(
                field, text, "has an unescaped % (a LaTeX comment; percent is \\%)")
    return label(math(source, theme, size=theme.type.title, what="answer"), "answer")


def _badge(option: str, theme: Theme) -> VGroup:
    """The option letter, ringed in the accent colour.

    Title-sized like the value, so the badge never sets a stricter legibility
    floor than the answer it sits beside.
    """
    letter = title_text(option, theme, theme.palette.accent)
    ring = Circle(radius=BADGE_RING * max(letter.width, letter.height),
                  color=theme.palette.accent)
    ring.move_to(letter)
    return label(VGroup(ring, letter), "option")


def _braces_balance(text: str) -> bool:
    """True when every unescaped { has a matching } after it."""
    depth = 0
    escaped = False
    for ch in text:
        if escaped:
            escaped = False
        elif ch == "\\":
            escaped = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def _has_comment(text: str) -> bool:
    """True when the text has a % not escaped as \\%."""
    escaped = False
    for ch in text:
        if escaped:
            escaped = False
        elif ch == "\\":
            escaped = True
        elif ch == "%":
            return True
    return False
