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
    SurroundingRectangle,
    VGroup,
    Write,
)
from pydantic import StringConstraints, model_validator

from chalkdust.core.models import Region
from chalkdust.scenes.base import ChalkdustScene
from chalkdust.scenes.components.base import (
    Component,
    ComponentParams,
    label,
    register,
)
from chalkdust.scenes.regions import (
    LayoutError,
    Rect,
    fit_to_region,
    region_rect,
)
from chalkdust.scenes.theme import (
    Theme,
    body_cap_height,
    caption_text,
    math,
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
        p: AnswerBoxParams = self.params
        if p.value is None:
            return None
        if p.units is None:
            return p.value
        units = r"\,".join(p.units.split())
        # A thick space before the units, as in typeset physics: "24 N".
        return rf"{p.value} \; \mathrm{{{units}}}"

    # --- build ---------------------------------------------------------------

    def build(self, scene: ChalkdustScene) -> None:
        p: AnswerBoxParams = self.params
        theme = scene.theme
        cap = body_cap_height(theme)

        # Value and badge share the title size, so one font size governs how
        # far the fit may shrink them before refusing.
        answer = None
        source = self._source()
        if source is not None:
            answer = _answer_tex(source, self.params, theme)
        badge = _badge(p.option, theme) if p.option is not None else None

        contents = [m for m in (badge, answer) if m is not None]
        if badge is not None and answer is not None:
            badge.next_to(answer, LEFT, buff=cap * BADGE_GAP)
        row = VGroup(*contents)
        box = label(SurroundingRectangle(row, color=theme.palette.accent,
                                         buff=cap * BOX_PAD, corner_radius=cap),
                    "box")
        boxed = VGroup(box, row)

        # The label stays OUT of the fitted group: it is caption-sized, so
        # scaling it with the answer would refuse at a ~0.9 scale -- long
        # before the answer itself is anywhere near illegible. Reserve a band
        # for it at the top of the stage and fit the box below.
        tag = label(caption_text("ANSWER", theme, theme.palette.accent), "label")
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

        times = iter(scene.budget(*self._segments()))
        if answer is not None:
            scene.play(Write(answer), run_time=next(times))
        if badge is not None:
            scene.play(GrowFromCenter(badge), run_time=next(times))
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
        # Invalid LaTeX is pinned in tests/test_component_answer_box.py rather
        # than here: it refuses with kind "invalid_latex", which the
        # registry-wide stress test does not (yet) count as a clean refusal.
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
        ]


def _answer_tex(source: str, params: AnswerBoxParams, theme: Theme) -> MathTex:
    """Compile the answer through the theme's maths constructor, at title size.

    Manim reports a LaTeX failure as a bare ValueError from deep inside its
    compile step. Converting it here gives the repair loop a typed refusal it
    can dispatch on (regenerate the spec) instead of a crash, and the probe
    reports it as a finding of that kind.
    """
    # LaTeX recovers from some unbalanced braces without an error and draws
    # something else ("\frac{1}{" becomes a lone 1), and a stray "}" in the
    # units would close the \mathrm wrapper early. Either way the box would
    # state an answer nobody wrote, so refuse before compiling.
    for field, text in (("value", params.value), ("units", params.units)):
        if text is not None and not _braces_balance(text):
            raise LayoutError(f"{field} has unbalanced braces: {text!r}",
                              kind="invalid_latex")
    try:
        tex = math(source, theme, size=theme.type.title)
    except ValueError as exc:
        raise LayoutError(f"answer is not valid LaTeX: {source!r} ({exc})",
                          kind="invalid_latex") from exc
    if tex.width == 0 and tex.height == 0:
        # Compiles, draws nothing (e.g. only spacing commands): an empty box.
        raise LayoutError(f"answer renders nothing: {source!r}", kind="invalid_latex")
    return label(tex, "answer")


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
