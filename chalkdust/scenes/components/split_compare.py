"""SplitCompare: two things side by side, with an optional verdict.

Each side is a card in its own stage half -- a title plus a line of text
and/or a short piece of maths. The verdict, when present, sits in the lower
third, because it is the conclusion of the comparison rather than a third
thing being compared.
"""

from __future__ import annotations

from itertools import accumulate
from typing import Annotated, Literal

from manim import DOWN, FadeIn, Mobject, RoundedRectangle, VGroup, config
from pydantic import StringConstraints, model_validator

from chalkdust.core.models import Region
from chalkdust.scenes.base import ChalkdustScene
from chalkdust.scenes.components.base import (
    Component,
    ComponentParams,
    label,
    register,
    wrap,
)
from chalkdust.scenes.regions import (
    DEFAULT_PADDING,
    MIN_FONT_SIZE,
    LayoutError,
    bbox,
    fit_to_region,
    region_rect,
    smallest_font_size,
)
from chalkdust.scenes.theme import Theme, body_text, heading_text, math

# Wrap widths in characters, sized so a typical side fits a stage half at
# natural size. fit_to_region still covers fonts that run wider than expected.
TITLE_WRAP = 18
BODY_WRAP = 24
VERDICT_WRAP = 40

CARD_PAD = 0.3      # space between a card's edge and its content
STACK_BUFF = 0.3    # vertical gap between title, body and maths
CARD_RADIUS = 0.15
CARD_STROKE = 2.0
CARD_STROKE_EMPHASIS = 4.0

# Relative step weights; budget() turns them into seconds (D-002).
REVEAL_WEIGHT = 2.0
VERDICT_WEIGHT = 1.5
HOLD_WEIGHT = 3.0
# Fraction of a frame used to steer Manim's frame rounding; see build().
FRAME_EPS = 0.01

# Per-step floors, in seconds, below which a step stops reading as a step:
# a side needs long enough to register its title, and the finished frame needs
# a beat of stillness before the cut. The semantic rung (SCENE_SPEC.md §8)
# compares narration duration against their sum.
REVEAL_MIN_SECONDS = 0.6
VERDICT_MIN_SECONDS = 0.6
HOLD_MIN_SECONDS = 1.0

# Whitespace-only text is as empty as "" on screen, so strip before checking.
NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Side(ComponentParams):
    """One side of the comparison. A ComponentParams so it is frozen and
    forbids unknown keys like the outer model -- a stray key inside `left`
    must fail as loudly as one at the top level."""

    title: NonEmpty
    body: NonEmpty | None = None
    # Short LaTeX, rendered under the body. Optional param beyond the §5 key
    # set: SplitCompare is the natural home for "O(1) vs O(n)" and the like.
    math: NonEmpty | None = None


class SplitCompareParams(ComponentParams):
    left: Side
    right: Side
    verdict: NonEmpty | None = None
    # Which element gets the accent colour (SCENE_SPEC.md §3). None means
    # nothing is singled out; the comparison is neutral.
    emphasis: Literal["left", "right", "verdict"] | None = None

    @model_validator(mode="after")
    def _emphasis_has_target(self) -> SplitCompareParams:
        if self.emphasis == "verdict" and self.verdict is None:
            raise ValueError("emphasis='verdict' requires a verdict")
        return self


@register
class SplitCompare(Component):
    name = "SplitCompare"
    Params = SplitCompareParams

    def regions(self) -> set[Region]:
        r = {Region.STAGE_LEFT, Region.STAGE_RIGHT}
        if self.params.verdict is not None:
            r.add(Region.LOWER_THIRD)
        return r

    def min_seconds(self) -> float:
        total = 2 * REVEAL_MIN_SECONDS + HOLD_MIN_SECONDS
        if self.params.verdict is not None:
            total += VERDICT_MIN_SECONDS
        return total

    def latex_strings(self) -> list[str]:
        return [s.math for s in (self.params.left, self.params.right) if s.math]

    def build(self, scene: ChalkdustScene) -> None:
        p: SplitCompareParams = self.params
        theme = scene.theme
        halves = (("left", p.left, Region.STAGE_LEFT),
                  ("right", p.right, Region.STAGE_RIGHT))

        contents = [_side_content(side, theme, p.emphasis == key)
                    for key, side, _ in halves]

        # One scale for both sides. Fitting each independently would render
        # the busier side in smaller type, and the comparison would read as
        # lopsided before the viewer has read a word.
        scale = 1.0
        for content, (_, _, region) in zip(contents, halves):
            inner = region_rect(region).inset(DEFAULT_PADDING + CARD_PAD)
            box = bbox(content)
            scale = min(scale, inner.width / box.width, inner.height / box.height)
        if scale < 1.0:
            for content in contents:
                _scale(content, scale)
        smallest = min(smallest_font_size(c) for c in contents)
        if smallest < MIN_FONT_SIZE:
            raise LayoutError(
                f"SplitCompare sides do not fit legibly: smallest text would "
                f"render at font_size {smallest:.1f} (floor is "
                f"{MIN_FONT_SIZE:.0f}) after scaling by {scale:.2f}. Shorten "
                f"the sides or split the comparison across beats.",
                kind="overflow",
            )

        # Equal-height cards, content hung from the top, so the two titles sit
        # on one line however much text is under each.
        card_height = max(c.height for c in contents) + 2 * CARD_PAD
        groups = []
        for content, (key, _, region) in zip(contents, halves):
            emphasized = p.emphasis == key
            card = RoundedRectangle(
                width=region_rect(region).width - 2 * DEFAULT_PADDING,
                height=card_height,
                corner_radius=CARD_RADIUS,
                stroke_color=theme.palette.accent if emphasized else theme.palette.muted,
                stroke_width=CARD_STROKE_EMPHASIS if emphasized else CARD_STROKE,
            )
            content.next_to(card.get_top(), DOWN, buff=CARD_PAD)
            group = label(VGroup(card, content), f"{key} side")
            # Content is already scaled; this places the card and re-checks
            # it against the region and the legibility floor.
            fit_to_region(group, region)
            groups.append(group)

        verdict = None
        if p.verdict is not None:
            colour = theme.palette.accent if p.emphasis == "verdict" else None
            verdict = label(
                heading_text(wrap(p.verdict, VERDICT_WRAP), theme, colour),
                "verdict",
            )
            fit_to_region(verdict, Region.LOWER_THIRD)

        scene.exclusive(*groups, *([verdict] if verdict is not None else []))

        weights = [REVEAL_WEIGHT, REVEAL_WEIGHT]
        if verdict is not None:
            weights.append(VERDICT_WEIGHT)
        weights.append(HOLD_WEIGHT)
        fps = config.frame_rate
        frames = _whole_frames(scene.budget(*weights), fps)

        def play_for(n: int) -> float:
            # play() renders ceil(run_time * fps) frames; just under n lands
            # on exactly n without float noise tipping it to n + 1. One frame
            # is Manim's floor, so ask for it outright rather than be clamped.
            return max(n - FRAME_EPS, 1) / fps

        for group, n in zip(groups, frames):
            scene.play(FadeIn(group, shift=DOWN * 0.2), run_time=play_for(n))
        if verdict is not None:
            scene.play(FadeIn(verdict), run_time=play_for(frames[2]))
        scene.settle("split compare revealed")
        # A frozen wait renders floor(duration * fps) frames, so aim just over.
        scene.wait((frames[-1] + FRAME_EPS) / fps, frozen_frame=True)

    @classmethod
    def examples(cls):
        return [
            # SCENE_SPEC.md §3, verbatim: the spec's own example must validate.
            {"left": {"title": "key: \"cat\"", "body": "hash → 4"},
             "right": {"title": "key: \"act\"", "body": "hash → 4"},
             "verdict": "same bucket",
             "emphasis": "verdict"},
            {"left": {"title": "Array lookup", "body": "Jump straight to the index",
                      "math": r"O(1)"},
             "right": {"title": "Linked list", "body": "Walk node by node from the head",
                       "math": r"O(n)"},
             "emphasis": "left"},
            {"left": {"title": "Before", "math": r"(a+b)^2"},
             "right": {"title": "After", "math": r"a^2 + 2ab + b^2"},
             "verdict": "Same value, expanded"},
        ]

    @classmethod
    def stress(cls):
        long_body = ("A side carrying far more explanation than a comparison "
                     "card is meant to hold, the kind of paragraph that belongs "
                     "in its own beat rather than squeezed beside another one")
        return [
            # 3x realistic volume on every slot at once.
            {"left": {"title": "An unusually long title for one side of it",
                      "body": long_body, "math": r"\sum_{i=1}^{n} i = \frac{n(n+1)}{2}"},
             "right": {"title": "And an equally long title on the other side",
                       "body": long_body, "math": r"\int_0^1 x^2 \, dx = \frac{1}{3}"},
             "verdict": "A verdict that also goes on far longer than any "
                        "conclusion needs to, then keeps going past that",
             "emphasis": "verdict"},
            # Unwrappable tokens: a 60-char identifier and a URL with no spaces.
            {"left": {"title": "x" * 60,
                      "body": "https://example.com/a/very/long/path/without/any/spaces"},
             "right": {"title": "ok", "body": "y" * 60},
             "verdict": "z" * 60},
            # Minimal: titles only, single characters, no verdict.
            {"left": {"title": "a"}, "right": {"title": "b"}},
            # Lopsided: one side dense, the other nearly empty. The shared
            # scale must keep both sides legible and the titles aligned.
            {"left": {"title": "Dense", "body": long_body},
             "right": {"title": "Sparse"},
             "emphasis": "right"},
        ]


def _side_content(side: Side, theme: Theme, emphasized: bool) -> VGroup:
    """Title, then body, then maths, stacked and centred."""
    parts: list[Mobject] = [
        heading_text(wrap(side.title, TITLE_WRAP), theme,
                     theme.palette.accent if emphasized else None)
    ]
    if side.body:
        parts.append(body_text(wrap(side.body, BODY_WRAP), theme))
    if side.math:
        parts.append(_math(side.math, theme))
    return VGroup(*parts).arrange(DOWN, buff=STACK_BUFF)


def _math(tex: str, theme: Theme) -> Mobject:
    """Compile maths, turning a LaTeX failure into a typed LayoutError.

    Manim raises a bare ValueError when latex exits non-zero. Left alone that
    reaches the validator as a build_error -- indistinguishable from a bug in
    this component -- when the real fault is the spec's maths, which the repair
    loop can fix by regenerating it. A missing latex binary (FileNotFoundError)
    is deliberately not caught: that is the environment, not the spec.
    """
    try:
        return math(tex, theme)
    except ValueError as exc:
        raise LayoutError(f"LaTeX failed to compile: {tex!r} ({exc})",
                          kind="latex") from exc


def _whole_frames(times: list[float], fps: float) -> list[int]:
    """Convert a budget split into whole frame counts summing to the beat.

    Manim renders every play() and wait() as a whole number of frames, rounding
    each step on its own. Four steps can drift several frames off the audio,
    and the drift compounds across beats at concat (D-002). Rounding the
    cumulative boundaries instead keeps the total within half a frame of the
    budget; every step keeps at least one frame so it still exists.
    """
    bounds = [round(t * fps) for t in accumulate(times)]
    return [max(1, b - a) for a, b in zip([0, *bounds], bounds)]


def _scale(mob: Mobject, factor: float) -> None:
    """Scale and carry the font-size tags along, as fit_to_region does, so the
    legibility check sees the size the text will actually render at."""
    mob.scale(factor)
    for m in mob.get_family():
        size = getattr(m, "_chalk_font_size", None)
        if size is not None:
            m._chalk_font_size = size * factor  # type: ignore[attr-defined]
