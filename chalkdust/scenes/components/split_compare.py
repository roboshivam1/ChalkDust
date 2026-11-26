"""SplitCompare: two things side by side, with an optional verdict.

Each side is a card in its own stage half -- a title plus a line of text
and/or a short piece of maths. The verdict, when present, sits in the lower
third, because it is the conclusion of the comparison rather than a third
thing being compared.

The settled frame (both cards and the verdict) is also this component's
carry-in artifact (SCENE_SPEC.md §6): a later beat that carries it in gets the
same comparison rebuilt from these params, by the same layout code build()
uses.
"""

from __future__ import annotations

from typing import Annotated, Literal

from manim import DOWN, FadeIn, Mobject, RoundedRectangle, VGroup
from pydantic import StringConstraints, model_validator

from chalkdust.continuity import artifact_builder
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
    scale_with_tags,
    smallest_font_size,
)
from chalkdust.scenes.theme import Theme, body_text, heading_text, math

# Wrap widths in characters, sized so a typical side fits a stage half at
# natural size. fit_to_region still covers fonts that run wider than expected.
TITLE_WRAP = 18
BODY_WRAP = 26
VERDICT_WRAP = 40

CARD_PAD = 0.3      # space between a card's edge and its content
STACK_BUFF = 0.3    # vertical gap between title, body and maths
CARD_RADIUS = 0.15
CARD_STROKE = 2.0
CARD_STROKE_EMPHASIS = 4.0

# Shortest legible duration of each segment, in seconds: a side needs long
# enough to register its title, and the finished frame needs a beat of
# stillness before the cut. These double as the budget weights -- build()
# scales all of them by one factor -- so every segment gets at least its
# minimum exactly when the narration is at least min_seconds() long, which is
# what the semantic rung (SCENE_SPEC.md §8 rung 2) checks.
REVEAL_S = 0.6      # fading in one side
VERDICT_S = 0.6     # fading in the verdict
HOLD_S = 1.0        # holding the finished comparison

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

    # --- hooks for the semantic rung (SCENE_SPEC.md §8 rung 2) ---------------

    def min_seconds(self) -> float:
        """Shortest narration at which every segment still gets its minimum."""
        return sum(self._segments())

    def latex_strings(self) -> list[str]:
        """The maths of each side, exactly as build() compiles it."""
        return [s.math for s in (self.params.left, self.params.right) if s.math]

    def _segments(self) -> list[float]:
        """Minimum durations of every timed segment, in the order build() plays
        them. Shared by build() and min_seconds() so they cannot drift apart."""
        out = [REVEAL_S, REVEAL_S]
        if self.params.verdict is not None:
            out.append(VERDICT_S)
        out.append(HOLD_S)
        return out

    # --- build ---------------------------------------------------------------

    def build(self, scene: ChalkdustScene) -> None:
        sides, verdict = _layout(self.params, scene.theme)
        scene.exclusive(*sides, *([verdict] if verdict is not None else []))

        # Whole-frame run times summing to the beat (D-002); the base scene
        # renders each as exactly that many frames.
        times = iter(scene.budget(*self._segments()))
        for side in sides:
            scene.play(FadeIn(side, shift=DOWN * 0.2), run_time=next(times))
        if verdict is not None:
            scene.play(FadeIn(verdict), run_time=next(times))
        scene.settle("split compare revealed")
        scene.wait(next(times), frozen_frame=True)

    # --- fixtures ------------------------------------------------------------

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
            # 3x realistic volume on every slot at once: refuses as overflow.
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
            # Invalid LaTeX refuses as "invalid_latex", naming the side:
            # an undefined command on the left...
            {"left": {"title": "Broken", "math": r"\notacommand{x}"},
             "right": {"title": "Fine", "math": r"x^2"}},
            # ...a dangling superscript on the right...
            {"left": {"title": "Fine", "math": r"x^2"},
             "right": {"title": "Broken", "math": r"x^"}},
            # ...and LaTeX that compiles but draws nothing.
            {"left": {"title": "Empty maths", "math": r"\quad"},
             "right": {"title": "Fine"}},
            # Text that survives whitespace stripping but draws no glyph
            # (U+200B zero-width space) refuses as "illegible", naming the
            # field -- never a blank card, never a zero-size division.
            {"left": {"title": "\u200b"}, "right": {"title": "R"}},
            # ...the same for a body (U+2060 word joiner)...
            {"left": {"title": "L", "body": "\u2060"}, "right": {"title": "R"}},
            # ...and for the verdict (U+FEFF byte-order mark).
            {"left": {"title": "L"}, "right": {"title": "R"}, "verdict": "\ufeff"},
        ]


@artifact_builder(SplitCompare.name)
def _artifact(params: SplitCompareParams, theme: Theme) -> Mobject:
    """The settled comparison -- both cards and the verdict -- for carry-in.
    Built by the same layout code as build(), so a carried SplitCompare is the
    frame the producing beat ended on (CarryIn re-places it in STAGE)."""
    sides, verdict = _layout(params, theme)
    return VGroup(*sides, *([verdict] if verdict is not None else []))


# --- layout -------------------------------------------------------------------


def _layout(p: SplitCompareParams, theme: Theme) -> tuple[list[Mobject], Mobject | None]:
    """Both side cards placed in their stage halves, and the verdict (if any)
    in the lower third. Pure: same params and theme, same mobjects.

    Raises LayoutError: "overflow" when the sides cannot fit legibly,
    "invalid_latex" (via theme.math) when a side's maths cannot be drawn,
    "illegible" when a title, body or the verdict draws no glyphs at all.
    """
    halves = (("left", p.left, Region.STAGE_LEFT),
              ("right", p.right, Region.STAGE_RIGHT))

    contents = [_side_content(key, side, theme, p.emphasis == key)
                for key, side, _ in halves]

    # One scale for both sides. Fitting each independently would render the
    # busier side in smaller type, and the comparison would read as lopsided
    # before the viewer has read a word.
    scale = 1.0
    for content, (_, _, region) in zip(contents, halves):
        inner = region_rect(region).inset(DEFAULT_PADDING + CARD_PAD)
        box = bbox(content)
        # _side_content refuses a side that draws nothing, but a part can
        # still be flat in one axis; only a real extent constrains the scale
        # (as in fit_to_region).
        if box.width > 0:
            scale = min(scale, inner.width / box.width)
        if box.height > 0:
            scale = min(scale, inner.height / box.height)
    if scale < 1.0:
        for content in contents:
            scale_with_tags(content, scale)
    smallest = min(smallest_font_size(c) for c in contents)
    if smallest < MIN_FONT_SIZE:
        raise LayoutError(
            f"SplitCompare sides do not fit legibly: smallest text would "
            f"render at font_size {smallest:.1f} (floor is "
            f"{MIN_FONT_SIZE:.0f}) after scaling by {scale:.2f}. Shorten "
            f"the sides or split the comparison across beats.",
            kind="overflow",
        )

    # Equal-height cards, content hung from the top, so the two titles sit on
    # one line however much text is under each.
    card_height = max(c.height for c in contents) + 2 * CARD_PAD
    sides = []
    for content, (key, _, region) in zip(contents, halves):
        emphasized = p.emphasis == key
        # Filled with the background colour: invisible on an empty stage, but
        # a carried-in artifact dimmed beneath (SCENE_SPEC.md §6) would
        # otherwise show through and collide with this side's text.
        card = RoundedRectangle(
            fill_color=theme.palette.bg,
            fill_opacity=1.0,
            width=region_rect(region).width - 2 * DEFAULT_PADDING,
            height=card_height,
            corner_radius=CARD_RADIUS,
            stroke_color=theme.palette.accent if emphasized else theme.palette.muted,
            stroke_width=CARD_STROKE_EMPHASIS if emphasized else CARD_STROKE,
        )
        content.next_to(card.get_top(), DOWN, buff=CARD_PAD)
        side = label(VGroup(card, content), f"{key} side")
        # Content is already scaled; this places the card and re-checks it
        # against the region and the legibility floor.
        fit_to_region(side, region)
        sides.append(side)

    verdict = None
    if p.verdict is not None:
        colour = theme.palette.accent if p.emphasis == "verdict" else None
        verdict = label(_drawn(heading_text(_wrap_balanced(p.verdict, VERDICT_WRAP),
                                            theme, colour, what="SplitCompare verdict"),
                               p.verdict, "verdict"),
                        "verdict")
        fit_to_region(verdict, Region.LOWER_THIRD)
    return sides, verdict


def _side_content(key: str, side: Side, theme: Theme, emphasized: bool) -> VGroup:
    """Title, then body, then maths, stacked and centred."""
    parts: list[Mobject] = [
        _drawn(heading_text(_wrap_balanced(side.title, TITLE_WRAP), theme,
                            theme.palette.accent if emphasized else None,
                            what=f"SplitCompare {key}.title"),
               side.title, f"{key}.title")
    ]
    if side.body:
        parts.append(_drawn(body_text(_wrap_balanced(side.body, BODY_WRAP), theme,
                                      what=f"SplitCompare {key}.body"),
                            side.body, f"{key}.body"))
    if side.math:
        # theme.math refuses LaTeX that does not compile, or draws nothing, as
        # LayoutError kind "invalid_latex" naming the side.
        parts.append(math(side.math, theme, what=f"{key}.math"))
    return VGroup(*parts).arrange(DOWN, buff=STACK_BUFF)


def _drawn(text: Mobject, source: str, what: str) -> Mobject:
    """`text`, unless it draws nothing. The schema strips whitespace, but
    Pango also draws no glyph for zero-width characters (U+200B, U+2060,
    U+FEFF) and for characters the resolved font lacks. Such a part would
    leave a blank where the viewer expects words -- a title-less card, or a
    side with zero size that cannot be scaled at all -- so it refuses as
    "illegible", naming the field, the way theme.math refuses maths that
    renders nothing."""
    if not text.family_members_with_points() or (text.width == 0 and text.height == 0):
        raise LayoutError(
            f"SplitCompare {what} renders nothing on screen: {source!r}. Its "
            f"characters draw no glyphs in this font (zero-width or "
            f"unsupported). Rewrite it in plain, visible text.",
            kind="illegible",
        )
    return text


def _wrap_balanced(s: str, width: int) -> str:
    """wrap() at `width`, then narrowed as far as it goes without adding a
    line, so the lines come out even and the last one is not a lone word
    ("...from the / head"). Text that fits one line is unchanged."""
    best = wrap(s, width)
    lines = best.count("\n")
    for w in range(width - 1, 0, -1):
        candidate = wrap(s, w)
        if candidate.count("\n") != lines:
            break
        best = candidate
    return best
