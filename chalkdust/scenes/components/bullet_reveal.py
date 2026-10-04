"""BulletReveal: a heading plus sequentially revealed points."""

from __future__ import annotations

import unicodedata
from typing import Annotated, Literal

from manim import DOWN, LEFT, ORIGIN, UP, Dot, FadeIn, Text, VGroup
from pydantic import AfterValidator, Field, StringConstraints

from chalkdust.continuity import artifact_builder
from chalkdust.core.models import Region
from chalkdust.scenes.base import ChalkdustScene
from chalkdust.scenes.components.base import (
    MIN_STEP_SECONDS,
    Component,
    ComponentParams,
    label,
    register,
    wrap,
)
from chalkdust.scenes.regions import LayoutError, fit_to_region
from chalkdust.scenes.theme import Theme, body_cap_height, body_text, heading_text

WRAP_WIDTH = 46

# Vertical rhythm, expressed in cap heights so it scales with the theme.
LINE_HEIGHT = 1.7   # one line of text, anchor to anchor
PARA_GAP = 0.9      # additional space between bullets
DOT_GAP = 0.28      # horizontal space between dot and text (absolute units)

# Blank or invisible text is refused at the schema rung. A blank bullet is a
# dot with nothing beside it, and a blank heading builds an empty mobject at
# the origin that "overlaps" the bullets. Leave the heading out instead of passing "".
def _visible(s: str) -> str:
    # Whitespace is not the only text that draws nothing: zero-width and
    # format characters (U+200B, U+2060) and controls are invisible too, and
    # strip_whitespace keeps them. Require one character outside Z* and C*.
    if not any(unicodedata.category(c)[0] not in "ZC" for c in s):
        raise ValueError("text has no visible character")
    return s


NonBlank = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1),
                     AfterValidator(_visible)]


def _drawn(text: Text, part: str) -> Text:
    """Refuse a text part that builds no glyphs. A visible character the font
    cannot draw (an emoji-only string) passes the schema yet builds an empty
    Text -- an empty card or a bare dot that passes every layout check."""
    if not text.submobjects:
        raise LayoutError(f"{part} draws no glyphs; the font has none for its "
                          f"characters -- rewrite it in plain text",
                          kind="illegible")
    return text


class BulletRevealParams(ComponentParams):
    heading: NonBlank | None = None
    # Capped at 6. A density limit that fires at schema validation -- cheaper
    # than the legibility check, and its error points at the real fix (split
    # the beat) rather than at a font size.
    items: list[NonBlank] = Field(min_length=1, max_length=6)
    reveal: Literal["sequential", "all"] = "sequential"


@register
class BulletReveal(Component):
    name = "BulletReveal"
    Params = BulletRevealParams

    def regions(self) -> set[Region]:
        r = {Region.STAGE}
        if self.params.heading:
            r.add(Region.TITLE_BAR)
        return r

    def _weights(self) -> list[float]:
        """Budget weights, one per play() plus the closing hold: the heading,
        then each bullet (or, with reveal="all", every bullet in one play).
        Shared by build() and min_seconds() so they cannot drift apart."""
        p: BulletRevealParams = self.params
        heading = [1] if p.heading else []
        if p.reveal == "all":
            return heading + [2 * len(p.items)] + [2]
        return heading + [2] * len(p.items) + [2]

    def min_seconds(self) -> float:
        # One step per play() the viewer must register -- the heading, then
        # each bullet (or the one "all" fade); the closing hold is not a step.
        return MIN_STEP_SECONDS * (len(self._weights()) - 1)

    def build(self, scene: ChalkdustScene) -> None:
        p: BulletRevealParams = self.params
        theme = scene.theme
        heading = None
        if p.heading:
            heading = label(_drawn(heading_text(wrap(p.heading, 34), theme),
                                   "heading"), "heading")
            fit_to_region(heading, Region.TITLE_BAR)

        rows = _stacked_rows(p.items, theme)
        bullets = label(VGroup(*rows), "bullets")
        fit_to_region(bullets, Region.STAGE)

        # Add the group up front so the scene holds one top-level mobject the
        # overlap check can reason about; reveal by animating opacity.
        if p.reveal == "sequential":
            for row in rows:
                row.set_opacity(0)
        scene.add(bullets)

        if heading is not None:
            scene.exclusive(heading, bullets)

        times = scene.budget(*self._weights())
        idx = 0

        if heading is not None:
            scene.play(FadeIn(heading), run_time=times[idx])
            idx += 1

        if p.reveal == "all":
            scene.play(FadeIn(bullets), run_time=times[idx])
        else:
            for row in rows:
                scene.play(row.animate.set_opacity(1), run_time=times[idx])
                idx += 1

        scene.settle("bullets revealed")
        scene.wait(times[-1])

    @classmethod
    def examples(cls):
        return [
            {"items": ["One jump becomes a walk"]},
            {"heading": "Three causes",
             "items": ["A weak hash function",
                       "A load factor left too high",
                       "Adversarial keys chosen to collide"]},
            {"items": [f"Point number {i}" for i in range(6)], "reveal": "all"},
        ]

    @classmethod
    def stress(cls):
        return [
            # Max items, each far longer than a real bullet.
            {"heading": "A heading that runs considerably longer than it should",
             "items": ["This bullet carries a great deal more text than any "
                       "single point in a well-constructed beat should ever "
                       "hold, and it continues at length"] * 6},
            # One unwrappable token.
            {"items": ["antidisestablishmentarianism" * 4]},
            {"heading": "Short", "items": ["a"] * 6},  # minimal content
            {"items": ["a"]},  # minimal: one one-character bullet, no heading
            {"heading": "H", "items": ["a"], "reveal": "all"},
            # 3x volume revealed all at once.
            {"heading": "A heading that runs considerably longer than it should",
             "items": ["Each of these bullets says far more than a viewer can "
                       "take in while it fades in"] * 6,
             "reveal": "all"},
            # Visible characters the font has no glyph for: refused as
            # illegible rather than built into a bare dot.
            {"items": ["\U0001F600" * 80]},
        ]


# --- the list itself ----------------------------------------------------------


def _stacked_rows(items: list[str], theme: Theme) -> list[VGroup]:
    """One row per item -- VGroup(dot, text), labelled bullet[i] -- stacked at
    a constant pitch, unplaced and at full strength. Shared by build() and
    the artifact builder, so a carried list is the list the viewer saw --
    and refuses a glyphless item (_drawn) on both paths alike."""
    cap = body_cap_height(theme)
    rows, dots, line_counts = [], [], []
    for i, item in enumerate(items):
        wrapped = wrap(item, WRAP_WIDTH)
        line_counts.append(wrapped.count("\n") + 1)

        text = _drawn(body_text(wrapped, theme), f"bullet {i}")
        dot = Dot(radius=0.07, color=theme.palette.accent)
        dot.next_to(text, LEFT, buff=DOT_GAP)
        # Sit the dot on the optical centre of the FIRST line, measured
        # down from the top by half a cap height. Independent of whether
        # the row happens to contain descenders.
        dot.set_y(text.get_top()[1] - cap / 2)

        row = label(VGroup(dot, text), f"bullet[{i}]")
        # Common left edge, so the dots form a straight column.
        row.align_to(ORIGIN, LEFT)
        rows.append(row)
        dots.append(dot)

    # Stack manually at a constant pitch instead of arrange(), which would
    # space by bounding box and reintroduce the descender problem. The dot
    # is already at the row's anchor, so we align dot y positions.
    cursor = 0.0
    for row, dot, n_lines in zip(rows, dots, line_counts):
        row.shift(UP * (cursor - dot.get_y()))
        cursor -= cap * (LINE_HEIGHT * n_lines + PARA_GAP)
    return rows


@artifact_builder("BulletReveal")
def _artifact(params: BulletRevealParams, theme: Theme) -> VGroup:
    """The settled list, for a later beat's carry_in (SCENE_SPEC.md §6).

    The heading is not part of it: it lives in TITLE_BAR, and a carried
    artifact is placed in STAGE. Parts, which Callout's `part` indexes, are
    the rows in item order -- submobjects[i] is item i's VGroup(dot, text).
    """
    return label(VGroup(*_stacked_rows(params.items, theme)), "bullets")
