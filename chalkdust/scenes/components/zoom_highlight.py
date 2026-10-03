"""ZoomHighlight: focus on part of a visual an earlier beat put on screen.

The target is never built here. It is a carried artifact (SCENE_SPEC.md §6):
the producing beat registers it, this beat lists it in `carry_in`, and the
compiler rebuilds it in STAGE, dimmed, before build() runs. `target_id` is
that carry-in name.

ChalkdustScene is a plain Scene, so there is no camera to zoom. The zoom is
done with mobjects instead: the focused parts return to full strength, then a
magnified copy grows out of them on an opaque lens card that covers their
dimmed neighbours. Parts already too large to magnify meaningfully (the whole
target, a full-width row) get an accent frame instead. The callout sits in
LOWER_THIRD, where it can never cover what it is talking about.
"""

from __future__ import annotations

from typing import Annotated, Any

import numpy as np
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    Create,
    Dot,
    FadeIn,
    Group,
    Mobject,
    Restore,
    RoundedRectangle,
    SurroundingRectangle,
    Transform,
    VGroup,
)
from pydantic import Field, field_validator

from chalkdust.continuity import carried
from chalkdust.core.models import CarryInError, Region
from chalkdust.scenes.base import ChalkdustScene
from chalkdust.scenes.components.base import (
    MIN_STEP_SECONDS,
    Component,
    ComponentParams,
    label,
    register,
    wrap,
)
from chalkdust.scenes.components.bullet_reveal import WRAP_WIDTH as BULLET_WRAP
from chalkdust.scenes.components.bullet_reveal import BulletRevealParams
from chalkdust.scenes.regions import (
    DEFAULT_PADDING,
    Rect,
    bbox,
    fit_to_region,
    region_rect,
    scale_with_tags,
)
from chalkdust.scenes.theme import Theme, body_text

# Relative weights of the beat's steps, in units of MIN_STEP_SECONDS: bring
# the focus up, zoom (or frame) it, show the callout, hold while it is read.
# budget() scales them to the narration, so the ratio is all that matters at
# run time; min_seconds() is where their absolute value counts.
_WEIGHTS = (1, 2, 1, 3)

# The lens may take at most this share of STAGE in either direction -- enough
# to magnify, little enough that the rest of the target stays visible around
# it as context.
LENS_FILL = 0.6
MAX_ZOOM = 2.5
# Below this the "zoom" would be a twitch, not a magnification; frame instead.
MIN_ZOOM = 1.25
LENS_PAD = 0.25       # card edge to magnified content
LENS_CORNER = 0.12
FRAME_BUFF = 0.12     # frame to focus, inside fit_to_region's default padding
STROKE_WIDTH = 4
# The unfocused parts recede further while the focus comes up (on top of
# continuity's DIM_DARKNESS), so a neighbour half under the lens card reads as
# background rather than as clipped text.
RECEDE = 0.7

# One LOWER_THIRD line of body text holds about this many characters; two
# lines still fit legibly, three do not (fit_to_region then raises overflow).
CALLOUT_WRAP = 56

PartIndex = Annotated[int, Field(ge=0)]


class ZoomHighlightParams(ComponentParams):
    # A carry-in name, not a mobject id: the artifact an earlier beat
    # registered and this beat lists in carry_in (SCENE_SPEC.md §6).
    target_id: str = Field(min_length=1)
    # What the viewer should notice about the focused part. Required: a zoom
    # with nothing to say about what it shows is decoration.
    callout: str
    # Optional, added beyond SCENE_SPEC §5's key params because "part of an
    # existing visual" needs a selector. Indices into the carried artifact's
    # top-level parts, whose order is the producing component's documented
    # artifact structure (one bullet row, one array cell, ...). None focuses
    # the whole target.
    parts: list[PartIndex] | None = Field(default=None, min_length=1)

    @field_validator("target_id", "callout")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be blank")
        return v

    @field_validator("parts")
    @classmethod
    def _distinct(cls, v: list[int] | None) -> list[int] | None:
        if v is not None and len(set(v)) != len(v):
            raise ValueError(f"parts must be distinct, got {v}")
        return v


@register
class ZoomHighlight(Component):
    name = "ZoomHighlight"
    Params = ZoomHighlightParams

    def regions(self) -> set[Region]:
        # STAGE for the lens or frame (drawn over the carried target, which
        # CarryIn already placed there); LOWER_THIRD for the callout.
        return {Region.STAGE, Region.LOWER_THIRD}

    def min_seconds(self) -> float:
        # budget() splits the narration by _WEIGHTS, so the beat is legible
        # once every step gets its weight's worth of MIN_STEP_SECONDS.
        return MIN_STEP_SECONDS * sum(_WEIGHTS)

    def latex_strings(self) -> list[str]:
        # The callout is plain Text and the target was compiled by its
        # producer, so this beat compiles no LaTeX of its own.
        return []

    def carried_names(self) -> list[str]:
        # The semantic rung refuses a target the beat does not carry in.
        return [self.params.target_id]

    def build(self, scene: ChalkdustScene) -> None:
        p: ZoomHighlightParams = self.params
        theme = scene.theme

        target = carried(scene, p.target_id)
        # CarryIn saved the undimmed target AFTER placing it, so saved_state
        # is the full-strength twin at the same position, part for part.
        full = target.saved_state
        focus, focus_full = self._select(target), self._select(full)

        callout = label(body_text(wrap(p.callout, CALLOUT_WRAP), theme), "callout")
        fit_to_region(callout, Region.LOWER_THIRD)

        box = bbox(focus)
        zoom = _zoom_factor(box, region_rect(Region.STAGE))
        if zoom >= MIN_ZOOM:
            mag = Group(*(m.copy() for m in focus_full))
            scale_with_tags(mag, zoom)
            card = RoundedRectangle(
                corner_radius=LENS_CORNER,
                width=mag.width + 2 * LENS_PAD,
                height=mag.height + 2 * LENS_PAD,
                fill_color=theme.palette.bg, fill_opacity=1.0,
                stroke_color=theme.palette.accent, stroke_width=STROKE_WIDTH,
            ).move_to(mag)
            marker = label(Group(card, mag), "zoom lens")
            marker.move_to(focus)
            _clamp_into(marker, region_rect(Region.STAGE).inset(DEFAULT_PADDING))
            # Grows out of the focus: starts at the focus's own size and spot,
            # opaque throughout. A FadeIn would cross-fade the magnified copy
            # over the original for the whole step -- double-exposed text.
            # Transform to a copy, not Restore: Restore calls become(), which
            # a plain Group cannot interpolate.
            settled = marker.copy()
            marker.scale(1 / zoom).move_to(focus)
            reveal = Transform(marker, settled)
        else:
            marker = label(
                SurroundingRectangle(focus, buff=FRAME_BUFF, color=theme.palette.accent,
                                     stroke_width=STROKE_WIDTH),
                "focus frame",
            )
            reveal = Create(marker)

        scene.exclusive(marker, callout)

        t_focus, t_zoom, t_callout, t_hold = scene.budget(*_WEIGHTS)
        if p.parts is None:
            scene.play(Restore(target), run_time=t_focus)
        else:
            # One animation per part, never one on a Group of them: Scene.play
            # adds an animated mobject that is not already on screen, and a new
            # Group would be -- drawing those parts a second time.
            others = [m for m in target.submobjects if not any(m is f for f in focus)]
            scene.play(*(Transform(m, f.copy()) for m, f in zip(focus, focus_full)),
                       *(m.animate.fade(RECEDE) for m in others),
                       run_time=t_focus)
        # On screen before the reveal starts, as Scene.play would put it: the
        # lens's Transform is not an introducer, and LayoutProbe only adds
        # what introducers introduce -- without this the probe never sees it.
        scene.add(marker)
        scene.play(reveal, run_time=t_zoom)
        scene.play(FadeIn(callout), run_time=t_callout)
        scene.settle("zoom highlight shown")
        scene.wait(t_hold)

    def _select(self, artifact: Mobject) -> Group:
        """The focused parts of `artifact` (or of its saved twin), in order.

        The part count is only known once the artifact is rebuilt, so an
        out-of-range index surfaces here, as the same typed error as any
        other bad reference into a carried artifact -- a spec bug for the
        spec stage, not a layout failure for the repair loop.
        """
        parts = self.params.parts
        if parts is None:
            return Group(artifact)
        n = len(artifact.submobjects)
        bad = [i for i in parts if i >= n]
        if bad:
            raise CarryInError(
                f"ZoomHighlight focuses part(s) {bad} of {self.params.target_id!r}, "
                f"which has {n} part(s) (indices 0..{n - 1})",
                beat_id=None, name=self.params.target_id,
            )
        return Group(*(artifact.submobjects[i] for i in parts))

    # --- fixtures -------------------------------------------------------------
    # examples() and stress() stay empty: every ZoomHighlight case needs a
    # carried target, so its fixtures carry the target's recipe with them
    # ({"carry_in": [recipe], "params": {...}}, Component.carried_examples)
    # and the registry walks build them with that artifact on screen
    # (validate/fixtures.py). The producer is BulletReveal, which ships no
    # artifact builder yet; fixture_builders() lends it one for the duration
    # of a fixture build -- never over a real builder, never into a render.

    @classmethod
    def fixture_builders(cls) -> dict[str, Any]:
        return {"BulletReveal": _bullet_rows}

    @classmethod
    def carried_examples(cls) -> list[dict[str, Any]]:
        """Realistic cases that MUST validate clean."""
        causes = _recipe({"heading": "Three causes",
                          "items": ["A weak hash function",
                                    "A load factor left too high",
                                    "Adversarial keys chosen to collide"]})
        return [
            # One mid-length row: magnified on a lens.
            {"carry_in": [causes],
             "params": {"target_id": "chain_causes", "parts": [1],
                        "callout": "Past 0.75 full, chains grow faster than the table"}},
            # Short rows: two of them, magnified together at full zoom.
            {"carry_in": [_recipe({"items": ["hash(key)", "mod 8", "bucket 4"]})],
             "params": {"target_id": "chain_causes", "parts": [1, 2],
                        "callout": "The modulo is where two keys collide"}},
            # The whole target: too large to magnify, so it is framed.
            {"carry_in": [causes],
             "params": {"target_id": "chain_causes",
                        "callout": "All three end the same way: one long chain"}},
        ]

    @classmethod
    def carried_stress(cls) -> list[dict[str, Any]]:
        """Hostile cases: each must render correctly or raise LayoutError.

        No invalid-LaTeX case: ZoomHighlight compiles no LaTeX (its callout is
        plain Text). Typed refusals -- a target not carried in or never
        registered, a part out of range, empty input -- are spec errors, not
        LayoutErrors, and are pinned in tests/test_component_zoom_highlight.py.
        """
        # As many rows as BulletReveal allows, each near its wrap width.
        long_rows = [f"Row {i} carries more text than a bullet should" for i in range(6)]
        short = _recipe({"items": ["hash(key)", "mod 8", "bucket 4"]})
        return [
            # (a) callout at 3x realistic volume: refuses with overflow.
            {"carry_in": [short],
             "params": {"target_id": "chain_causes", "parts": [0],
                        "callout": "Past 0.75 full, chains grow faster than the table, so "
                                   "every lookup walks further, every insert walks further, "
                                   "and resizing late costs a full rehash of everything"}},
            # (a) target at 3x volume, every part focused at once.
            {"carry_in": [_recipe({"items": long_rows})],
             "params": {"target_id": "chain_causes", "parts": [0, 1, 2, 3, 4, 5],
                        "callout": "Every row says the same thing"}},
            # (b) an unwrappable 60-character token in the callout.
            {"carry_in": [_recipe({"items": ["hash(key)", "mod 8"]})],
             "params": {"target_id": "chain_causes", "parts": [0],
                        "callout": "x" * 20 + "_identifier_with_no_spaces_at_all" + "y" * 7}},
            # (b) an unwrappable token (a URL) as the focused part itself.
            {"carry_in": [_recipe({"items": ["https://example.com/" + "a" * 40, "mod 8"]})],
             "params": {"target_id": "chain_causes", "parts": [0],
                        "callout": "The key is the whole URL"}},
            # (c) minimal: one tiny part, one-character callout.
            {"carry_in": [_recipe({"items": ["a"]})],
             "params": {"target_id": "chain_causes", "parts": [0], "callout": "a"}},
        ]


def _recipe(bullet_params: dict[str, Any]) -> dict[str, Any]:
    """A fixture's carried target: a BulletReveal registered as chain_causes."""
    return {"name": "chain_causes", "producer": "BulletReveal", "params": bullet_params}


def _bullet_rows(params: BulletRevealParams, theme: Theme) -> Mobject:
    """Fixture-only artifact for BulletReveal: its rows as BulletReveal draws
    them (accent dot, body text wrapped at its width), one top-level part per
    item, the heading left out. Lent by fixture_builders() until BulletReveal
    registers its own @artifact_builder."""
    rows = [VGroup(Dot(radius=0.07, color=theme.palette.accent),
                   body_text(wrap(item, BULLET_WRAP), theme)).arrange(RIGHT, buff=0.28)
            for item in params.items]
    return VGroup(*rows).arrange(DOWN, buff=0.35, aligned_edge=LEFT)


def _zoom_factor(box: Rect, stage: Rect) -> float:
    """How far the focus can be magnified with its lens inside LENS_FILL of
    STAGE. A zero extent (a bare line) does not limit that direction."""
    limits = [MAX_ZOOM]
    for extent, room in ((box.width, stage.width), (box.height, stage.height)):
        if extent > 0:
            limits.append((room * LENS_FILL - 2 * LENS_PAD) / extent)
    return min(limits)


def _clamp_into(mob: Mobject, rect: Rect) -> None:
    """Shift `mob` the least distance that puts it inside `rect`. Callers
    guarantee it is no larger than `rect` (the lens is sized to LENS_FILL)."""
    box = bbox(mob)
    dx = max(rect.left - box.left, 0.0) + min(rect.right - box.right, 0.0)
    dy = max(rect.bottom - box.bottom, 0.0) + min(rect.top - box.top, 0.0)
    mob.shift(np.array([dx, dy, 0.0]))
