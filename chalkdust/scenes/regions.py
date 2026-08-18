"""Layout grid, safe area, and fit helpers.

Manim's frame is 14.222 x 8 units with the origin at the centre. We carve that
into named regions with a safe margin, and components place content by asking
for a region rather than by computing coordinates.

The rule from SCENE_SPEC.md §4: no component ever positions by absolute
coordinate. Everything is relative to a region or to another mobject.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from manim import Mobject, config

from chalkdust.core.models import Region

# --- constants --------------------------------------------------------------

# Margin between the frame edge and anything we draw. Protects against YouTube
# player chrome, mobile safe areas, and the general ugliness of text touching
# the edge.
MARGIN_X = 0.5
MARGIN_Y = 0.4

TITLE_BAR_HEIGHT = 1.2
LOWER_THIRD_HEIGHT = 1.0
GUTTER = 0.5  # horizontal gap between STAGE_LEFT and STAGE_RIGHT

# Manim font_size units. Below this, text is unreadable on a phone at 1080p.
# Hitting this limit means the content is too dense for the component -- the
# fix is splitting the beat, not shrinking the type.
MIN_FONT_SIZE = 22.0

# Default breathing room inside a region when fitting content.
DEFAULT_PADDING = 0.15


class LayoutError(Exception):
    """Raised when content cannot be placed legibly.

    This is a real failure, not a warning. It is caught by the repair loop
    (SCENE_SPEC.md §9), which will try to fix it mechanically, then ask the
    model to revise, then degrade the beat.
    """


# --- geometry ---------------------------------------------------------------


@dataclass(frozen=True)
class Rect:
    """An axis-aligned rectangle in Manim units."""

    x: float  # centre
    y: float  # centre
    width: float
    height: float

    @property
    def left(self) -> float:
        return self.x - self.width / 2

    @property
    def right(self) -> float:
        return self.x + self.width / 2

    @property
    def bottom(self) -> float:
        return self.y - self.height / 2

    @property
    def top(self) -> float:
        return self.y + self.height / 2

    @property
    def center(self) -> np.ndarray:
        """As a Manim point. Manim is 3D internally; z is always 0 for us."""
        return np.array([self.x, self.y, 0.0])

    def inset(self, padding: float) -> Rect:
        """A smaller rect with `padding` removed on every side."""
        return Rect(
            self.x,
            self.y,
            max(self.width - 2 * padding, 0.01),
            max(self.height - 2 * padding, 0.01),
        )

    def intersects(self, other: Rect) -> bool:
        return not (
            self.right <= other.left
            or self.left >= other.right
            or self.top <= other.bottom
            or self.bottom >= other.top
        )

    def contains(self, other: Rect, tol: float = 1e-6) -> bool:
        return (
            other.left >= self.left - tol
            and other.right <= self.right + tol
            and other.bottom >= self.bottom - tol
            and other.top <= self.top + tol
        )


def safe_area() -> Rect:
    """The whole usable frame, inside the margins."""
    return Rect(
        x=0.0,
        y=0.0,
        width=config.frame_width - 2 * MARGIN_X,
        height=config.frame_height - 2 * MARGIN_Y,
    )


def region_rect(region: Region) -> Rect:
    """Resolve a named region to geometry.

    Layout (inside the safe area):

        +-----------------------------+
        |          TITLE_BAR          |
        +--------------+--------------+
        |  STAGE_LEFT  | STAGE_RIGHT  |   <- their union is STAGE
        +--------------+--------------+
        |         LOWER_THIRD         |
        +-----------------------------+
    """
    safe = safe_area()

    if region is Region.TITLE_BAR:
        return Rect(
            x=safe.x,
            y=safe.top - TITLE_BAR_HEIGHT / 2,
            width=safe.width,
            height=TITLE_BAR_HEIGHT,
        )

    if region is Region.LOWER_THIRD:
        return Rect(
            x=safe.x,
            y=safe.bottom + LOWER_THIRD_HEIGHT / 2,
            width=safe.width,
            height=LOWER_THIRD_HEIGHT,
        )

    # STAGE is whatever is left between the bar and the lower third.
    stage_height = safe.height - TITLE_BAR_HEIGHT - LOWER_THIRD_HEIGHT
    stage_y = safe.bottom + LOWER_THIRD_HEIGHT + stage_height / 2

    if region is Region.STAGE:
        return Rect(safe.x, stage_y, safe.width, stage_height)

    half_width = (safe.width - GUTTER) / 2
    if region is Region.STAGE_LEFT:
        return Rect(safe.left + half_width / 2, stage_y, half_width, stage_height)
    if region is Region.STAGE_RIGHT:
        return Rect(safe.right - half_width / 2, stage_y, half_width, stage_height)

    raise ValueError(f"unknown region: {region}")


def bbox(mob: Mobject) -> Rect:
    """Bounding box of a mobject as a Rect.

    We use get_left/right/top/bottom rather than any internal bounding-box API
    because these are stable across Manim versions.
    """
    if not mob.submobjects and mob.width == 0 and mob.height == 0:
        # Empty group -- degenerate box at its own centre.
        c = mob.get_center()
        return Rect(float(c[0]), float(c[1]), 0.0, 0.0)

    left = float(mob.get_left()[0])
    right = float(mob.get_right()[0])
    bottom = float(mob.get_bottom()[1])
    top = float(mob.get_top()[1])
    return Rect(
        x=(left + right) / 2,
        y=(bottom + top) / 2,
        width=right - left,
        height=top - bottom,
    )


# --- font-size tracking -----------------------------------------------------
#
# Manim does not record what font size a Text was created at once you scale it,
# so we track it ourselves. Text created through theme.py carries a
# `_chalk_font_size` attribute, and every scale we apply multiplies it. That
# gives us an honest effective font size to check legibility against.


def tag_font_size(mob: Mobject, font_size: float) -> Mobject:
    """Record the font size a text mobject was created at."""
    mob._chalk_font_size = float(font_size)  # type: ignore[attr-defined]
    return mob


def _apply_scale_to_tags(mob: Mobject, factor: float) -> None:
    """Propagate a scale factor to every tagged descendant."""
    for m in mob.get_family():
        existing = getattr(m, "_chalk_font_size", None)
        if existing is not None:
            m._chalk_font_size = existing * factor  # type: ignore[attr-defined]


def smallest_font_size(mob: Mobject) -> float | None:
    """Effective font size of the smallest tagged text, or None if untagged."""
    sizes = [
        getattr(m, "_chalk_font_size")
        for m in mob.get_family()
        if getattr(m, "_chalk_font_size", None) is not None
    ]
    return min(sizes) if sizes else None


# --- the main helper --------------------------------------------------------


def fit_to_region(
    mob: Mobject,
    region: Region | Rect,
    padding: float = DEFAULT_PADDING,
    min_font_size: float = MIN_FONT_SIZE,
    align: np.ndarray | None = None,
) -> float:
    """Scale `mob` down to fit inside `region` and move it there.

    Never scales up -- if content is smaller than its region, it stays at its
    natural size. Growing text to fill space produces wildly inconsistent
    typography across beats.

    Returns the scale factor applied. Raises LayoutError if fitting would push
    text below the legibility floor.

    `align` optionally aligns to an edge of the region instead of centring,
    e.g. align=LEFT pins content to the region's left edge.
    """
    rect = region_rect(region) if isinstance(region, Region) else region
    inner = rect.inset(padding)

    box = bbox(mob)
    if box.width <= 0 or box.height <= 0:
        # Nothing to place. Not an error -- some components legitimately build
        # empty groups on early beats.
        mob.move_to(inner.center)
        return 1.0

    scale = min(1.0, inner.width / box.width, inner.height / box.height)

    if scale < 1.0:
        mob.scale(scale)
        _apply_scale_to_tags(mob, scale)

    # Legibility check, after scaling, using tracked font sizes.
    effective = smallest_font_size(mob)
    if effective is not None and effective < min_font_size:
        raise LayoutError(
            f"Content does not fit legibly: smallest text would render at "
            f"font_size {effective:.1f} (floor is {min_font_size:.0f}) after "
            f"scaling by {scale:.2f}. Split this beat or reduce its content."
        )

    mob.move_to(inner.center)
    if align is not None:
        # Pin to an edge: shift so the relevant side touches the inner rect.
        mob.align_to(_edge_point(inner, align), align)
    return scale


def _edge_point(rect: Rect, direction: np.ndarray) -> np.ndarray:
    """A point on the rect's boundary in the given direction."""
    return np.array(
        [
            rect.x + direction[0] * rect.width / 2,
            rect.y + direction[1] * rect.height / 2,
            0.0,
        ]
    )


# --- assertions -------------------------------------------------------------
# Called at settle points by ChalkdustScene. These are validation rung 3 from
# SCENE_SPEC.md §8 -- cheap, deterministic, catches most real breakage.


def assert_in_safe_area(mob: Mobject, label: str = "mobject") -> None:
    box = bbox(mob)
    safe = safe_area()
    if not safe.contains(box):
        raise LayoutError(
            f"{label} extends outside the safe area: "
            f"x[{box.left:.2f},{box.right:.2f}] y[{box.bottom:.2f},{box.top:.2f}] "
            f"vs safe x[{safe.left:.2f},{safe.right:.2f}] "
            f"y[{safe.bottom:.2f},{safe.top:.2f}]"
        )


def assert_no_overlap(a: Mobject, b: Mobject, a_label: str, b_label: str) -> None:
    if bbox(a).intersects(bbox(b)):
        raise LayoutError(f"{a_label} overlaps {b_label}")


def assert_legible(mob: Mobject, label: str = "mobject",
                   min_font_size: float = MIN_FONT_SIZE) -> None:
    effective = smallest_font_size(mob)
    if effective is not None and effective < min_font_size:
        raise LayoutError(
            f"{label} contains text at font_size {effective:.1f}, "
            f"below the {min_font_size:.0f} floor"
        )