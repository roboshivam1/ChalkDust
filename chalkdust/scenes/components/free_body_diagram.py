"""FreeBodyDiagram: a body and the forces acting on it (JEE problem setup).

The body is a box with its name inside. Each force is an arrow drawn outward
from the box's edge in the force's direction, with its LaTeX label beyond the
tip. Forces appear one at a time, in spec order, so narration can name them as
they arrive.

Three layout decisions carry the component:

  * Arrow length is proportional to magnitude, clamped. The largest force gets
    ARROW_MAX; anything that would come out shorter than ARROW_MIN_FRACTION of
    that is held at the floor, so a 1 N friction next to a 1000 N weight is
    still a visible arrow. Below the floor lengths stop being proportional --
    legibility wins, and the label carries the quantity.
  * Forces that point (nearly) the same way are drawn side by side rather than
    on top of each other: a cluster of close directions is spread across the
    body's face, perpendicular to the cluster's mean direction. The box grows
    if a large cluster needs a wider face.
  * Labels never overlap. Each label starts just past its arrow tip; if it
    touches the body, another arrow, or an earlier label, it is pushed further
    out along its own arrow until it is clear. Moving outward always ends up
    clear of a bounded drawing, so the only failure is the diagram growing too
    large to fit legibly -- which fit_to_region refuses as overflow.

Directions are physics, not screen positions: `angle` is the force's direction
in degrees counter-clockwise from rightward. Where the arrow lands on screen is
still the component's business (SCENE_SPEC.md §3).
"""

from __future__ import annotations

import math as pymath
from typing import Annotated

import numpy as np
from manim import FadeIn, GrowArrow, MathTex, Mobject, Rectangle, VGroup
from manim import Arrow as ManimArrow
from pydantic import Field, StringConstraints

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
from chalkdust.scenes.regions import LayoutError, Rect, bbox, fit_to_region
from chalkdust.scenes.theme import Theme, body_cap_height, body_text, math

# Blank strings are rejected at the schema rung: a blank label compiles to
# nothing, and a body with no name is a box nobody can refer to.
NonBlank = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

NAME_WRAP = 14      # characters per line of the body's name; keeps the box squarish

# Geometry, in body cap heights so it scales with the theme (see bullet_reveal).
BODY_MIN_SIDE = 4.0       # the box is never smaller than this, however short the name
BODY_PAD = 1.0            # space between the name and the box edge
ARROW_MAX = 5.0           # length of the largest force's arrow
ARROW_MIN_FRACTION = 0.35  # clamp floor, as a fraction of ARROW_MAX
TIP_LENGTH = 0.8          # every arrowhead is this long, whatever the arrow's length
LABEL_BUFF = 0.5          # gap between an arrow tip and its label
CLEARANCE = 0.25          # how close a label may come to anything else
PARALLEL_GAP = 2.2        # spacing between side-by-side arrows in one cluster
NUDGE_STEP = 0.5          # how far a colliding label moves out per attempt
MAX_NUDGES = 60           # ~30 cap heights; past that the diagram cannot fit anyway

# Forces whose directions are within this many degrees of a neighbour form one
# cluster and are drawn side by side. Further apart than this, two arrows from
# one base diverge quickly enough to read as separate forces.
CLUSTER_DEG = 20.0
# Clustered arrow bases stay within this fraction of the box's half-extent on
# each axis, so even the outermost base sits on the face, not on a corner.
FACE_USE = 0.8
# Manim thins and shortens the head of a short arrow by default, which would
# read as a second magnitude encoding. Ratios this loose switch that off, so
# length is the only thing that varies between arrows.
NO_THINNING = 1000.0
TIP_RATIO_CAP = 0.5       # an arrowhead never takes more than half its arrow

# Minimum legible duration of each kind of segment, in seconds. These double as
# the budget weights: build() scales all of them by the same factor, so the
# beat stays legible exactly when the narration is at least min_seconds() long
# (SCENE_SPEC.md §8 rung 2 compares the two).
BODY_S = 0.75       # fading in the body
FORCE_S = 0.75      # growing one force's arrow and fading in its label
HOLD_S = 1.0        # holding the finished diagram still long enough to read it


class Force(ComponentParams):
    # LaTeX, maths mode: "mg", "N", "f_k", "T_1", r"\vec{F}".
    label: NonBlank
    # Degrees counter-clockwise from rightward: 0 right, 90 up, 180 left,
    # 270 (or -90) down.
    angle: float = Field(ge=-360, le=360)
    # Relative size, in any one consistent unit. Only ratios matter. Defaults
    # to 1 so a qualitative diagram can omit it and get equal arrows.
    magnitude: float = Field(default=1.0, gt=0, allow_inf_nan=False)


class FreeBodyDiagramParams(ComponentParams):
    # Plain-text name shown inside the box: "block", "2 kg block", "m".
    body: NonBlank
    # Capped at 12 -- three times a realistic four-force diagram -- to bound
    # LaTeX compile cost on runaway input. A crowded diagram that cannot stay
    # legible refuses at layout time with a LayoutError, well before 12.
    forces: list[Force] = Field(min_length=1, max_length=12)


@register
class FreeBodyDiagram(Component):
    name = "FreeBodyDiagram"
    Params = FreeBodyDiagramParams

    def regions(self) -> set[Region]:
        # Arrows can point any way, so the diagram claims the whole stage.
        return {Region.STAGE}

    # --- hooks for the semantic rung (SCENE_SPEC.md §8 rung 2) ---------------

    def latex_strings(self) -> list[str]:
        """Every LaTeX string build() compiles: one label per force."""
        return [f.label for f in self.params.forces]

    def min_seconds(self) -> float:
        """Shortest narration at which every segment still gets its minimum."""
        return sum(self._segments())

    def _segments(self) -> list[float]:
        """Minimum durations of every timed segment, in the order build() plays
        them. Shared by build() and min_seconds() so they cannot drift apart."""
        return [BODY_S] + [FORCE_S] * len(self.params.forces) + [HOLD_S]

    # --- build ---------------------------------------------------------------

    def build(self, scene: ChalkdustScene) -> None:
        body, arrows, labels = _diagram(self.params, scene.theme)

        # One fit for everything, so arrows, labels and body scale together and
        # keep their relationships. Raises LayoutError (overflow) when the
        # diagram cannot fit legibly: the fix is splitting the beat.
        diagram = label(VGroup(body, *arrows, *labels), "free body diagram")
        fit_to_region(diagram, Region.STAGE)

        # Every part is a top-level scene mobject once revealed. The body and
        # labels are exclusive so settle() re-checks what _place_label promised;
        # arrows are not, since a diagonal arrow's bounding box legitimately
        # spans space its line never touches.
        scene.exclusive(body, *labels)

        # budget() hands out whole frames summing exactly to the beat (D-002).
        times = iter(scene.budget(*self._segments()))
        scene.play(FadeIn(body), run_time=next(times))
        scene.settle("body")
        for i, (arrow, tex) in enumerate(zip(arrows, labels)):
            scene.play(GrowArrow(arrow), FadeIn(tex), run_time=next(times))
            scene.settle(f"force {i}")
        scene.wait(next(times))

    # --- fixtures ------------------------------------------------------------

    @classmethod
    def examples(cls):
        return [
            # Block dragged along a rough floor by a horizontal rope.
            {"body": "2 kg block",
             "forces": [{"label": "mg", "angle": 270, "magnitude": 19.6},
                        {"label": "N", "angle": 90, "magnitude": 19.6},
                        {"label": "T", "angle": 0, "magnitude": 10},
                        {"label": "f_k", "angle": 180, "magnitude": 4}]},
            # Block at rest on a 30 degree incline: normal perpendicular to the
            # slope, static friction up it.
            {"body": "m",
             "forces": [{"label": "mg", "angle": -90, "magnitude": 10},
                        {"label": "N", "angle": 120, "magnitude": 8.66},
                        {"label": "f_s", "angle": 30, "magnitude": 5}]},
            # Lamp hung from two strings, plus a qualitative push (no
            # magnitude) along the weight's line.
            {"body": "lamp",
             "forces": [{"label": "T_1", "angle": 135, "magnitude": 7},
                        {"label": "T_2", "angle": 45, "magnitude": 7},
                        {"label": "mg", "angle": 270, "magnitude": 9.8},
                        {"label": r"\vec{F}", "angle": 270}]},
        ]

    @classmethod
    def stress(cls):
        return [
            # 3x realistic volume: the schema maximum of forces with long
            # labels, several crowded into near-identical directions, and a
            # body name far longer than a box should carry.
            {"body": "a uniform rigid rod resting against a frictionless wall "
                     "with its lower end on a rough floor",
             "forces": [{"label": rf"F_{{\mathrm{{applied}},{i}}} + \Delta F_{i}",
                         "angle": a, "magnitude": m}
                        for i, (a, m) in enumerate([
                            (0, 5), (10, 3), (20, 8), (90, 12), (95, 1),
                            (180, 4), (185, 4), (200, 9), (270, 20), (275, 2),
                            (280, 6), (315, 7)])]},
            # Unwrappable tokens: a 60-character identifier as the body and a
            # space-free URL and identifier as labels.
            {"body": "identifier" * 6,
             "forces": [{"label": r"\mathrm{" + "identifier" * 6 + "}", "angle": 0},
                        {"label": "https://example.com/" + "a" * 40, "angle": 270}]},
            # Minimal: one-character body, one force.
            {"body": "m", "forces": [{"label": "F", "angle": 0}]},
            # Every force the same way at the same size: the whole cluster
            # spreads across one face and the labels must still separate.
            {"body": "crate",
             "forces": [{"label": f"F_{i}", "angle": 270} for i in range(6)]},
            # Magnitudes twelve orders apart: the tiny one is held at the floor.
            {"body": "dust grain",
             "forces": [{"label": "F_g", "angle": 270, "magnitude": 1e-6},
                        {"label": "F_E", "angle": 90, "magnitude": 1e6}]},
            # Directions straddling 0/360, which must still cluster together.
            {"body": "puck",
             "forces": [{"label": "F_a", "angle": 355},
                        {"label": "F_b", "angle": 5},
                        {"label": "F_c", "angle": -360}]},
            # Invalid LaTeX in the second label -- an unclosed brace, an
            # undefined command, and LaTeX that compiles but draws nothing.
            # Each refuses as "invalid_latex" naming forces[1].label.
            *({"body": "m",
               "forces": [{"label": "N", "angle": 90},
                          {"label": bad, "angle": 270}]}
              for bad in (r"\frac{m", r"\notacommand{g}", r"\,")),
        ]


# --- the settled visual -----------------------------------------------------------


def _diagram(p: FreeBodyDiagramParams, theme: Theme
             ) -> tuple[Mobject, list[ManimArrow], list[MathTex]]:
    """The finished diagram's parts -- body, one arrow per force, one label per
    force -- laid out relative to each other but not yet fitted or placed.

    A pure function of params and theme, shared by build() and the carry-in
    artifact builder, so a later beat's carried copy is the very picture this
    beat settled on (SCENE_SPEC.md §6).
    """
    cap = body_cap_height(theme)

    dirs = [_unit(f.angle) for f in p.forces]
    bases = _cluster_bases([f.angle for f in p.forces], cap * PARALLEL_GAP)

    # The box: big enough for its name, and large enough on each axis that
    # every clustered arrow's base sits on the face.
    name = body_text(wrap(p.body, NAME_WRAP), theme)
    reach_x = 2 * max(abs(b[0]) for b in bases) / FACE_USE
    reach_y = 2 * max(abs(b[1]) for b in bases) / FACE_USE
    box = Rectangle(
        width=max(cap * BODY_MIN_SIDE, reach_x, name.width + 2 * cap * BODY_PAD),
        height=max(cap * BODY_MIN_SIDE, reach_y, name.height + 2 * cap * BODY_PAD),
        color=theme.palette.fg,
        fill_color=theme.palette.muted,
        fill_opacity=0.2,
    )
    name.move_to(box)
    body = label(VGroup(box, name), "body")

    # Arrows: base on the box edge, length from magnitude with the clamp.
    centre = box.get_center()
    half = np.array([box.width / 2, box.height / 2, 0.0])
    biggest = max(f.magnitude for f in p.forces)
    arrows, segments = [], []
    for i, (f, d, base) in enumerate(zip(p.forces, dirs, bases)):
        length = cap * ARROW_MAX * max(ARROW_MIN_FRACTION, f.magnitude / biggest)
        start = centre + base + d * _exit_distance(base, d, half)
        end = start + d * length
        arrows.append(label(
            ManimArrow(start, end, buff=0, color=theme.palette.accent,
                       tip_length=cap * TIP_LENGTH,
                       max_tip_length_to_length_ratio=TIP_RATIO_CAP,
                       max_stroke_width_to_length_ratio=NO_THINNING),
            f"force[{i}]"))
        segments.append((start, end))

    # Labels, placed after every arrow exists so no later arrow can cross an
    # already-placed label. theme.math refuses bad LaTeX (or LaTeX that draws
    # nothing) as LayoutError kind "invalid_latex", naming the force.
    obstacles = [bbox(body)]
    labels = []
    for i, (f, d) in enumerate(zip(p.forces, dirs)):
        tex = label(math(f.label, theme, size=theme.type.body,
                         what=f"forces[{i}].label"), f"label[{i}]")
        others = [s for j, s in enumerate(segments) if j != i]
        _place_label(tex, segments[i][1], d, cap, obstacles, others)
        obstacles.append(bbox(tex))
        labels.append(tex)
    return body, arrows, labels


@artifact_builder("FreeBodyDiagram")
def _artifact(params: FreeBodyDiagramParams, theme: Theme) -> Mobject:
    """The settled diagram for a later beat's carry_in (SCENE_SPEC.md §6):
    body, every arrow and every label, unplaced -- CarryIn fits and dims it."""
    body, arrows, labels = _diagram(params, theme)
    return VGroup(body, *arrows, *labels)


# --- geometry helpers -----------------------------------------------------------
# Private to this component. _segment_hits_rect is general enough to share if a
# second component needs label-vs-line clearance (noted for the integrator).


def _unit(angle_deg: float) -> np.ndarray:
    a = pymath.radians(angle_deg)
    return np.array([pymath.cos(a), pymath.sin(a), 0.0])


def _perp(d: np.ndarray) -> np.ndarray:
    """`d` rotated a quarter turn counter-clockwise."""
    return np.array([-d[1], d[0], 0.0])


def _cluster_bases(angles: list[float], gap: float) -> list[np.ndarray]:
    """Where each force's arrow starts from, relative to the body centre, before
    being carried out to the box edge along the force's own direction.

    Forces whose directions chain together within CLUSTER_DEG (wrapping at
    360) form a cluster. Its members are spread `gap` apart along the
    perpendicular to the cluster's mean direction, ordered by angle, so arrows
    that diverge are offset the way they diverge and never cross. A lone force
    starts from the centre.
    """
    n = len(angles)
    norm = [a % 360.0 for a in angles]
    order = sorted(range(n), key=lambda i: (norm[i], i))

    clusters: list[list[int]] = [[order[0]]]
    for prev, cur in zip(order, order[1:]):
        if norm[cur] - norm[prev] <= CLUSTER_DEG:
            clusters[-1].append(cur)
        else:
            clusters.append([cur])
    # Join across the 0/360 seam: 355 and 5 degrees are neighbours. The seam
    # cluster is listed first so the members stay in counter-clockwise order.
    if len(clusters) > 1 and norm[order[0]] + 360.0 - norm[order[-1]] <= CLUSTER_DEG:
        clusters[0] = clusters.pop() + clusters[0]

    bases = [np.zeros(3) for _ in range(n)]
    for members in clusters:
        mean = sum(_unit(angles[i]) for i in members)
        # A long chain of clusters can wrap far enough round for its directions
        # to cancel; then the first member's direction stands in for the mean.
        norm_mean = float(np.linalg.norm(mean))
        axis = _perp(mean / norm_mean if norm_mean > 1e-6 else _unit(angles[members[0]]))
        k = len(members)
        for j, i in enumerate(members):
            bases[i] = axis * (j - (k - 1) / 2) * gap
    return bases


def _exit_distance(inside: np.ndarray, d: np.ndarray, half: np.ndarray) -> float:
    """How far a ray from `inside` (relative to the box centre) travels along
    `d` before leaving a box with half-extents `half`."""
    ts = [(np.sign(d[k]) * half[k] - inside[k]) / d[k]
          for k in (0, 1) if abs(d[k]) > 1e-9]
    return min(ts)


def _place_label(tex: MathTex, tip: np.ndarray, d: np.ndarray, cap: float,
                 rects: list[Rect], segments: list[tuple[np.ndarray, np.ndarray]]) -> None:
    """Put `tex` just past `tip` along `d`, then push it outward until it clears
    every rect and every segment by CLEARANCE.

    The label's centre sits so that its bounding box's nearest edge (in
    direction d) is LABEL_BUFF beyond the tip, whatever the label's aspect.
    """
    reach = tex.width / 2 * abs(d[0]) + tex.height / 2 * abs(d[1])
    tex.move_to(tip + d * (cap * LABEL_BUFF + reach))
    pad = cap * CLEARANCE
    for _ in range(MAX_NUDGES):
        box = bbox(tex)
        padded = Rect(box.x, box.y, box.width + 2 * pad, box.height + 2 * pad)
        if not any(padded.intersects(r) for r in rects) and not any(
            _segment_hits_rect(a, b, padded) for a, b in segments
        ):
            return
        tex.shift(d * cap * NUDGE_STEP)
    raise LayoutError(
        f"{getattr(tex, '_chalk_label', 'label')} cannot be placed clear of the "
        f"other forces. Too many forces crowd one direction; split this beat.",
        kind="overflow",
    )


def _segment_hits_rect(a: np.ndarray, b: np.ndarray, r: Rect) -> bool:
    """Whether segment a-b passes through rect r (Liang-Barsky clipping)."""
    t0, t1 = 0.0, 1.0
    dx, dy = b[0] - a[0], b[1] - a[1]
    for p, q in ((-dx, a[0] - r.left), (dx, r.right - a[0]),
                 (-dy, a[1] - r.bottom), (dy, r.top - a[1])):
        if abs(p) < 1e-12:
            if q < 0:
                return False
            continue
        t = q / p
        if p < 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
        if t0 > t1:
            return False
    return True
