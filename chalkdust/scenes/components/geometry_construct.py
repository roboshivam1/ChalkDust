"""GeometryConstruct: ruler-and-compass builds, revealed step by step.

The spec describes a figure in its own local frame -- named points with
coordinates in whatever units suit the problem, y pointing up -- and every
other element by reference to those points ("circle centred at A through B").
The component maps that frame uniformly into STAGE. So the model chooses the
*shape* of the figure, never where it sits on screen or how big it is
(SCENE_SPEC.md §3, §4): the local frame is unitless, and the same construction
written at 1e-3 or 1e3 lands identically.

All geometry is resolved in plain Python at param validation (rung 1). A
construction that references an undefined point, or asks where two circles
meet when they never do, is a malformed spec -- it fails before any Manim
object exists, with a message naming the offending step. What can only be
judged on screen (labels colliding, a circle too small to see) is a LayoutError
at build time, like every other component (SCENE_SPEC.md §11 rule 1).
"""

from __future__ import annotations

import re
from typing import Annotated, Literal, Union

import numpy as np
from manim import (
    DL,
    DOWN,
    DR,
    LEFT,
    RIGHT,
    UL,
    UP,
    UR,
    Circle,
    Create,
    Dot,
    DrawBorderThenFill,
    FadeIn,
    FadeOut,
    Line,
    Mobject,
    Polygon,
    VGroup,
)
from pydantic import Field, field_validator, model_validator

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
    LayoutError,
    bbox,
    fit_to_region,
    region_rect,
)
from chalkdust.scenes.theme import Theme, body_text

# A construction longer than this is two beats. Fires at schema validation,
# where the error points at the real fix (split the beat) -- same reasoning as
# BulletReveal's item cap.
MAX_ELEMENTS = 16

NOTE_WRAP = 56        # one LOWER_THIRD line of body text; two still fit legibly
DOT_RADIUS = 0.06
STROKE = 3.0
COMPASS_STROKE = 2.0  # construction circles recede behind the figure
POLYGON_FILL = 0.18
LABEL_BUFF = 0.1
# Smallest circle radius / segment length, in scene units, that reads as the
# shape it is rather than a speck.
MIN_FEATURE = 0.15

# Relative tolerance for coincidence, tangency and "lies on the segment",
# scaled by the figure's extent so the local frame stays unitless.
TOL = 1e-7

# Minimum legible seconds per step (the semantic rung compares narration
# against their sum). They double as budget weights, so a beat that meets
# min_seconds() gives every step at least its minimum.
STEP_MIN = {"point": 0.5, "segment": 0.8, "circle": 1.2, "polygon": 1.0,
            "intersect": 0.6}
INTRO_MIN = 1.0   # the given figure, revealed together
HOLD_MIN = 1.0    # the finished construction, held
READ_WPS = 3.0    # caption reading speed for a step's note

# Label candidate directions, scanned in this order (ties keep the first).
_DIRECTIONS = (UR, UL, DR, DL, UP, RIGHT, DOWN, LEFT)

# One or two letters, optionally primed or numbered: A, B', O1, P12. A name is
# also the label drawn on screen, so it is short by construction.
PointName = Annotated[str, Field(pattern=r"^[A-Za-z][A-Za-z0-9']{0,3}$")]
Coord = Annotated[float, Field(allow_inf_nan=False)]
# A TeX control word (\frac, \sqrt) or an inline $...$ span: maths source that
# a plain-text caption would draw literally.
_TEX_MARKUP = re.compile(r"\\[A-Za-z]+|\$[^$]+\$")


# --- params -----------------------------------------------------------------


class _Element(ComponentParams):
    # A caption for this step, shown in LOWER_THIRD while it is drawn. Only
    # construction steps may carry one; the given figure appears all at once.
    note: str | None = None

    @field_validator("note")
    @classmethod
    def _blank_is_none(cls, v: str | None) -> str | None:
        # A blank caption is no caption. Normalised here, once, so regions(),
        # the timing weights and build() cannot disagree about whether a step
        # has one -- wrap() would strip it to an empty Text sitting on the
        # figure.
        if v is None or not v.strip():
            return None
        if _TEX_MARKUP.search(v):
            # Captions are plain Text: TeX here would reach the screen as its
            # source, '$\frac{AB}{2}$' and all. Nothing in this component
            # compiles maths (latex_strings() is empty), so refuse at rung 1
            # rather than render it broken.
            raise ValueError(
                f"note {v!r} contains LaTeX markup, but notes are plain text "
                "and GeometryConstruct compiles no maths; say it in words "
                "('half of AB') or show the formula in an EquationDerivation beat"
            )
        return v.strip()


class PointElement(_Element):
    kind: Literal["point"]
    name: PointName
    at: tuple[Coord, Coord]  # local frame, y up


class SegmentElement(_Element):
    kind: Literal["segment"]
    ends: tuple[PointName, PointName]
    id: str | None = None  # needed only if an intersect refers to it


class CircleElement(_Element):
    """A compass stroke: centred on one point, passing through another."""

    kind: Literal["circle"]
    center: PointName
    through: PointName
    id: str | None = None


class PolygonElement(_Element):
    kind: Literal["polygon"]
    vertices: list[PointName] = Field(min_length=3, max_length=8)


class IntersectElement(_Element):
    """Name where two drawn segments/circles cross.

    Points that coincide with an already-named point are skipped -- that is
    what a geometer means by "the circles meet again at C". The remaining
    points are ordered top-first, then left-first, in the local frame; `names`
    takes them in that order, so one name picks the upper (or left) point.
    """

    kind: Literal["intersect"]
    of: tuple[str, str]
    names: list[PointName] = Field(min_length=1, max_length=2)


Element = Annotated[
    Union[PointElement, SegmentElement, CircleElement, PolygonElement,
          IntersectElement],
    Field(discriminator="kind"),
]


class GeometryConstructParams(ComponentParams):
    # The given figure, revealed together at the start.
    shapes: list[Element] = Field(default_factory=list, max_length=MAX_ELEMENTS)
    # The build, one element per step.
    construction: list[Element] = Field(default_factory=list,
                                        max_length=MAX_ELEMENTS)

    @model_validator(mode="after")
    def _resolves(self) -> GeometryConstructParams:
        if not self.shapes and not self.construction:
            raise ValueError("a construction needs at least one shape or step")
        for i, el in enumerate(self.shapes):
            if el.note:
                raise ValueError(
                    f"shapes[{i}] has a note; notes belong to construction "
                    "steps -- the given figure is revealed all at once"
                )
        _resolve(self)  # raises ValueError naming the step that does not resolve
        return self


# --- geometry, in the local frame --------------------------------------------


def _cross(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _tolerance(points: dict[str, np.ndarray]) -> float:
    """Absolute tolerance for the figure as defined so far."""
    if not points:
        return 0.0
    arr = np.array(list(points.values()))
    extent = float(np.ptp(arr, axis=0).max()) or float(np.abs(arr).max())
    return TOL * extent


def _seg_seg(p1, p2, q1, q2, tol) -> list[np.ndarray]:
    r, s = p2 - p1, q2 - q1
    denom = _cross(r, s)
    if abs(denom) <= TOL * np.linalg.norm(r) * np.linalg.norm(s):
        return []  # parallel; collinear overlap has no single crossing
    t = _cross(q1 - p1, s) / denom
    u = _cross(q1 - p1, r) / denom
    slack_t = tol / np.linalg.norm(r)
    slack_u = tol / np.linalg.norm(s)
    if -slack_t <= t <= 1 + slack_t and -slack_u <= u <= 1 + slack_u:
        return [p1 + t * r]
    return []


def _seg_circle(p1, p2, c, radius, tol) -> list[np.ndarray]:
    length = float(np.linalg.norm(p2 - p1))
    d = (p2 - p1) / length
    foot = p1 + float(np.dot(c - p1, d)) * d
    dist = float(np.linalg.norm(c - foot))
    if dist > radius + tol:
        return []
    h = float(np.sqrt(max(radius * radius - dist * dist, 0.0)))
    found = [foot] if h <= tol else [foot - h * d, foot + h * d]
    return [x for x in found if -tol <= float(np.dot(x - p1, d)) <= length + tol]


def _circle_circle(c1, r1, c2, r2, tol) -> list[np.ndarray]:
    dist = float(np.linalg.norm(c2 - c1))
    if dist <= tol or dist > r1 + r2 + tol or dist < abs(r1 - r2) - tol:
        return []
    along = (c2 - c1) / dist
    a = (r1 * r1 - r2 * r2 + dist * dist) / (2 * dist)
    h = float(np.sqrt(max(r1 * r1 - a * a, 0.0)))
    base = c1 + a * along
    if h <= tol:
        return [base]
    perp = np.array([-along[1], along[0]])
    return [base + h * perp, base - h * perp]


def _resolve(p: GeometryConstructParams) -> dict[str, np.ndarray]:
    """Every named point's local position, in definition order.

    Raises ValueError, prefixed with the step it came from, for anything that
    does not resolve to a drawable figure.
    """
    points: dict[str, np.ndarray] = {}
    # id -> ("segment", a, b) | ("circle", centre, through), by point name
    curves: dict[str, tuple[str, str, str]] = {}

    steps = [("shapes", i, el) for i, el in enumerate(p.shapes)]
    steps += [("construction", i, el) for i, el in enumerate(p.construction)]
    for where, i, el in steps:
        try:
            _apply(el, points, curves)
        except ValueError as exc:
            raise ValueError(f"{where}[{i}] ({el.kind}): {exc}") from None
    return points


def _apply(el: _Element, points: dict[str, np.ndarray],
           curves: dict[str, tuple[str, str, str]]) -> None:
    def need(name: str) -> np.ndarray:
        if name not in points:
            raise ValueError(
                f"point {name!r} is not defined yet; defined so far: "
                f"{sorted(points) or 'none'}"
            )
        return points[name]

    def define(name: str, at: np.ndarray) -> None:
        if name in points:
            raise ValueError(f"point {name!r} is already defined")
        points[name] = at

    def register_curve(curve_id: str | None, curve: tuple[str, str, str]) -> None:
        if curve_id is None:
            return
        if curve_id in curves:
            raise ValueError(f"id {curve_id!r} is already used")
        curves[curve_id] = curve

    tol = _tolerance(points)

    if isinstance(el, PointElement):
        define(el.name, np.array(el.at, dtype=float))

    elif isinstance(el, SegmentElement):
        a, b = el.ends
        if np.linalg.norm(need(a) - need(b)) <= tol:
            raise ValueError(f"segment {a}{b} has zero length")
        register_curve(el.id, ("segment", a, b))

    elif isinstance(el, CircleElement):
        if np.linalg.norm(need(el.center) - need(el.through)) <= tol:
            raise ValueError(f"circle centred at {el.center} through "
                             f"{el.through} has zero radius")
        register_curve(el.id, ("circle", el.center, el.through))

    elif isinstance(el, PolygonElement):
        verts = [need(v) for v in el.vertices]
        if len(set(el.vertices)) != len(el.vertices):
            raise ValueError("polygon repeats a vertex")
        # Two names on one spot make a zero-length side, exactly as a
        # zero-length segment: there is no side to draw (and no direction to
        # measure a label's clearance from).
        for k, v in enumerate(el.vertices):
            w = el.vertices[(k + 1) % len(el.vertices)]
            if np.linalg.norm(points[w] - points[v]) <= tol:
                raise ValueError(f"polygon side {v}{w} has zero length: "
                                 f"{v} and {w} are the same point")
        area = sum(_cross(verts[k], verts[(k + 1) % len(verts)])
                   for k in range(len(verts)))
        if abs(area) <= tol * tol:
            raise ValueError("polygon vertices are collinear")

    elif isinstance(el, IntersectElement):
        for curve_id in el.of:
            if curve_id not in curves:
                raise ValueError(
                    f"no segment or circle with id {curve_id!r}; ids so far: "
                    f"{sorted(curves) or 'none'}"
                )
        if el.of[0] == el.of[1]:
            raise ValueError("cannot intersect an element with itself")
        found = _intersections(curves[el.of[0]], curves[el.of[1]], points, tol)
        # Skip crossings at points that already have a name.
        found = [x for x in found
                 if all(np.linalg.norm(x - q) > tol for q in points.values())]
        if len(found) < len(el.names):
            raise ValueError(
                f"{el.of[0]} and {el.of[1]} meet in {len(found)} new point(s) "
                f"but {len(el.names)} name(s) were given"
            )
        scale = max(tol / TOL, 1e-300)
        found.sort(key=lambda x: (-round(float(x[1]) / scale, 6), float(x[0])))
        for name, at in zip(el.names, found):
            define(name, at)


def _intersections(c1, c2, points, tol) -> list[np.ndarray]:
    def as_geom(curve):
        kind, a, b = curve
        if kind == "segment":
            return kind, (points[a], points[b])
        return kind, (points[a], float(np.linalg.norm(points[b] - points[a])))

    (k1, g1), (k2, g2) = as_geom(c1), as_geom(c2)
    if k1 == "segment" and k2 == "segment":
        return _seg_seg(*g1, *g2, tol)
    if k1 == "circle" and k2 == "circle":
        return _circle_circle(*g1, *g2, tol)
    seg, circ = (g1, g2) if k1 == "segment" else (g2, g1)
    return _seg_circle(*seg, *circ, tol)


# --- component ----------------------------------------------------------------


@register
class GeometryConstruct(Component):
    name = "GeometryConstruct"
    Params = GeometryConstructParams

    def regions(self) -> set[Region]:
        r = {Region.STAGE}
        if any(el.note for el in self.params.construction):
            r.add(Region.LOWER_THIRD)
        return r

    def _weights(self) -> list[float]:
        p: GeometryConstructParams = self.params
        weights = [INTRO_MIN] if p.shapes else []
        for el in p.construction:
            reading = len(el.note.split()) / READ_WPS if el.note else 0.0
            weights.append(max(STEP_MIN[el.kind], reading))
        return weights + [HOLD_MIN]

    def min_seconds(self) -> float:
        """Shortest beat at which every step stays legible.

        The weights passed to budget() are these same minimums, so any beat at
        least this long gives each step at least its own minimum.
        """
        return sum(self._weights())

    def latex_strings(self) -> list[str]:
        # Point labels are plain Text; nothing here goes through LaTeX.
        return []

    def build(self, scene: ChalkdustScene) -> None:
        p: GeometryConstructParams = self.params
        theme = scene.theme
        pieces, labels = _figure(p, theme)
        _check_legible(labels, pieces)

        notes: dict[int, Mobject] = {}
        for i, el in enumerate(p.construction):
            if el.note:
                note = label(body_text(wrap(el.note, NOTE_WRAP), theme),
                             f"note[{i}]")
                fit_to_region(note, Region.LOWER_THIRD)
                notes[i] = note

        # One top-level group that pieces join as they are drawn, so the
        # overlap check sees "the figure" rather than its strokes.
        fig = label(VGroup(), "figure")
        scene.add(fig)
        scene.exclusive(fig, *notes.values())

        times = scene.budget(*self._weights())
        idx = 0
        n_given = len(p.shapes)

        if n_given:
            intro = []
            for piece, el in zip(pieces[:n_given], p.shapes):
                fig.add(piece)
                intro.append(_reveal(piece, el.kind))
            scene.play(*intro, run_time=times[idx])
            idx += 1
            scene.settle("given figure")

        shown: Mobject | None = None
        for i, (piece, el) in enumerate(zip(pieces[n_given:], p.construction)):
            fig.add(piece)
            anims = [_reveal(piece, el.kind)]
            # A note captions its own step only; a stale caption would
            # describe a stroke that is no longer being drawn.
            if shown is not None:
                anims.append(FadeOut(shown))
                shown = None
            if i in notes:
                shown = notes[i]
                anims.append(FadeIn(shown))
            scene.play(*anims, run_time=times[idx])
            idx += 1
            scene.settle(f"construction[{i}]")

        scene.settle("construction complete")
        scene.wait(times[-1])

    @classmethod
    def examples(cls):
        return [
            # Euclid I.1: an equilateral triangle on a given segment.
            {"shapes": [_pt("A", 0, 0), _pt("B", 1, 0), _seg("A", "B")],
             "construction": [
                 _circ("A", "B", "cA", note="Compass on A, opened to B"),
                 _circ("B", "A", "cB", note="Same opening, centred on B"),
                 _meet("cA", "cB", ["C"], note="The circles cross at C"),
                 _seg("A", "C"),
                 _seg("B", "C"),
                 {"kind": "polygon", "vertices": ["A", "B", "C"],
                  "note": "AC and BC are radii, so all three sides equal AB"},
             ]},
            # Perpendicular bisector, ending at the midpoint.
            {"shapes": [_pt("A", -2, 0), _pt("B", 2, 0), _seg("A", "B", "AB")],
             "construction": [
                 _circ("A", "B", "c1", note="Circle at A through B"),
                 _circ("B", "A", "c2", note="Circle at B through A"),
                 _meet("c1", "c2", ["P", "Q"], note="They meet at P and Q"),
                 _seg("P", "Q", "PQ", note="PQ is the perpendicular bisector"),
                 _meet("PQ", "AB", ["M"], note="M is the midpoint of AB"),
             ]},
            # Angle bisector, uncaptioned: the narration carries it.
            {"shapes": [_pt("O", 0, 0), _pt("A", 4, 0), _pt("B", 2, 3.4641016),
                        _seg("O", "A", "OA"), _seg("O", "B", "OB"),
                        _pt("R", 1.5, 0)],
             "construction": [
                 _circ("O", "R", "cO"),
                 _meet("cO", "OB", ["S"]),
                 _circ("R", "S", "cR"),
                 _circ("S", "R", "cS"),
                 _meet("cR", "cS", ["T"]),  # the other crossing is O itself
                 _seg("O", "T"),
             ]},
        ]

    @classmethod
    def stress(cls):
        url = "https://example.org/euclid/elements/book-one/proposition-1"
        return [
            # (a) 3x volume: a full hexagon build, every step captioned at
            # length. Refuses: a caption three lines deep cannot sit legibly
            # in LOWER_THIRD.
            {"shapes": [_pt("O", 0, 0), _pt("A", 1, 0)],
             "construction": [
                 {**step, "note": "This step is described at far greater "
                  "length than any caption should be, repeating what the "
                  "narration already says and then saying it once more"}
                 for step in _hexagon_steps()]},
            # (a) 3x volume, uncaptioned: seven labelled points and fifteen
            # strokes in one figure. Must lay out or refuse cleanly.
            {"shapes": [_pt("O", 0, 0), _pt("A", 1, 0)],
             "construction": _hexagon_steps()},
            # (b) Unwrappable tokens: a URL in a caption and a 60-char id.
            {"shapes": [_pt("A", 0, 0), _pt("B", 1, 0)],
             "construction": [
                 _circ("A", "B", "x" * 60, note=f"Radius from {url}"),
                 _circ("B", "A", "cB"),
                 _meet("x" * 60, "cB", ["C"]),
             ]},
            # (c) Minimal: a lone point, given or constructed.
            {"shapes": [_pt("A", 0, 0)]},
            {"construction": [_pt("A", 0, 0, note="A point")]},
            # (c) Empty caption: whitespace only is no note at all.
            {"shapes": [_pt("A", 0, 0), _pt("B", 1, 0)],
             "construction": [_circ("A", "B", "cA", note="   ")]},
            # Hostile geometry, all schema-valid: two points almost on top
            # of each other (labels collide), a circle that is a speck beside
            # a long segment, and a figure at an absurd offset and scale.
            {"shapes": [_pt("A", 0, 0), _pt("B", 1e-4, 0), _pt("C", 10, 0),
                        _seg("A", "C")]},
            {"shapes": [_pt("A", 0, 0), _pt("B", 1000, 0), _seg("A", "B"),
                        _pt("C", 500, 3), _pt("D", 500.01, 3)],
             "construction": [_circ("C", "D", "tiny")]},
            {"shapes": [_pt("A", 1e6, 1e6), _pt("B", 1e6 + 1e5, 1e6),
                        _seg("A", "B")],
             "construction": [_circ("A", "B", "cA"), _circ("B", "A", "cB"),
                              _meet("cA", "cB", ["C", "D"])]},
            # A polygon side one notch above the zero-length rejection: two
            # vertices 1e-6 apart in a unit figure. Schema-valid, so it must
            # reach layout and refuse as illegible (one visible dot, two
            # names) -- never crash placing labels around a speck of a side.
            {"shapes": [_pt("A", 0, 0), _pt("B", 1e-6, 0), _pt("C", 1, 0),
                        _pt("D", 0, 1),
                        {"kind": "polygon", "vertices": ["A", "B", "C", "D"]}]},
            # (d) Invalid LaTeX has no stress case: this component compiles no
            # maths, and a caption carrying TeX markup is refused at rung 1
            # (tests pin it). Symbols that are plain text draw as typed.
            {"shapes": [_pt("A", 0, 0), _pt("B", 1, 0)],
             "construction": [_circ("A", "B", "cA",
                                    note="r = |AB|, angle BAC = 60°, {} $ ^ _")]},
        ]


# --- the finished figure -------------------------------------------------------


def _figure(p: GeometryConstructParams,
            theme: Theme) -> tuple[list[Mobject], dict[str, Mobject]]:
    """Every piece of the finished figure at its final place in STAGE, one per
    element (shapes, then construction steps), plus the point labels by name.

    Built in full before anything is animated, so the layout is decided once
    for the finished figure and never shifts mid-beat. Pure in (params, theme):
    the beat's build() and the carry-in builder both start here, so a carried
    figure is the one the producing beat ended on.
    """
    local = _resolve(p)
    labels = {n: label(body_text(n, theme), f"label {n}") for n in local}
    pos = _place_in_stage(p, local, labels)

    segs: list[tuple[np.ndarray, np.ndarray]] = []
    circles: list[tuple[np.ndarray, float]] = []
    pieces: list[Mobject] = []
    for el in [*p.shapes, *p.construction]:
        given = any(el is s for s in p.shapes)
        pieces.append(_piece(el, given, pos, labels, theme, segs, circles))

    _place_labels(pos, labels, segs, circles)
    fit_to_region(VGroup(*pieces), Region.STAGE)
    return pieces, labels


def _piece(el, given, pos, labels, theme, segs, circles) -> Mobject:
    pal = theme.palette
    if isinstance(el, (PointElement, IntersectElement)):
        names = [el.name] if isinstance(el, PointElement) else el.names
        color = pal.fg if given else pal.accent
        group = VGroup()
        for n in names:
            dot = Dot(pos[n], radius=DOT_RADIUS, color=color)
            dot._chalk_point = n  # type: ignore[attr-defined]
            # Points and their names sit above every stroke.
            dot.set_z_index(2)
            labels[n].set_z_index(2)
            group.add(dot, labels[n])
        return label(group, f"point {' '.join(names)}")

    if isinstance(el, SegmentElement):
        a, b = (pos[n] for n in el.ends)
        segs.append((a, b))
        line = Line(a, b, color=pal.fg, stroke_width=STROKE)
        return label(line, f"segment {''.join(el.ends)}")

    if isinstance(el, CircleElement):
        c, t = pos[el.center], pos[el.through]
        radius = float(np.linalg.norm(t - c))
        circles.append((c, radius))
        # Start the stroke at the `through` point, as the compass would.
        angle = float(np.arctan2(t[1] - c[1], t[0] - c[0]))
        circle = Circle(radius=radius,
                        color=pal.fg if given else pal.muted,
                        stroke_width=STROKE if given else COMPASS_STROKE)
        circle.rotate(angle).move_to(c)
        return label(circle, f"circle {el.center}->{el.through}")

    verts = [pos[v] for v in el.vertices]
    segs.extend(zip(verts, verts[1:] + verts[:1]))
    poly = Polygon(*verts, color=pal.accent, stroke_width=STROKE,
                   fill_color=pal.accent, fill_opacity=POLYGON_FILL)
    poly.set_z_index(-1)  # fill sits beneath the strokes it encloses
    return label(poly, f"polygon {''.join(el.vertices)}")


def _reveal(piece: Mobject, kind: str):
    """The animation that brings one piece on screen.

    The piece is already inside the scene's figure group, so nothing here may
    be an introducer: an introducer gets re-added at the top level, which
    splits the group apart (and LayoutProbe adds introducers unconditionally).
    FadeIn hard-codes introducer=True, hence the opacity route for points.
    """
    if kind in ("point", "intersect"):
        piece.set_opacity(0)
        return piece.animate.set_opacity(1)
    if kind == "polygon":
        return DrawBorderThenFill(piece, introducer=False)
    return Create(piece, introducer=False)


# --- layout -------------------------------------------------------------------


def _place_in_stage(p: GeometryConstructParams, local: dict[str, np.ndarray],
                    labels: dict[str, Mobject]) -> dict[str, np.ndarray]:
    """Map the local frame uniformly into STAGE, leaving room for labels.

    Unlike fit_to_region this scales up as well as down: the local frame has
    no intrinsic size, so "natural size" means nothing for the geometry. Text
    is never scaled here -- labels keep their theme size.
    """
    lo = np.min(list(local.values()), axis=0)
    hi = np.max(list(local.values()), axis=0)
    for el in [*p.shapes, *p.construction]:
        if isinstance(el, CircleElement):
            c = local[el.center]
            r = float(np.linalg.norm(local[el.through] - c))
            lo = np.minimum(lo, c - r)
            hi = np.maximum(hi, c + r)

    margin = max(max(m.width, m.height) for m in labels.values()) + LABEL_BUFF
    inner = region_rect(Region.STAGE).inset(DEFAULT_PADDING + margin)
    span = hi - lo
    ratios = [avail / size for avail, size in
              ((inner.width, span[0]), (inner.height, span[1])) if size > 0]
    s = min(ratios) if ratios else 1.0  # a lone point: any scale will do
    mid = (lo + hi) / 2

    out = {}
    for n, v in local.items():
        x, y = s * (v - mid)
        out[n] = inner.center + np.array([x, y, 0.0])
    return out


def _place_labels(pos: dict[str, np.ndarray], labels: dict[str, Mobject],
                  segs: list[tuple[np.ndarray, np.ndarray]],
                  circles: list[tuple[np.ndarray, float]]) -> None:
    """Put each name beside its point, on the side with the most clearance.

    Clearance is the distance from the label's centre to the nearest stroke,
    other point, or already-placed label. Ties go to the side facing away from
    the figure's centre, which keeps names on the outside of the drawing.
    """
    centroid = np.mean(list(pos.values()), axis=0)
    placed: list[np.ndarray] = []
    for n, at in pos.items():
        lab = labels[n]
        outward = at - centroid
        norm = np.linalg.norm(outward)
        outward = outward / norm if norm > 0 else outward
        others = [q for m, q in pos.items() if m != n] + placed

        best, best_score = None, -np.inf
        for d in _DIRECTIONS:
            lab.next_to(at, d, buff=DOT_RADIUS + LABEL_BUFF)
            c = lab.get_center()
            clearance = min(
                [_dist_to_segment(c, a, b) for a, b in segs]
                + [abs(float(np.linalg.norm(c - o)) - r) for o, r in circles]
                + [float(np.linalg.norm(c - q)) for q in others],
                default=np.inf,
            )
            score = clearance + 1e-3 * float(np.dot(d / np.linalg.norm(d), outward))
            if score > best_score:
                best, best_score = d, score
        lab.next_to(at, best, buff=DOT_RADIUS + LABEL_BUFF)
        placed.append(lab.get_center())


def _dist_to_segment(c: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    length_sq = float(np.dot(ab, ab))
    if length_sq == 0.0:
        # A degenerate segment is a point. Dividing by zero here yields nan,
        # every label direction then scores nan, and none is ever chosen.
        return float(np.linalg.norm(c - a))
    t = float(np.clip(np.dot(c - a, ab) / length_sq, 0.0, 1.0))
    return float(np.linalg.norm(c - (a + t * ab)))


def _check_legible(labels: dict[str, Mobject], pieces: list[Mobject]) -> None:
    """Refuse figures that would render but not read.

    Runs after the final fit, on real scene geometry. Collisions and specks
    are 'illegible': the content is valid, there is just too much of it, too
    close together, for this frame -- the fix is a simpler figure or a split
    beat (SCENE_SPEC.md §4).
    """
    for piece in pieces:
        if isinstance(piece, Circle) and piece.width / 2 < MIN_FEATURE:
            raise LayoutError(f"{piece._chalk_label} is too small to see "
                              f"(radius {piece.width / 2:.2f})", kind="illegible")
        if isinstance(piece, Line) and piece.get_length() < MIN_FEATURE:
            raise LayoutError(f"{piece._chalk_label} is too short to see "
                              f"(length {piece.get_length():.2f})",
                              kind="illegible")

    dots = [m for piece in pieces for m in piece.get_family()
            if hasattr(m, "_chalk_point")]
    # Two names on one visible dot read as one point: misleading, not merely
    # crowded. Labels alone cannot catch this -- they dodge each other.
    for i, a in enumerate(dots):
        for b in dots[i + 1:]:
            if np.linalg.norm(a.get_center() - b.get_center()) < 3 * DOT_RADIUS:
                raise LayoutError(
                    f"points {a._chalk_point} and {b._chalk_point} are too close "
                    "to tell apart at this scale", kind="illegible")

    names = list(labels)
    for i, n in enumerate(names):
        box = bbox(labels[n])
        for m in names[i + 1:]:
            if box.intersects(bbox(labels[m])):
                raise LayoutError(f"labels {n} and {m} collide: the points are "
                                  "too close together at this scale",
                                  kind="illegible")
        for dot in dots:
            if dot._chalk_point != n and box.intersects(bbox(dot)):
                raise LayoutError(f"label {n} covers point {dot._chalk_point}",
                                  kind="illegible")


# --- fixture shorthand --------------------------------------------------------
# Spec dicts exactly as the LLM would emit them; these just keep the fixtures
# readable.


def _pt(name, x, y, **kw):
    return {"kind": "point", "name": name, "at": [x, y], **kw}


def _seg(a, b, id=None, **kw):
    return {"kind": "segment", "ends": [a, b], **({"id": id} if id else {}), **kw}


def _circ(center, through, id, **kw):
    return {"kind": "circle", "center": center, "through": through, "id": id, **kw}


def _meet(a, b, names, **kw):
    return {"kind": "intersect", "of": [a, b], "names": names, **kw}


def _hexagon_steps():
    """Regular hexagon inscribed in a circle, by stepping the radius around:
    fifteen steps from two given points."""
    return [
        _circ("O", "A", "c0"),
        _circ("A", "O", "c1"),
        _meet("c0", "c1", ["B", "F"]),
        _circ("B", "O", "c2"),
        _meet("c0", "c2", ["C"]),   # the other crossing is A
        _circ("C", "O", "c3"),
        _meet("c0", "c3", ["D"]),
        _circ("D", "O", "c4"),
        _meet("c0", "c4", ["E"]),
        _seg("O", "A"), _seg("O", "B"), _seg("O", "C"), _seg("O", "D"),
        _seg("O", "E"),
        {"kind": "polygon", "vertices": ["A", "B", "C", "D", "E", "F"]},
    ]


# --- continuity (SCENE_SPEC.md §6) ---------------------------------------------

# Draw order of a carried figure, back to front: polygon fills, then strokes,
# then points with their names.
_LAYER = {PolygonElement: 0, SegmentElement: 1, CircleElement: 1,
          PointElement: 2, IntersectElement: 2}


@artifact_builder("GeometryConstruct")
def _artifact(params: GeometryConstructParams, theme: Theme) -> Mobject:
    """The finished figure as the beat's last frame shows it: every given and
    constructed piece, every point named. Step captions are not part of it --
    each describes a stroke being drawn, not the figure.

    The beat orders its layers with z-indices (fills under strokes, points
    over both). A carried artifact must not keep them: z-index is global to a
    scene, so a dimmed carried point would draw over the new beat's own
    content. The same order is kept by submobject order instead.
    """
    pieces, _ = _figure(params, theme)
    elements = [*params.shapes, *params.construction]
    order = sorted(range(len(pieces)), key=lambda i: _LAYER[type(elements[i])])
    figure = VGroup(*(pieces[i] for i in order))
    figure.set_z_index(0)  # whole family
    return figure
