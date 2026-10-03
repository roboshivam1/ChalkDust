"""VectorField: arrows sampled on a grid over a 2D domain.

The field is a declarative expression pair, e.g. {"x": "-y", "y": "x"}. It is
parsed into a whitelisted AST and evaluated by walking that tree with numpy --
never by eval(). The spec is model output (SCENE_SPEC.md §1); executing it as
Python would hand the model the arbitrary-code escape hatch that RawScene
exists to sandbox (SCENE_SPEC.md §7).
"""

from __future__ import annotations

import ast
import math
from collections.abc import Callable
from typing import Annotated

import numpy as np
from manim import (
    Arrow,
    Create,
    GrowArrow,
    LaggedStart,
    Line,
    ManimColor,
    Rectangle,
    VGroup,
    interpolate_color,
)
from pydantic import Field, field_validator, model_validator

from chalkdust.core.models import Region
from chalkdust.scenes.base import ChalkdustScene
from chalkdust.scenes.components.base import (
    Component,
    ComponentParams,
    label,
    register,
)
from chalkdust.scenes.regions import (
    DEFAULT_PADDING,
    LayoutError,
    fit_to_region,
    region_rect,
)

# --- expression language ----------------------------------------------------
# Deliberately tiny: the two coordinates, two constants, arithmetic, and a
# handful of one-argument functions. Enough for every textbook field (rotation,
# source, saddle, dipole, shear); anything more is a RawScene.

MAX_EXPR_CHARS = 120

_BINOPS: dict[type[ast.operator], Callable] = {
    ast.Add: np.add,
    ast.Sub: np.subtract,
    ast.Mult: np.multiply,
    ast.Div: np.divide,
    ast.Pow: np.power,
}
_FUNCS: dict[str, Callable] = {
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "tanh": np.tanh,
    "exp": np.exp,
    "log": np.log,
    "sqrt": np.sqrt,
    "abs": np.abs,
}
_CONSTS = {"pi": math.pi, "e": math.e}

# (x, y) grids in, values out. A constant expression returns a scalar.
Evaluator = Callable[[np.ndarray, np.ndarray], np.ndarray | float]


def compile_expr(expr: str) -> Evaluator:
    """Parse `expr` into an evaluator, rejecting anything outside the grammar.

    Validation and construction are one walk, so there is no way for a node
    to be evaluated without having been checked. Raises ValueError naming the
    offending construct; pydantic surfaces it as a schema failure (rung 1).
    """
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError:
        raise ValueError(f"{expr!r} is not a valid expression") from None
    return _compile(tree.body)


def _compile(node: ast.AST) -> Evaluator:
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        # float() so ** and / go through numpy and overflow to inf instead
        # of raising OverflowError on Python ints.
        value = float(node.value)
        return lambda x, y: value

    if isinstance(node, ast.Name):
        if node.id == "x":
            return lambda x, y: x
        if node.id == "y":
            return lambda x, y: y
        if node.id in _CONSTS:
            value = _CONSTS[node.id]
            return lambda x, y: value
        raise ValueError(
            f"unknown name {node.id!r}; allowed: x, y, {', '.join(_CONSTS)}"
        )

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        operand = _compile(node.operand)
        if isinstance(node.op, ast.UAdd):
            return operand
        return lambda x, y: np.negative(operand(x, y))

    if isinstance(node, ast.BinOp):
        op = _BINOPS.get(type(node.op))
        if op is None:
            hint = " (use ** for powers)" if isinstance(node.op, ast.BitXor) else ""
            raise ValueError(
                f"operator {type(node.op).__name__} is not allowed{hint}"
            )
        left, right = _compile(node.left), _compile(node.right)
        return lambda x, y: op(left(x, y), right(x, y))

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS:
            raise ValueError(
                f"only these functions are allowed: {', '.join(_FUNCS)}"
            )
        if len(node.args) != 1 or node.keywords:
            raise ValueError(f"{node.func.id}() takes exactly one argument")
        fn, arg = _FUNCS[node.func.id], _compile(node.args[0])
        return lambda x, y: fn(arg(x, y))

    raise ValueError(f"{type(node).__name__} is not allowed in a field expression")


# --- params -----------------------------------------------------------------

# Domain bounds are maths coordinates, not frame coordinates: the component
# maps whatever domain it is given onto the stage.
Bound = Annotated[float, Field(ge=-1000, le=1000)]

# Width/height of the domain. Outside this the plane becomes a strip the
# stage cannot show at a useful size.
MIN_ASPECT, MAX_ASPECT = 0.5, 6.0


class FieldFn(ComponentParams):
    """F(x, y) = (x_component, y_component), each a whitelisted expression."""

    x: str = Field(min_length=1, max_length=MAX_EXPR_CHARS)
    y: str = Field(min_length=1, max_length=MAX_EXPR_CHARS)

    @field_validator("x", "y")
    @classmethod
    def _parses(cls, v: str) -> str:
        compile_expr(v)
        return v


class VectorFieldParams(ComponentParams):
    field_fn: FieldFn
    # Arrows along the longer side of the domain; the shorter side gets
    # proportionally fewer. Bounded at schema level, and build() still refuses
    # a grid too fine for the plane it lands on (a square domain fills less of
    # the stage than a wide one).
    sample_density: int = Field(default=12, ge=4, le=20)
    # Optional (not in the §5 key params): the maths domain. A field like
    # sin(x) means nothing without its interval, exactly as GraphPlot needs
    # x_range. The default matches the stage's aspect ratio.
    x_range: tuple[Bound, Bound] = (-5.0, 5.0)
    y_range: tuple[Bound, Bound] = (-2.0, 2.0)

    @model_validator(mode="after")
    def _drawable(self) -> VectorFieldParams:
        for name, (lo, hi) in (("x_range", self.x_range), ("y_range", self.y_range)):
            if not lo < hi:
                raise ValueError(f"{name} must be increasing, got [{lo}, {hi}]")
        aspect = _span(self.x_range) / _span(self.y_range)
        if not MIN_ASPECT <= aspect <= MAX_ASPECT:
            raise ValueError(
                f"domain aspect (x span / y span) is {aspect:.2f}; must be "
                f"between {MIN_ASPECT} and {MAX_ASPECT} to fill the stage"
            )
        _, _, u, v = sample(self)
        if not _drawable_mask(u, v).any():
            # Caught here rather than at build: an empty plane would render
            # "fine" and show nothing, which is a broken beat (SCENE_SPEC §11.1).
            raise ValueError(
                "field_fn is zero or undefined at every sample point of the "
                "domain; there is nothing to draw"
            )
        return self


def _span(r: tuple[float, float]) -> float:
    return r[1] - r[0]


def grid_shape(p: VectorFieldParams) -> tuple[int, int]:
    """(columns, rows) of sample points."""
    xs, ys = _span(p.x_range), _span(p.y_range)
    longest = max(xs, ys)
    nx = max(2, round(p.sample_density * xs / longest))
    ny = max(2, round(p.sample_density * ys / longest))
    return nx, ny


def sample(p: VectorFieldParams) -> tuple[np.ndarray, ...]:
    """Evaluate the field at cell centres. Returns flat (x, y, u, v).

    Cell centres rather than edges so every arrow, centred on its sample and
    shorter than its cell, stays inside the domain rectangle.
    """
    nx, ny = grid_shape(p)
    (x0, x1), (y0, y1) = p.x_range, p.y_range
    xs = x0 + (np.arange(nx) + 0.5) * (x1 - x0) / nx
    ys = y0 + (np.arange(ny) + 0.5) * (y1 - y0) / ny
    # indexing="ij" so the flattened order runs column by column, which is
    # the left-to-right order the reveal sweeps in.
    X, Y = (a.ravel() for a in np.meshgrid(xs, ys, indexing="ij"))
    # Singularities (1/r at the origin), domain errors (log of a negative) and
    # overflow (exp of a large number) are expected, not exceptional: they
    # become inf/nan and the arrow at that point is simply not drawn.
    with np.errstate(all="ignore"):
        U = np.broadcast_to(compile_expr(p.field_fn.x)(X, Y), X.shape)
        V = np.broadcast_to(compile_expr(p.field_fn.y)(X, Y), X.shape)
        return X, Y, U.astype(float), V.astype(float)


def _drawable_mask(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        mag = np.hypot(u, v)
    return np.isfinite(mag) & (mag > 0)


# --- layout constants -------------------------------------------------------

# Below this cell size (Manim units) arrows read as noise rather than a field.
# Roughly 18px at 480p.
MIN_CELL = 0.3
# Longest arrow as a fraction of its cell, so neighbours never touch.
ARROW_FILL = 0.8
# Arrows are scaled against this percentile of magnitude, not the maximum, so
# one point near a singularity cannot shrink every other arrow to nothing.
# Anything above it is clamped to full length.
REF_PERCENTILE = 90
# Arrows shorter than this fraction of full length are omitted: at that size
# the tip is the whole arrow and it reads as a blob, not a direction.
MIN_VISIBLE = 0.12
# Background grid aims for about this many lines across the longer side.
GRID_LINES = 8

# Per-step legibility floors in seconds (SCENE_SPEC.md §8 rung 2). They double
# as the budget weights, so at exactly min_seconds() every step gets its floor
# and above it every step stretches in proportion.
MIN_PLANE_SECONDS = 0.5
MIN_ARROWS_SECONDS = 1.5
# A field is read by looking at it, so the still frame gets the largest share.
MIN_HOLD_SECONDS = 2.0


@register
class VectorField(Component):
    name = "VectorField"
    Params = VectorFieldParams

    def regions(self) -> set[Region]:
        return {Region.STAGE}

    def min_seconds(self) -> float:
        return MIN_PLANE_SECONDS + MIN_ARROWS_SECONDS + MIN_HOLD_SECONDS

    def latex_strings(self) -> list[str]:
        # No axis numbers or formula: nothing here goes through LaTeX.
        return []

    def build(self, scene: ChalkdustScene) -> None:
        p: VectorFieldParams = self.params
        theme = scene.theme
        muted = ManimColor(theme.palette.muted)
        accent = ManimColor(theme.palette.accent)

        (x0, x1), (y0, y1) = p.x_range, p.y_range
        inner = region_rect(Region.STAGE).inset(DEFAULT_PADDING)
        # One scale for both axes, so a rotation field looks like a rotation
        # and not an ellipse.
        unit = min(inner.width / (x1 - x0), inner.height / (y1 - y0))
        nx, ny = grid_shape(p)
        cell = unit * min((x1 - x0) / nx, (y1 - y0) / ny)
        if cell < MIN_CELL:
            raise LayoutError(
                f"VectorField grid is {nx}x{ny} on a {unit * (x1 - x0):.1f}x"
                f"{unit * (y1 - y0):.1f} plane: cells of {cell:.2f} are below "
                f"the {MIN_CELL} floor. Lower sample_density or widen the domain.",
                kind="overflow",
            )

        def local(x: float, y: float) -> np.ndarray:
            # Relative to the domain's centre; fit_to_region places the
            # whole group afterwards (SCENE_SPEC.md §4: no absolute coords).
            return np.array([(x - (x0 + x1) / 2) * unit, (y - (y0 + y1) / 2) * unit, 0.0])

        # --- plane: border, light grid, stronger axes where 0 is in range --
        border = Rectangle(width=unit * (x1 - x0), height=unit * (y1 - y0),
                           stroke_color=muted, stroke_width=2, stroke_opacity=0.5)
        lines = []
        step = _nice_step(max(x1 - x0, y1 - y0))
        for gx in _ticks(x0, x1, step):
            lines.append(_grid_line(local(gx, y0), local(gx, y1), muted, gx == 0))
        for gy in _ticks(y0, y1, step):
            lines.append(_grid_line(local(x0, gy), local(x1, gy), muted, gy == 0))
        plane = label(VGroup(border, *lines), "field plane")

        # --- arrows --------------------------------------------------------
        X, Y, U, V = sample(p)
        ok = _drawable_mask(U, V)
        with np.errstate(all="ignore"):
            mag = np.hypot(U, V)
        ref = float(np.percentile(mag[ok], REF_PERCENTILE))
        full = ARROW_FILL * cell

        arrows = []
        for x, y, u, v, m in zip(X[ok], Y[ok], U[ok], V[ok], mag[ok]):
            t = min(1.0, m / ref)
            if t < MIN_VISIBLE:
                continue
            # Centred on the sample point and at most ARROW_FILL of a cell
            # long, so the clamp is also what keeps arrows inside the plane.
            half = np.array([u / m, v / m, 0.0]) * full * t / 2
            centre = local(x, y)
            arrows.append(label(
                Arrow(centre - half, centre + half, buff=0, stroke_width=4,
                      max_tip_length_to_length_ratio=0.35,
                      color=interpolate_color(muted, accent, t)),
                f"arrow({x:.2f}, {y:.2f})",
            ))
        field = label(VGroup(*arrows), "field arrows")

        fit_to_region(VGroup(plane, field), Region.STAGE)

        # Plane first so the viewer has the frame of reference, then the
        # arrows sweep in left to right, then the hold, which carries most of
        # the narration.
        t_plane, t_arrows, t_hold = scene.budget(
            MIN_PLANE_SECONDS, MIN_ARROWS_SECONDS, MIN_HOLD_SECONDS)
        scene.play(Create(plane), run_time=t_plane)
        # Added explicitly so the geometric probe sees the arrows as a group;
        # GrowArrow then reveals each one from its centre-tail.
        scene.add(field)
        scene.play(
            LaggedStart(*(GrowArrow(a) for a in arrows),
                        lag_ratio=1 / max(len(arrows) - 1, 1)),
            run_time=t_arrows,
        )
        scene.settle("vector field drawn")
        scene.wait(t_hold)

    @classmethod
    def examples(cls):
        return [
            # Rotation: the canonical curl picture.
            {"field_fn": {"x": "-y", "y": "x"}},
            # Saddle, on a squarer domain with a sparser grid.
            {"field_fn": {"x": "x", "y": "-y"}, "sample_density": 8,
             "x_range": [-3, 3], "y_range": [-2, 2]},
            # A periodic field over a few periods.
            {"field_fn": {"x": "sin(y)", "y": "sin(x)"}, "sample_density": 14,
             "x_range": [-2 * math.pi, 2 * math.pi], "y_range": [-math.pi, math.pi]},
        ]

    @classmethod
    def stress(cls):
        long_expr = "+".join(["sin(x)*cos(y)"] * 8)  # 111 chars, no spaces
        return [
            # 3x realistic density on a square domain: cells fall below the
            # floor, so this must refuse with overflow.
            {"field_fn": {"x": "-y", "y": "x"}, "sample_density": 20,
             "x_range": [-2, 2], "y_range": [-2, 2]},
            # Maximum density where it does fit, with maximum-length
            # expressions that contain no spaces at all.
            {"field_fn": {"x": long_expr, "y": long_expr.replace("sin", "cos")},
             "sample_density": 20, "x_range": [-6, 6], "y_range": [-2, 2]},
            # Minimal: sparsest grid, constant field.
            {"field_fn": {"x": "1", "y": "0"}, "sample_density": 4},
            # Singular at the origin, which is a sample point (odd grid):
            # inf/nan there, huge magnitudes nearby that must be clamped.
            {"field_fn": {"x": "x/(x**2+y**2)**1.5", "y": "y/(x**2+y**2)**1.5"},
             "sample_density": 9, "x_range": [-3, 3], "y_range": [-1, 1]},
            # Hundreds of orders of magnitude across the domain, overflowing
            # to inf along the top row.
            {"field_fn": {"x": "exp(10*x)", "y": "10**(400*y)"}},
            # Undefined over three quadrants.
            {"field_fn": {"x": "sqrt(x)", "y": "log(y)"}},
        ]


def _nice_step(span: float) -> float:
    """A 1/2/5 x 10^k grid spacing giving about GRID_LINES lines over `span`."""
    raw = span / GRID_LINES
    k = 10 ** math.floor(math.log10(raw))
    return next(m * k for m in (1, 2, 5, 10) if m * k >= raw)


def _ticks(lo: float, hi: float, step: float) -> list[float]:
    """Multiples of `step` strictly inside (lo, hi); the border covers the ends."""
    first = math.floor(lo / step) + 1
    last = math.ceil(hi / step) - 1
    return [i * step for i in range(first, last + 1)]


def _grid_line(start: np.ndarray, end: np.ndarray, color: ManimColor,
               is_axis: bool) -> Line:
    return Line(start, end, stroke_color=color,
                stroke_width=2 if is_axis else 1,
                stroke_opacity=0.7 if is_axis else 0.25)
