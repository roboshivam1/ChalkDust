"""What RawScene code may see (SCENE_SPEC.md §7): an explicit allowlist.

Generated code never touches a real module. It runs in a namespace built from
these names alone -- each looked up once in the installed package and copied
into a plain namespace object -- so nothing it can name leads back to a
module's globals (manim.utils.commands, manim.utils.file_ops, manim._config,
numpy.lib, ctypes, os ...). Adding a name here is a reviewed decision: it is
the whole surface the escape hatch exposes.

Categories, per the review that set this policy:
  mobjects, animations, scenes  -- manim classes; file-reading ones excluded
                                   (ImageMobject, SVGMobject, Code, Typst)
  constants, colours            -- directions, buffers, PI/TAU, font weights,
                                   the standard ManimColor constants
  math utilities                -- rate functions, space_ops, bezier, colour
                                   arithmetic, updater helpers
  math                          -- the whole math module (pure functions)
  numpy                         -- a vetted subset: array construction and
                                   elementwise maths, plus linalg / random
                                   subsets; no numpy.lib, ctypes, load/save,
                                   f2py or anything that reaches the OS

Every name must exist in the pinned Manim / numpy; `vetted_modules()` raises
if one does not, so a version bump that drops a name fails loudly (the
RawScene drift guard in tests/test_raw_scene.py).
"""

from __future__ import annotations

import math
from functools import cache
from types import SimpleNamespace

MOBJECTS = frozenset("""
Angle AnimatedBoundary AnnotationDot AnnularSector Annulus Arc ArcBetweenPoints
ArcBrace ArcPolygon ArcPolygonFromArcs Arrow Arrow3D ArrowCircleFilledTip
ArrowCircleTip ArrowSquareFilledTip ArrowSquareTip ArrowTip
ArrowTriangleFilledTip ArrowTriangleTip ArrowVectorField Axes
BackgroundRectangle BarChart Brace BraceBetweenPoints BraceLabel BraceText
BulletedList Circle ComplexPlane ComplexValueTracker Cone ConvexHull
ConvexHull3D Cross Cube CubicBezier CurvedArrow CurvedDoubleArrow
CurvesAsSubmobjects Cutout Cylinder DashedLine DashedVMobject DecimalMatrix
DecimalNumber DecimalTable DiGraph Difference Dodecahedron Dot Dot3D
DoubleArrow Elbow Ellipse Exclusion FullScreenRectangle FunctionGraph Graph
Group Icosahedron ImplicitFunction Integer IntegerMatrix IntegerTable
Intersection Label LabeledArrow LabeledDot LabeledLine LabeledPolygram Line
Line3D ManimBanner MarkupText MathTable MathTex Matrix Mobject Mobject1D
Mobject2D MobjectMatrix MobjectTable NumberLine NumberPlane Octahedron PGroup
PMobject Paragraph ParametricFunction Point PointCloudDot PolarPlane Polygon
Polygram Polyhedron Prism Rectangle RegularPolygon RegularPolygram RightAngle
RoundedRectangle SampleSpace ScreenRectangle Sector SingleStringMathTex Sphere
Square Star StealthTip StreamLines Surface SurroundingRectangle Table
TangentLine TangentialArc Tetrahedron Tex Text ThreeDAxes ThreeDVMobject
TipableVMobject Title Torus TracedPath Triangle Underline Union UnitInterval
VDict VGroup VMobject ValueTracker Variable Vector VectorField
VectorizedPoint
""".split())

ANIMATIONS = frozenset("""
Add AddTextLetterByLetter AddTextWordByWord Animation AnimationGroup
ApplyComplexFunction ApplyFunction ApplyMatrix ApplyMethod
ApplyPointwiseFunction ApplyPointwiseFunctionToCenter ApplyWave Blink
Broadcast ChangeDecimalToValue ChangeSpeed ChangingDecimal Circumscribe
ClockwiseTransform ComplexHomotopy CounterclockwiseTransform Create
CyclicReplace DrawBorderThenFill FadeIn FadeOut FadeToColor FadeTransform
FadeTransformPieces Flash FocusOn GrowArrow GrowFromCenter GrowFromEdge
GrowFromPoint Homotopy Indicate LaggedStart LaggedStartMap
MaintainPositionRelativeTo MoveAlongPath MoveToTarget PhaseFlow
RemoveTextLetterByLetter ReplacementTransform Restore Rotate Rotating
ScaleInPlace ShowIncreasingSubsets ShowPartial ShowPassingFlash
ShowPassingFlashWithThinningStrokeWidth ShowSubmobjectsOneByOne
ShrinkToCenter SmoothedVectorizedHomotopy SpinInFromNothing SpiralIn
Succession Swap Transform TransformAnimations TransformFromCopy
TransformMatchingShapes TransformMatchingTex TypeWithCursor Uncreate
UntypeWithCursor Unwrite UpdateFromAlphaFunc UpdateFromFunc Wait Wiggle Write
""".split())

SCENES = frozenset("""
LinearTransformationScene MovingCameraScene Scene SpecialThreeDScene
ThreeDScene VectorScene ZoomedScene
""".split())

CONSTANTS = frozenset("""
BOLD BOOK DEFAULT_ARROW_TIP_LENGTH DEFAULT_DASH_LENGTH DEFAULT_DOT_RADIUS
DEFAULT_FONT_SIZE DEFAULT_MOBJECT_TO_EDGE_BUFFER
DEFAULT_MOBJECT_TO_MOBJECT_BUFFER DEFAULT_POINTWISE_FUNCTION_RUN_TIME
DEFAULT_POINT_DENSITY_1D DEFAULT_POINT_DENSITY_2D DEFAULT_SMALL_DOT_RADIUS
DEFAULT_STROKE_WIDTH DEFAULT_WAIT_TIME DEGREES DL DOWN DR HEAVY IN ITALIC
LARGE_BUFF LEFT LIGHT MEDIUM MED_LARGE_BUFF MED_SMALL_BUFF NORMAL OBLIQUE
ORIGIN OUT PI RIGHT SCALE_FACTOR_PER_FONT_POINT SEMIBOLD SEMILIGHT SMALL_BUFF
TAU THIN UL ULTRABOLD ULTRAHEAVY ULTRALIGHT UP UR X_AXIS Y_AXIS Z_AXIS
CapStyleType LineJointType
""".split())

COLOURS = frozenset("""
BLACK BLUE BLUE_A BLUE_B BLUE_C BLUE_D BLUE_E DARKER_GRAY DARKER_GREY
DARK_BLUE DARK_BROWN DARK_GRAY DARK_GREY GOLD GOLD_A GOLD_B GOLD_C GOLD_D
GOLD_E GRAY GRAY_A GRAY_B GRAY_BROWN GRAY_C GRAY_D GRAY_E GREEN GREEN_A
GREEN_B GREEN_C GREEN_D GREEN_E GREY GREY_A GREY_B GREY_BROWN GREY_C GREY_D
GREY_E LIGHTER_GRAY LIGHTER_GREY LIGHT_BROWN LIGHT_GRAY LIGHT_GREY LIGHT_PINK
LOGO_BLACK LOGO_BLUE LOGO_GREEN LOGO_RED LOGO_WHITE MAROON MAROON_A MAROON_B
MAROON_C MAROON_D MAROON_E ORANGE PINK PURE_BLUE PURE_GREEN PURE_RED PURPLE
PURE_CYAN PURE_MAGENTA PURE_YELLOW PURPLE_A PURPLE_B PURPLE_C PURPLE_D PURPLE_E RED
RED_A RED_B RED_C RED_D
RED_E TEAL TEAL_A TEAL_B TEAL_C TEAL_D TEAL_E WHITE YELLOW YELLOW_A YELLOW_B
YELLOW_C YELLOW_D YELLOW_E
ManimColor average_color color_gradient color_to_int_rgb color_to_int_rgba
color_to_rgb color_to_rgba hex_to_rgb interpolate_color invert_color
random_bright_color random_color rgb_to_color rgb_to_hex rgba_to_color
""".split())

# Pure functions of their arguments: no file, process, config or global
# state. Deliberately absent: capture/get_video_metadata (utils.commands),
# everything in utils.file_ops, tempconfig/config, register_font, images,
# sounds, plugins, logger/console.
MATH_UTILITIES = frozenset("""
always always_redraw always_rotate always_shift cycle_animation f_always
turn_animation_into_updater
bezier integer_interpolate interpolate inverse_interpolate match_interpolate
mid partial_bezier_points
clockwise_path counterclockwise_path path_along_arc straight_path
binary_search choose clip sigmoid
R3_to_complex angle_axis_from_quaternion angle_between_vectors angle_of_vector
cartesian_to_spherical center_of_mass compass_directions
complex_func_to_R3_func complex_to_R3 cross2d earclip_triangulation
find_intersection get_unit_normal get_winding_number line_intersection
midpoint normalize perpendicular_bisector quaternion_conjugate
quaternion_from_angle_axis quaternion_mult regular_vertices rotate_vector
rotation_about_z rotation_matrix shoelace shoelace_direction
spherical_to_cartesian thick_diagonal z_to_vector
index_labels get_det_text matrix_to_mobject matrix_to_tex_string
""".split())

RATE_FUNCTIONS = frozenset("""
double_smooth ease_in_back ease_in_bounce ease_in_circ ease_in_cubic
ease_in_elastic ease_in_expo ease_in_out_back ease_in_out_bounce
ease_in_out_circ ease_in_out_cubic ease_in_out_elastic ease_in_out_expo
ease_in_out_quad ease_in_out_quart ease_in_out_quint ease_in_out_sine
ease_in_quad ease_in_quart ease_in_quint ease_in_sine ease_out_back
ease_out_bounce ease_out_circ ease_out_cubic ease_out_elastic ease_out_expo
ease_out_quad ease_out_quart ease_out_quint ease_out_sine exponential_decay
linear lingering not_quite_there running_start rush_from rush_into slow_into
smooth smoothererstep smootherstep smoothstep squish_rate_func there_and_back
there_and_back_with_pause wiggle
""".split())

# The star-importable rate functions; the rest are reached as
# rate_functions.<name>, as in Manim itself.
_STAR_RATE_FUNCTIONS = frozenset("""
double_smooth exponential_decay linear lingering not_quite_there
running_start rush_from rush_into slow_into smooth smoothererstep
smootherstep smoothstep squish_rate_func there_and_back
there_and_back_with_pause wiggle
""".split())

NUMPY = frozenset("""
abs absolute add all allclose any append arange arccos arccosh arcsin arcsinh
arctan arctan2 arctanh argmax argmin argsort around array array_equal asarray
average bool_ cbrt ceil clip column_stack complex128 concatenate conj cos
cosh count_nonzero cross cumprod cumsum deg2rad degrees diag diff divide dot
e empty_like exp exp2 expm1 eye flip fliplr flipud float32 float64 floor
floor_divide fmod full full_like gcd heaviside hstack hypot identity imag inf
inner int32 int64 interp isclose isfinite isinf isnan isscalar lcm linspace
log log10 log1p log2 logspace matmul max maximum mean median meshgrid min
minimum mod modf multiply nan nan_to_num ndim negative newaxis ones ones_like
outer pi polyfit polyval power prod rad2deg radians ravel real reciprocal
remainder repeat reshape rint roll rot90 round shape sign sin sinh size sort
sqrt square squeeze stack std subtract sum swapaxes take tan tanh tile trace
transpose tril triu trunc unique var vdot vstack where zeros zeros_like
""".split())

NUMPY_LINALG = frozenset("""
cross det eig eigh eigvals eigvalsh inv matrix_power matrix_rank norm pinv qr
solve svd
""".split())

NUMPY_RANDOM = frozenset("""
choice default_rng normal permutation rand randn randint random seed shuffle
uniform
""".split())

MATH = frozenset(name for name in vars(math) if not name.startswith("_"))

# What `from manim import *` binds in RawScene code (and what every RawScene
# namespace starts with).
MANIM = (MOBJECTS | ANIMATIONS | SCENES | CONSTANTS | COLOURS | MATH_UTILITIES
         | _STAR_RATE_FUNCTIONS | {"np", "rate_functions"})


def _copy(module: object, names: frozenset[str], **extra: object) -> SimpleNamespace:
    missing = sorted(n for n in names - set(extra) if not hasattr(module, n))
    if missing:
        raise RuntimeError(
            f"RawScene allowlist names missing from {getattr(module, '__name__', module)}: "
            f"{missing}; the installed version changed -- review the allowlist")
    values = {n: getattr(module, n) for n in names if n not in extra}
    return SimpleNamespace(**values, **extra)


@cache
def vetted_modules() -> dict[str, SimpleNamespace]:
    """Import path -> the namespace RawScene code gets for it. These are the
    only importable paths; each holds copies of allowlisted values, never a
    module object."""
    import manim
    import manim.utils.rate_functions as manim_rate_functions
    import numpy

    linalg = _copy(numpy.linalg, NUMPY_LINALG)
    random = _copy(numpy.random, NUMPY_RANDOM)
    np_ns = _copy(numpy, NUMPY, linalg=linalg, random=random)
    rates = _copy(manim_rate_functions, RATE_FUNCTIONS)
    manim_ns = _copy(manim, MANIM, np=np_ns, rate_functions=rates)
    return {
        "manim": manim_ns,
        "manim.utils.rate_functions": rates,
        "numpy": np_ns,
        "numpy.linalg": linalg,
        "numpy.random": random,
        "math": _copy(math, MATH),
    }


def importable_names(path: str) -> frozenset[str]:
    """Names `from <path> import <name>` may bind."""
    return frozenset(vars(vetted_modules()[path]))
