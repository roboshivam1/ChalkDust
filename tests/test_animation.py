"""Animation components (scenes/components/animation.py): the contract checks
the layout tests do not cover.

- Continuity: the first frame of phase k+1 equals the last frame of phase k
  for the same setup, compared on the camera's own pixels (no encoder in
  between, so the comparison is exact). This is what lets consecutive beats
  join with no visible cut while still rendering independently (D-005).
- Every animation's examples() walk its phases in order with one setup, so
  the continuity test covers every phase boundary it has.
- Cue checks run at rung 2 from the text alone.
"""

from __future__ import annotations

import numpy as np
import pytest
from manim import tempconfig
from manim.utils.exceptions import EndSceneEarlyException

import chalkdust.scenes.components  # noqa: F401
from chalkdust.core.models import BeatSpec
from chalkdust.render.worker import long_path
from chalkdust.scenes.base import ChalkdustScene
from chalkdust.scenes.components.animation import Animation
from chalkdust.scenes.components.base import _REGISTRY, make_component
from chalkdust.validate.semantic import check_continuity, validate_semantic

ANIMATIONS = sorted(n for n, c in _REGISTRY.items() if issubclass(c, Animation))
# Small and quick: continuity is a property of the mobjects, not the resolution.
# Frames are really drawn (no skip_animations): the last one drawn is the test.
CAMERA = {"pixel_width": 640, "pixel_height": 360, "frame_rate": 15,
          "write_to_movie": False, "disable_caching": True, "verbosity": "WARNING"}


class _EntryScene(ChalkdustScene):
    """Captures the frame an animation phase starts on, then stops."""

    entry: np.ndarray | None = None

    def chalk_on_entry(self) -> None:
        self.renderer.update_frame(self)
        self.entry = np.array(self.renderer.get_frame())
        raise EndSceneEarlyException()


def _final_frame(component, tmp_path) -> np.ndarray:
    with tempconfig({**CAMERA, "media_dir": str(long_path(tmp_path))}):
        scene = ChalkdustScene(component, duration=9.0)
        scene.render()
        # The camera still holds the last frame written to the movie. Do not
        # redraw the scene: what is on screen after the last frame (an end
        # state swapped in, a changed draw order) is never seen by anyone, and
        # comparing it hid a seam the encoded pilot showed (dots under lines).
        return np.array(scene.renderer.get_frame())


def _entry_frame(component, tmp_path) -> np.ndarray:
    with tempconfig({**CAMERA, "media_dir": str(long_path(tmp_path))}):
        scene = _EntryScene(component, duration=9.0)
        scene.render()
        assert scene.entry is not None
        return scene.entry


def _first_written_frame(component, tmp_path) -> np.ndarray:
    """The first frame the phase writes to the movie (its first play() at
    t = 0), a stronger check than the entry state. Neither check sees a frame
    written later, during the hold before a cued first moment: LimitAtHole's
    readouts blanked there in the encoded reference video (run 3), which only
    the encoded-video seam check caught (.lab/run3/tools/verify_video.py)."""
    with tempconfig({**CAMERA, "media_dir": str(long_path(tmp_path))}):
        scene = ChalkdustScene(component, duration=9.0)
        box: dict[str, np.ndarray] = {}

        def grab(frame, num_frames=1):
            box["frame"] = np.array(frame)
            raise EndSceneEarlyException()

        scene.renderer.add_frame = grab
        scene.render()
        assert "frame" in box
        return box["frame"]


def _chains(cls) -> list[tuple[dict, dict]]:
    ex = cls.examples()
    out = []
    for a, b in zip(ex, ex[1:]):
        ca, cb = cls(a), cls(b)
        if ca.setup() == cb.setup() and cb.phase_index() == ca.phase_index() + 1:
            out.append((a, b))
    return out


def test_there_are_animations():
    assert ANIMATIONS, "no Animation component registered"


@pytest.mark.parametrize("name", ANIMATIONS)
def test_examples_walk_every_phase_of_one_setup(name):
    cls = _REGISTRY[name]
    assert len(_chains(cls)) == len(cls.PHASES) - 1, (
        f"{name}.examples() must start with one example per phase, in phase order, "
        "all with the same setup, so every phase boundary is tested")


@pytest.mark.parametrize("name,pair", [(n, i) for n in ANIMATIONS
                                       for i in range(len(_REGISTRY[n].PHASES) - 1)])
def test_phase_starts_on_previous_phase_last_frame(name, pair, tmp_path):
    cls = _REGISTRY[name]
    a, b = _chains(cls)[pair]
    last = _final_frame(cls(a), tmp_path / "a")
    first = _entry_frame(cls(b), tmp_path / "b")
    diff = np.abs(last.astype(int) - first.astype(int)).max(axis=2)
    assert (diff > 8).sum() == 0, (
        f"{name}: phase {a['phase']!r} ends on a different frame than "
        f"{b['phase']!r} starts on ({int((diff > 8).sum())} pixels differ, max {diff.max()})")


def test_cue_not_in_narration_is_refused_at_rung_2():
    params = {**_REGISTRY["SecantToTangent"].examples()[1], "cues": {"line": "tangent of"}}
    spec = BeatSpec(id="b01", narration="Pick a second point. Draw the line through both. "
                    "The rise over the run is the slope.", component="SecantToTangent",
                    params=params)
    kinds = [f.kind for f in validate_semantic(spec).findings]
    assert kinds == ["cue"]


def test_continuing_a_phase_with_another_setup_is_refused():
    ex = _REGISTRY["SecantToTangent"].examples()
    a = BeatSpec(id="b01", narration="One. Two. Three.", component="SecantToTangent", params=ex[0])
    b = BeatSpec(id="b02", narration="One. Two. Three.", component="SecantToTangent",
                 params={**ex[1], "a": 2})
    (f,) = check_continuity(a, b)
    assert f.kind == "continuity" and "a" in f.message
    assert check_continuity(a, BeatSpec(id="b02", narration="x", component="SecantToTangent",
                                        params=ex[1])) == []


def test_starting_over_at_the_first_phase_is_allowed():
    ex = _REGISTRY["SecantToTangent"].examples()
    a = BeatSpec(id="b01", narration="x", component="SecantToTangent", params=ex[1])
    b = BeatSpec(id="b02", narration="x", component="SecantToTangent", params={**ex[0], "a": 2})
    assert check_continuity(a, b) == []


@pytest.mark.parametrize("name", ANIMATIONS)
def test_cue_names_are_the_phase_moments(name):
    for params in _REGISTRY[name].examples():
        c = make_component(name, params)
        assert c.moment_names() == [m.name for m in c.PHASES[params["phase"]]]


def test_a_smooth_steep_curve_is_not_called_a_kink():
    # One-sided first-order slopes of x^5 - 40x^3 + 3 at -5 differ by about
    # eps*|f''| (2 in 125); the check must not read that as a corner.
    params = _REGISTRY["SecantToTangent"].Params
    params(phase="pose", function="x^5 - 40*x^3 + 3", x_range=[-6, 6], a=-5, h=4)
    with pytest.raises(ValueError, match="not differentiable"):
        params(phase="pose", function="abs(x)", x_range=[-3, 3], a=0, h=1)


# --- SecantToTangent's numbers (run 3 review findings, each pinned) ---------------

import math  # noqa: E402

from chalkdust.scenes.components.secant_tangent import _shown, slope_plan  # noqa: E402
from chalkdust.scenes.plotting import Fn  # noqa: E402


@pytest.mark.parametrize("expr,a,span,true", [
    ("sin(x)", 1000, 10, math.cos(1000)),            # step tied to |x| read 0.55
    ("sin(1000*x)", 0.005, 0.01, 1000 * math.cos(5)),  # read 275.27
    ("exp(x)", 15, 16, math.exp(15)),
])
def test_the_derivative_shown_is_the_derivative(expr, a, span, true):
    plan = slope_plan(Fn(expr), a, 0.3 * span, span)
    assert plan.slope == pytest.approx(true, rel=1e-7)
    assert _shown(plan.slope, plan.decimals) == _shown(true, plan.decimals)


@pytest.mark.parametrize("expr,a,h,span,why", [
    ("sqrt(x)", 1e-10, 0.5, 1, "undefined"),       # f stops being defined right next to a
    ("100*x + 0.5*abs(x-1)", 1, 2, 6, "not differentiable"),  # a 1% corner passed
    ("abs(x)", 0, 1, 6, "not differentiable"),
])
def test_a_slope_that_cannot_be_shown_truly_is_refused(expr, a, h, span, why):
    with pytest.raises(ValueError, match=why):
        slope_plan(Fn(expr), a, h, span)


def test_the_slide_ends_on_the_digits_of_the_derivative():
    # Over a grid of smooth setups the secant's last readout and f'(a)'s
    # readout show the same digits (13% of random setups did not).
    fns = ["sin(x)", "x^3/3 - x", "exp(x/3)", "x^2/4 + 1", "cos(2*x) + x/5", "log(x + 5)"]
    for expr in fns:
        for a in (-2, -0.7, 0.3, 1.1, 2):
            for h in (-1.5, 0.9, 2):
                plan = slope_plan(Fn(expr), a, h, 8)
                f = Fn(expr)
                last = (f.at(a + plan.h_end) - f.at(a)) / plan.h_end
                assert _shown(last, plan.decimals) == _shown(plan.slope, plan.decimals), (expr, a, h)
                assert abs(plan.h_end) <= abs(h) / 16, "the slide must move"


def test_the_slide_moves_even_when_the_first_secant_is_already_tangent():
    plan = slope_plan(Fn("x^3"), -1, 3, 6)    # secant -1..2 has slope 3 = f'(-1)
    assert abs(plan.h_end) <= 3 / 16


def test_the_window_holds_every_height_q_passes():
    from chalkdust.scenes.components.secant_tangent import SecantTangentParams, _Picture
    from chalkdust.scenes.theme import get_theme

    p = SecantTangentParams(phase="shrink", function="1/(0.01+(x-2)^2)", x_range=[0, 10],
                            a=1, h=3)
    pic = _Picture(p, get_theme("default"))
    assert pic.plot.y1 >= 100      # Q crosses the peak f(2) = 100 on its way in


def test_an_exact_rounding_tie_reads_the_same_at_the_slide_end_and_after():
    for expr, a, h, span in [("x^2/4", 0.25, -1, 4), ("sqrt(x)", 16, -9, 25)]:
        plan = slope_plan(Fn(expr), a, h, span)
        f = Fn(expr)
        last = (f.at(a + plan.h_end) - f.at(a)) / plan.h_end
        assert _shown(last, plan.decimals) == _shown(plan.shown, plan.decimals)
        assert abs(plan.shown - plan.slope) <= 0.5 * 10 ** -plan.decimals + 1e-12


def test_small_slopes_keep_three_significant_figures():
    plan = slope_plan(Fn("x^2/1000"), 0.5, 0.4, 2)
    assert _shown(plan.shown, plan.decimals) == "0.00100"


def test_a_slope_near_a_singularity_is_computed_not_refused():
    plan = slope_plan(Fn("log(x)"), 0.05, 1, 4.99)
    assert plan.slope == pytest.approx(20.0, rel=1e-7)


def test_a_point_close_to_the_domain_edge_gets_its_true_slope():
    # Round 1 rendered NaN here (the stencil crossed 0); the slope is real.
    plan = slope_plan(Fn("sqrt(x)"), 0.0015, 0.5, 1)
    assert plan.slope == pytest.approx(0.5 / math.sqrt(0.0015), rel=1e-7)


@pytest.mark.parametrize("tex, start, end", [("n =", 4, 64), (r"\text{slope} =", 0.625, 1234.5),
                                             ("x =", -0.5, 0.998)])
def test_a_readouts_digits_sit_on_its_labels_baseline(tex, start, end):
    # Run 3 review: "slope =" digits sat 5 px low at 1080p (centred on the
    # label's box), and a minus sign or a thousands comma moved them. These
    # labels start with a letter on the baseline, so that letter is the truth.
    from manim import config

    from chalkdust.scenes.plotting import live, readout
    from chalkdust.scenes.theme import get_theme
    theme = get_theme("default")
    r = readout(theme, tex, start, "#ffffff")
    value = [start]
    live(r, lambda: value[0])
    px = 1080 / config.frame_height

    def off() -> float:
        num = r[1]
        digit = num[1] if num.number < 0 else num[0]
        return (digit.get_bottom()[1] - r[0].family_members_with_points()[0].get_bottom()[1]) * px

    assert abs(off()) < 1.0   # a round letter overshoots the line by ~0.6 px
    value[0] = end
    r[1].update()
    assert abs(off()) < 1.0   # a round letter overshoots the line by ~0.6 px



@pytest.mark.parametrize("name,pair", [(n, i) for n in ANIMATIONS
                                       for i in range(len(_REGISTRY[n].PHASES) - 1)])
def test_phase_first_written_frame_is_previous_phase_last_frame(name, pair, tmp_path):
    cls = _REGISTRY[name]
    a, b = _chains(cls)[pair]
    last = _final_frame(cls(a), tmp_path / "a")
    first = _first_written_frame(cls(b), tmp_path / "b")
    diff = np.abs(last.astype(int) - first.astype(int)).max(axis=2)
    # 24, the encoded-video seam check's threshold: the first written frame is
    # at t = 0 of a play(), where a fade may already add a faint pixel (9 in
    # SecantToTangent's name); something missing or moved differs by ~200.
    assert (diff > 24).sum() == 0, (
        f"{name}: the first frame {b['phase']!r} writes differs from the last frame "
        f"{a['phase']!r} wrote ({int((diff > 8).sum())} pixels differ, max {diff.max()})")
