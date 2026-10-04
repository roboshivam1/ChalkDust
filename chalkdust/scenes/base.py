"""ChalkdustScene: the base every component renders into.

Two jobs:

  1. Apply the theme (background colour, resolved fonts).
  2. Run layout assertions at settle points -- the moments when animation has
     stopped and the frame should be inspectable. This is where layout bugs
     get caught, at build time, before a frame is ever rendered.

A component that never calls settle() is a component whose layout is never
checked, so `construct` runs a final settle unconditionally.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol

from manim import Mobject, Scene, config

from chalkdust.scenes.regions import (
    LayoutError,
    assert_in_safe_area,
    assert_legible,
    bbox,
)
from chalkdust.scenes.theme import Theme, get_theme, resolve_fonts


class Component(Protocol):
    """Minimal contract. The real base class arrives with the component
    library; this keeps base.py independent of it."""

    def build(self, scene: "ChalkdustScene") -> None: ...


class ChalkdustScene(Scene):
    """Renders exactly one beat.

    Constructed programmatically rather than via the manim CLI, so we can pass
    the component, theme, and measured audio duration straight in.

    NOTE ON ATTRIBUTE NAMES: Manim's Scene owns `self.duration` and overwrites
    it on every play() call. Anything we add here is namespaced to avoid
    colliding with internals we do not control.
    """

    def __init__(
        self,
        component: Component,
        theme: Theme | str = "default",
        duration: float = 5.0,
        strict: bool = True,
        **kwargs,
    ) -> None:
        # Call this FIRST. Scene.__init__ initialises internal state and will
        # clobber any attribute we set beforehand.
        super().__init__(**kwargs)

        raw_theme = get_theme(theme) if isinstance(theme, str) else theme
        self.theme = resolve_fonts(raw_theme)
        self.component = component
        # Audio duration drives animation timing. Components divide this
        # budget; they never hardcode run times (D-002).
        self.beat_duration = float(duration)
        # Captured with the camera's, from the tier config the worker builds
        # the scene under; budget() and the frame quantisation below use it.
        self._chalk_fps = float(config.frame_rate)
        # strict=False downgrades assertion failures to recorded warnings,
        # used when rendering a degraded beat we already know is imperfect.
        self.strict = strict
        # (kind, message) pairs, populated when strict=False. The probe
        # validator reads these; the repair loop dispatches on kind.
        self.layout_warnings: list[tuple[str, str]] = []

    def setup(self) -> None:
        self.camera.background_color = self.theme.palette.bg

    def construct(self) -> None:
        self.component.build(self)
        # Unconditional final check: no component can opt out of validation.
        self.settle("end of beat")

    # --- timing -------------------------------------------------------------

    @property
    def fps(self) -> float:
        """Frame rate of the render tier this scene was built under -- the one
        its camera writes frames at -- not whatever config holds later."""
        return self._chalk_fps

    @property
    def beat_frames(self) -> int:
        """Whole frames this beat lasts: the audio duration rounded UP.

        Never shorter than the audio (D-002), and at most one frame longer --
        so the mux pads under one frame of silence instead of clipping
        narration or holding the last frame for several.
        """
        return frames_covering(self.beat_duration, self.fps)

    def budget(self, *weights: float) -> list[float]:
        """Split the beat's duration into run times by relative weight.

            fade_in, hold, fade_out = scene.budget(1, 4, 1)

        Every run time is a whole number of frames at the active tier's fps,
        and together they are exactly `beat_frames`. Manim renders a play()
        or wait() in whole frames; handing it fractions let each one round on
        its own (plays up, frozen waits down), so a beat drifted from its
        audio by up to a frame per segment. Here the rounding happens once,
        for the whole beat, and get_run_time() below makes Manim render each
        whole-frame run time as exactly that many frames.
        """
        if any(w < 0 for w in weights):
            raise ValueError(f"weights must not be negative: {weights}")
        if sum(weights) <= 0:
            raise ValueError("weights must sum to more than zero")
        if self.beat_duration <= 0:
            # Without this, a zero budget produces run_time=0 and Manim raises
            # deep inside compile_animation_data, far from the real cause.
            raise ValueError(
                f"beat_duration is {self.beat_duration}; must be positive. "
                "Was the speech stage run before rendering?"
            )
        fps = self.fps
        return [n / fps for n in split_frames(self.beat_frames, weights)]

    # --- frame quantisation ---------------------------------------------------
    # Stock Manim turns a run time into frames two different ways: a play()
    # renders len(arange(0, t, 1/fps)) frames (rounds up), a frozen wait()
    # writes int(t / (1/fps)) (rounds down), and float error flips both even
    # for exact multiples of 1/fps (e.g. 23/15 s plays 24 frames). These two
    # overrides give every play and wait in a beat one rule: round the run
    # time to the nearest whole frame, at least one.

    def get_run_time(self, animations) -> float:
        """Snap the run time to whole frames, then hand Manim the MIDDLE of
        the last frame: (n + 0.5) / fps. Both of Manim's rounding paths land
        on n from there whatever the float error -- the frozen-wait path
        truncates it directly, get_time_progression() below handles plays."""
        run_time = super().get_run_time(animations)
        return (whole_frames(run_time, self.fps) + 0.5) / self.fps

    def get_time_progression(self, run_time: float, *args, **kwargs):
        """Exactly n frame times for a run time snapped by get_run_time()."""
        n = int(run_time * self.fps)
        # arange(0, (n - 0.5)/fps, 1/fps) has n elements under any float error.
        return super().get_time_progression((n - 0.5) / self.fps, *args, **kwargs)

    # --- validation ---------------------------------------------------------

    def settle(self, label: str = "settle point") -> None:
        """Assert the current frame is valid.

        Call after any animation that leaves the scene in a state a viewer will
        actually look at. Cheap -- pure geometry on mobjects already in memory,
        no rendering involved.
        """
        for mob in self.mobjects:
            try:
                assert_in_safe_area(mob, label=f"{label}: {_name(mob)}")
                assert_legible(mob, label=f"{label}: {_name(mob)}")
            except LayoutError as exc:
                if self.strict:
                    raise
                self.layout_warnings.append((exc.kind, str(exc)))

        self._check_pairwise_overlap(label)

    def _check_pairwise_overlap(self, label: str) -> None:
        """Flag overlap between top-level mobjects marked mutually exclusive.

        Overlap is only a bug when unintended, so it is opt-in via
        `exclusive()`. Checking everything would fire constantly on legitimate
        composition (labels on axes, callouts on diagrams).
        """
        tagged = [m for m in self.mobjects if getattr(m, "_chalk_exclusive", False)]
        for i, a in enumerate(tagged):
            for b in tagged[i + 1:]:
                if bbox(a).intersects(bbox(b)):
                    msg = f"{label}: {_name(a)} overlaps {_name(b)}"
                    if self.strict:
                        raise LayoutError(msg, kind="overlap")
                    self.layout_warnings.append(("overlap", msg))

    # --- convenience --------------------------------------------------------

    def exclusive(self, *mobs: Mobject) -> None:
        """Mark mobjects that must never overlap each other."""
        for m in mobs:
            m._chalk_exclusive = True  # type: ignore[attr-defined]


def frames_covering(seconds: float, fps: float) -> int:
    """Fewest whole frames lasting at least `seconds`. Rounded to 1e-6 of a
    frame first, so 4.0 s at 15 fps is 60 frames, not 61 from float error."""
    return max(1, math.ceil(round(seconds * fps, 6)))


def whole_frames(seconds: float, fps: float) -> int:
    """Nearest whole number of frames, at least one."""
    return max(1, round(seconds * fps))


def split_frames(total: int, weights: Sequence[float]) -> list[int]:
    """Split `total` frames by relative weight into whole frames summing to
    exactly `total` (largest remainder; ties go to the earlier segment).

    Every positive weight gets at least one frame, since Manim cannot play
    zero frames. That is taken from the largest share; only when there are
    more positive weights than frames does the sum exceed `total` -- a beat
    that short is refused upstream by the duration check (SCENE_SPEC.md §8).
    """
    scale = total / sum(weights)
    shares = [w * scale for w in weights]
    out = [math.floor(x) for x in shares]
    by_remainder = sorted(range(len(out)), key=lambda i: (out[i] - shares[i], i))
    for i in by_remainder[: total - sum(out)]:
        out[i] += 1
    for i, w in enumerate(weights):
        if w > 0 and out[i] == 0:
            donor = max(range(len(out)), key=lambda j: (out[j], -j))
            if out[donor] > 1:
                out[donor] -= 1
            out[i] = 1
    return out


def _name(mob: Mobject) -> str:
    """Readable identifier for error messages. Components can set
    `_chalk_label` to make failures easier to trace."""
    return getattr(mob, "_chalk_label", type(mob).__name__)
