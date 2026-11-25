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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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
        cues: Mapping[str, float] | None = None,
        **kwargs,
    ) -> None:
        # Call this FIRST. Scene.__init__ initialises internal state and will
        # clobber any attribute we set beforehand.
        super().__init__(**kwargs)
        # Moment name -> seconds into the beat, resolved from the narration and
        # the voice's word timings (core/cues.py). Only an animation component
        # reads them, through cue_timeline(); empty means "no anchors", which every
        # timeline handles by weight. The render key holds them (D-004).
        self.cue_times: dict[str, float] = dict(cues or {})
        # Every Timeline built on this scene, so construct() can check each one
        # played all its moments.
        self._chalk_timelines: list[Timeline] = []

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
        for timeline in self._chalk_timelines:
            timeline.finish()
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

    def cue_timeline(self, *moments: "Moment") -> "Timeline":
        """The beat split into named moments, each starting on its cue.

            tl = scene.cue_timeline(Moment("pose", 2, motion=1.5),
                                Moment("slide", 4), Moment("name", 2))
            tl.play("pose", Create(curve))
            tl.play("slide", tracker.animate.set_value(0.01))
            tl.play("name", Write(formula))

        The word-timed sibling of budget() (D-002, SCENE_SPEC.md §11 rule 4):
        a moment starts at its cue time (self.cue_times) when it has one, and
        otherwise between its anchored neighbours by weight. Every run time is
        whole frames and the moments together last exactly beat_frames.
        """
        return Timeline(self, moments)

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


# Shortest window a moment may get, in seconds. Under about 0.3 s an animation
# reads as a glitch, not a motion; the floor is a little above that.
MIN_MOMENT_SECONDS = 0.35


@dataclass(frozen=True)
class Moment:
    """One named moment of an animation beat (Timeline).

    weight -- its share of an un-cued stretch of the beat, relative to the
              other un-cued moments around it.
    motion -- how long its animation naturally takes, in seconds; the rest of
              its window is a hold. None means the motion fills the window: a
              point that slides for as long as the voice talks about it.
    """

    name: str
    weight: float = 1.0
    motion: float | None = None


@dataclass(frozen=True)
class Window:
    """A moment's place in the beat, in whole frames."""

    start: int  # first frame of the window
    play: int   # frames its animation runs
    hold: int   # frames held still after it


def moment_windows(moments: Sequence[Moment], cue_frames: Mapping[str, int],
                   total: int, fps: float) -> tuple[int, list[Window]]:
    """(lead, windows): frames held before the first moment, then each moment's
    window, together exactly `total` frames.

    Anchors (cue_frames) are kept in moment order and at least
    MIN_MOMENT_SECONDS apart; an anchor that cannot be kept is dropped and its
    moment placed by weight, never refused -- every textual reason to refuse a
    cue was already checked at rung 2, and what is left here is the voice's
    real timing, which only moves anchors by fractions of a second.
    """
    n = len(moments)
    if n == 0:
        raise ValueError("a timeline needs at least one moment")
    floor = max(1, math.ceil(MIN_MOMENT_SECONDS * fps))
    if total < n * floor:
        floor = max(1, total // n)
    # bounds[k] is the first frame of moment k; bounds[n] is the end.
    bounds: list[int | None] = [None] * (n + 1)
    bounds[n] = total
    last_k, last_f = 0, 0  # moment 0 starts at frame 0 at the earliest
    for k, m in enumerate(moments):
        f = cue_frames.get(m.name)
        if f is None:
            continue
        # Room for every moment since the last anchor, a floor each, before
        # this one; and room for this one and every moment after it.
        earliest = last_f + (k - last_k) * floor
        f = max(int(f), earliest)
        latest = total - (n - k) * floor
        if f > latest:
            continue  # cannot keep it and still fit what follows: by weight
        bounds[k] = f
        last_k, last_f = k, f
    if bounds[0] is None:
        bounds[0] = 0
    # Fill each run of unanchored moments between two known bounds by weight.
    k = 0
    while k < n:
        j = k + 1
        while bounds[j] is None:
            j += 1
        span = bounds[j] - bounds[k]  # type: ignore[operator]
        if j - k > 1:
            weights = [max(m.weight, 1e-6) for m in moments[k:j]]
            parts = split_frames(span, weights)
            at = bounds[k]
            for i, p in enumerate(parts[:-1]):
                at += p  # type: ignore[operator]
                bounds[k + 1 + i] = at
        k = j
    lead = bounds[0]  # type: ignore[assignment]
    windows = []
    for k, m in enumerate(moments):
        width = bounds[k + 1] - bounds[k]  # type: ignore[operator]
        width = max(width, 1)
        play = width if m.motion is None else min(width, whole_frames(m.motion, fps))
        windows.append(Window(start=bounds[k], play=play, hold=width - play))  # type: ignore[arg-type]
    return lead, windows  # type: ignore[return-value]


class Timeline:
    """Plays an animation beat moment by moment, in whole frames.

    Built by ChalkdustScene.cue_timeline(). Moments must be played in order, each
    exactly once; finish() (called by the scene at the end of build) checks
    that every moment was played, so a component cannot silently leave part of
    the beat's time unspent.
    """

    def __init__(self, scene: ChalkdustScene, moments: Sequence[Moment]) -> None:
        names = [m.name for m in moments]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate moment names: {names}")
        self.scene = scene
        self.moments = tuple(moments)
        fps = scene.fps
        cue_frames = {name: round(t * fps) for name, t in scene.cue_times.items()
                      if name in names}
        self.lead, self.windows = moment_windows(moments, cue_frames,
                                                 scene.beat_frames, fps)
        self._next = 0
        scene._chalk_timelines.append(self)

    def seconds(self, name: str) -> float:
        """The whole window of a moment, in seconds (play + hold)."""
        w = self.windows[self._index(name)]
        return (w.play + w.hold) / self.scene.fps

    def play(self, name: str, *animations, **kwargs) -> None:
        """Play `animations` over the moment's motion frames, hold for the rest
        of its window, then settle (layout is checked at every moment's end,
        which is also the next moment's start: both ends of every sweep)."""
        self.sequence(name, [(list(animations), 1.0)], **kwargs)

    def sequence(self, name: str, parts: Sequence[tuple[Sequence, float]],
                 **kwargs) -> None:
        """Play several steps inside one moment: its motion frames are split
        between `parts` by weight (an empty step is a wait), then the hold."""
        k = self._index(name)
        if k != self._next:
            expected = self.moments[self._next].name if self._next < len(self.moments) else None
            raise ValueError(f"moment {name!r} played out of order; next is {expected!r}")
        scene, fps = self.scene, self.scene.fps
        if k == 0 and self.lead:
            scene.wait(self.lead / fps)
        w = self.windows[k]
        weights = [max(float(wt), 1e-6) for _, wt in parts]
        frames = split_frames(w.play, weights) if w.play >= len(parts) else [w.play] + [0] * (len(parts) - 1)
        for (anims, _), n in zip(parts, frames):
            if n <= 0:
                continue
            anims = [a for a in anims if a is not None]
            if anims:
                scene.play(*anims, run_time=n / fps, **kwargs)
            else:
                scene.wait(n / fps)
        if w.hold:
            scene.wait(w.hold / fps)
        scene.settle(f"moment {name!r}")
        self._next += 1

    def finish(self) -> None:
        if self._next != len(self.moments):
            left = [m.name for m in self.moments[self._next:]]
            raise ValueError(f"timeline ended with moments never played: {left}")

    def _index(self, name: str) -> int:
        for i, m in enumerate(self.moments):
            if m.name == name:
                return i
        raise KeyError(f"no moment {name!r}; moments are {[m.name for m in self.moments]}")


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
