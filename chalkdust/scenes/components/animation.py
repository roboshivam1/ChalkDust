"""Animation components: a picture that changes while the voice describes it.

The slide components (TitleCard, BulletReveal, AnswerBox...) put text up and
take it down. An animation component instead owns one mathematical picture --
a curve with a point sliding along it, a secant turning into a tangent,
rectangles refining under a curve -- and moves it in step with the words.

What this adds to the component contract (base.Component), and why:

  Phases. One picture usually needs more than one beat (8-25 s each,
  SCENE_SPEC.md §2). An animation declares ordered PHASES; a beat plays one
  of them. Phase k starts exactly where phase k-1 ended, because both are
  computed from the same setup parameters -- so consecutive beats that play
  consecutive phases of the same setup join with no visible cut, while every
  beat still renders on its own from its own spec (D-005 unchanged: no beat
  reads another's mobjects). tests/test_animation.py proves it per animation
  by comparing the last frame of phase k with the first frame of phase k+1.

  Moments and cues. A phase is a list of named moments (scenes/base.Moment).
  Each moment starts when the voice reaches it: at the first word of its cue
  phrase if the spec gives one (`cues`), else at sentence k of the narration,
  else by weight (core/cues.py). Times come from the voice's real word
  timings when it has them (Kokoro), so a moment lands on its word, not on an
  even share of the beat. The cue times are a term of the render key (D-004).

  Setup. Every param except `phase` and `cues` is the setup. Consecutive
  beats that continue a picture must repeat its setup exactly; the semantic
  rung refuses a beat that continues a phase with a different setup (kind
  "continuity"), because the picture would jump at the cut.

Everything else is the ordinary contract: validated, frozen params with no
coordinate, colour or size; examples() and stress(); render or raise. Numbers
on screen come from plotting.Fn and the setup, never from the spec's text.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, ClassVar

from pydantic import Field

from chalkdust.core.models import Region, WordTiming
from chalkdust.scenes.base import MIN_MOMENT_SECONDS, ChalkdustScene, Moment, Timeline
from chalkdust.scenes.components.base import Component, ComponentParams
from chalkdust.scenes.regions import Rect, region_rect

# The share of the frame's width the plot gets; the rest is the maths column
# (the reference layout: picture left, one line of maths at a time right).
PLOT_SHARE = 0.58
COLUMN_GAP = 0.3
# A moment shorter than this cannot be followed; min_seconds() uses it.
SECONDS_PER_MOMENT = 1.2
# Longest cue phrase: a cue quotes the few words a moment starts on.
MAX_CUE_CHARS = 60


class AnimationParams(ComponentParams):
    """Base params of every animation (see the module docstring). Subclasses
    narrow `phase` to a Literal of their phase names, so a JSON schema (and a
    decoding grammar) lists them, and add the setup fields.

    phase -- which phase of the animation this beat plays.
    cues  -- optional: moment name -> the words of the narration it starts on.
    """

    phase: str
    cues: dict[str, str] = Field(default_factory=dict, max_length=8)


class Animation(Component):
    """Base for animation components. Subclasses set `name`, `Params`, PHASES,
    and implement enter() and perform()."""

    # phase name -> its moments, in play order. Ordered: phase k continues k-1.
    PHASES: ClassVar[dict[str, tuple[Moment, ...]]]

    # --- what the rest of the system reads ------------------------------------

    def phase_names(self) -> list[str]:
        return list(self.PHASES)

    def phase_index(self) -> int:
        return self.phase_names().index(self.params.phase)  # type: ignore[attr-defined]

    def moments(self) -> tuple[Moment, ...]:
        return self.PHASES[self.params.phase]  # type: ignore[attr-defined]

    def moment_names(self) -> list[str]:
        return [m.name for m in self.moments()]

    def cue_phrases(self) -> dict[str, str]:
        return dict(self.params.cues)  # type: ignore[attr-defined]

    def setup(self) -> dict[str, Any]:
        """The params that define the picture: everything but phase and cues."""
        return self.params.model_dump(mode="json", exclude={"phase", "cues"})

    def regions(self) -> set[Region]:
        # The whole frame below the title bar: one picture, one owner.
        return {Region.STAGE, Region.LOWER_THIRD}

    def min_seconds(self) -> float:
        return max(SECONDS_PER_MOMENT * len(self.moments()),
                   MIN_MOMENT_SECONDS * len(self.moments()))

    # --- layout helpers (relative to regions only) ---------------------------

    @staticmethod
    def frame_box() -> Rect:
        """STAGE and LOWER_THIRD together: the animation's whole area."""
        s, low = region_rect(Region.STAGE), region_rect(Region.LOWER_THIRD)
        top, bottom = s.top, low.bottom
        return Rect(s.x, (top + bottom) / 2, s.width, top - bottom).inset(0.15)

    @classmethod
    def plot_box(cls) -> Rect:
        f = cls.frame_box()
        w = f.width * PLOT_SHARE
        return Rect(f.left + w / 2, f.y, w, f.height)

    @classmethod
    def column_box(cls) -> Rect:
        f = cls.frame_box()
        w = f.width * (1 - PLOT_SHARE) - COLUMN_GAP
        return Rect(f.right - w / 2, f.y, w, f.height)

    # --- build ----------------------------------------------------------------

    def build(self, scene: ChalkdustScene) -> None:
        self.enter(scene)
        # Test hook: tests/test_animation.py captures the first frame here to
        # compare it with the previous phase's last frame. Inert otherwise.
        hook = getattr(scene, "chalk_on_entry", None)
        if hook is not None:
            hook()
        timeline = scene.timeline(*self.moments())
        self.perform(scene, timeline)

    @abstractmethod
    def enter(self, scene: ChalkdustScene) -> None:
        """Put this phase's starting picture on screen with scene.add(), no
        animation: exactly what the previous phase ends on (nothing, for the
        first phase), so a cut between them is invisible."""

    @abstractmethod
    def perform(self, scene: ChalkdustScene, tl: Timeline) -> None:
        """Play every moment of this phase, in order, through `tl`."""


def cue_times_for(component: Component, narration: str, words: list[WordTiming],
                  duration: float) -> dict[str, float]:
    """Moment -> seconds for an animation beat; {} for any other component."""
    if not isinstance(component, Animation):
        return {}
    from chalkdust.core.cues import cue_times

    return cue_times(narration, component.moment_names(), component.cue_phrases(),
                     words, duration)
