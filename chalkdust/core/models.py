"""Core data model.

The load-bearing idea is the split between spec and state:

  Spec objects (BeatSpec, VideoSpec) are the declarative description of a
  video. They are frozen, fully serialisable, and they are the ONLY thing that
  feeds a cache key.

  State objects (Beat, Video) wrap a spec and carry the artifacts that pipeline
  stages produce. They are mutable and are never hashed.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from chalkdust.core.version import (
    COMPONENT_LIBRARY_VERSION,
    MANIM_VERSION,
    THEME_VERSION,
)

# A beat is one narrative idea. Longer than this and the visual sits static too
# long; shorter and the cut rate is exhausting. Enforced properly in the
# semantic validator later -- this is just a sanity bound on hand-written specs.
MAX_NARRATION_WORDS = 80


class Region(str, Enum):
    """Named areas of the frame. Components declare which ones they occupy so
    the compiler can reject two components claiming the same space."""

    TITLE_BAR = "title_bar"
    STAGE = "stage"  # union of STAGE_LEFT and STAGE_RIGHT
    STAGE_LEFT = "stage_left"
    STAGE_RIGHT = "stage_right"
    LOWER_THIRD = "lower_third"


class Transition(str, Enum):
    """Applied at ffmpeg assembly, not inside Manim -- doing it in Manim would
    couple adjacent beats and break independent rendering (D-005)."""

    CUT = "cut"
    CROSS_FADE = "cross_fade"
    FADE_BLACK = "fade_black"


class Quality(str, Enum):
    """Two-tier rendering (D-006). Everything validates at DRAFT; FINAL runs
    only after all gates pass."""

    DRAFT = "draft"  # 854x480 @ 15fps
    FINAL = "final"  # 1920x1080 @ 60fps

    @property
    def manim_flag(self) -> str:
        return {"draft": "-ql", "final": "-qh"}[self.value]


class VoiceConfig(BaseModel, frozen=True):
    """Frozen because it feeds the TTS cache key."""

    backend: str = "kokoro"
    voice_id: str = "am_michael"
    rate: float = 1.0


class WordTiming(BaseModel):
    """From forced alignment over generated audio. Drives sub-beat sync and
    gives us SRT captions for free."""

    word: str
    start: float
    end: float


class BeatSpec(BaseModel, frozen=True):
    """One beat, declaratively. This is what the LLM emits (Phase 2) and what
    you hand-write for now.

    Note what is absent: coordinates, colours, font sizes, run times. Those are
    the component's business. If it can't be specified, it can't be broken.
    """

    id: str = Field(pattern=r"^b\d{2,3}$")  # b01, b02, ... b103
    narration: str
    component: str
    params: dict[str, Any] = Field(default_factory=dict)
    carry_in: list[str] = Field(default_factory=list)
    transition: Transition = Transition.CUT

    @field_validator("narration")
    @classmethod
    def _narration_sane(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("narration cannot be empty")
        if len(v.split()) > MAX_NARRATION_WORDS:
            raise ValueError(
                f"narration is {len(v.split())} words (max {MAX_NARRATION_WORDS}). "
                "Split this into multiple beats."
            )
        return v


class VideoSpec(BaseModel, frozen=True):
    """A whole video, declaratively."""

    video_id: str
    channel: str = "default"
    theme: str = "default"
    voice: VoiceConfig = VoiceConfig()
    beats: tuple[BeatSpec, ...]

    @model_validator(mode="after")
    def _unique_beat_ids(self) -> VideoSpec:
        ids = [b.id for b in self.beats]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"duplicate beat ids: {sorted(dupes)}")
        return self


class BuildContext(BaseModel, frozen=True):
    """Everything outside the spec that can change rendered output.

    Bundling these into one object is a discipline device: cache-key functions
    take a BuildContext, so it's hard to forget one of the version strings.
    Forgetting one produces stale artifacts, which is a miserable class of bug.
    """

    quality: Quality = Quality.DRAFT
    component_library_version: str = COMPONENT_LIBRARY_VERSION
    theme_version: str = THEME_VERSION
    manim_version: str = MANIM_VERSION


class Beat(BaseModel):
    """Mutable runtime state for one beat. Stages fill these fields in."""

    spec: BeatSpec

    audio_path: Path | None = None
    duration: float | None = None  # seconds, measured from the audio file
    word_timings: list[WordTiming] = Field(default_factory=list)
    render_path: Path | None = None
    degraded: bool = False  # fell back to a simpler component (D-010)

    @property
    def id(self) -> str:
        return self.spec.id


class Video(BaseModel):
    """Mutable runtime state for a whole video."""

    spec: VideoSpec
    beats: list[Beat]
    output_path: Path | None = None

    @classmethod
    def from_spec(cls, spec: VideoSpec) -> Video:
        return cls(spec=spec, beats=[Beat(spec=b) for b in spec.beats])

    @property
    def total_duration(self) -> float | None:
        """None until every beat has been through the speech stage."""
        if any(b.duration is None for b in self.beats):
            return None
        return sum(b.duration for b in self.beats)  # type: ignore[misc]