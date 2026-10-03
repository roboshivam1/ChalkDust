"""Phase 0 pipeline: one hand-written spec file in, one finished MP4 out.

Stage order is the architecture, not a convenience (ARCHITECTURE.md §1-2):

  load      -- JSON -> VideoSpec, plus every beat's component params
  validate  -- schema, then the geometric rung, BEFORE any speech or render.
               Fail early: a spec that cannot lay out must not cost a TTS call.
  speech    -- every beat, measured, before any render (D-002)
  render    -- per beat, run time from the measured audio, cached by key (D-004)
  assemble  -- mux, concat, loudness-normalise

Failures are typed by the STAGE they happen in, not by exception class.
assemble.py raises TTSError for ffmpeg failures because it shares
speech.base.run, so catching by type would report an assembly failure as a
speech failure. The CLI maps each type to its own exit code.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from manim import tempconfig
from pydantic import ValidationError

from chalkdust.continuity import ArtifactRecipe, resolve_carry_in
from chalkdust.core.cache import Cache, tts_key
from chalkdust.core.models import BuildContext, CarryInError, Quality, Video, VideoSpec
from chalkdust.render.assemble import assemble
from chalkdust.render.worker import render_beat
from chalkdust.scenes.components import get_component
from chalkdust.scenes.theme import get_theme
from chalkdust.speech.base import TTSError
from chalkdust.speech.tts import resolve_voice, synthesize_beat
from chalkdust.validate.geometric import Report, validate_specs

DEFAULT_CACHE_DIR = Path(".cache")
# Manim scratch, assembly intermediates. Git-ignored; never cwd/media.
DEFAULT_WORK_DIR = Path("work")


# --- failures ---------------------------------------------------------------


class PipelineError(Exception):
    """A known failure, already phrased for the operator."""


class SpecInvalid(PipelineError):
    """Rung 1: the file is not a valid spec. Nothing ran."""


class LayoutRefused(PipelineError):
    """Rung 3: a component refused its content. Nothing was synthesised."""


class SpeechFailed(PipelineError):
    pass


class RenderFailed(PipelineError):
    pass


class AssemblyFailed(PipelineError):
    pass


# --- results ----------------------------------------------------------------


@dataclass(frozen=True)
class BeatOutcome:
    """What the cache did for one beat. Printed per beat so an incremental
    re-run visibly rebuilds only what changed (ROADMAP.md Phase 0 exit)."""

    beat_id: str
    speech_cached: bool
    render_cached: bool
    duration: float


@dataclass
class RunResult:
    output: Path
    beats: list[BeatOutcome] = field(default_factory=list)

    @property
    def rebuilt(self) -> list[str]:
        return [b.beat_id for b in self.beats if not b.render_cached]


# --- stages -----------------------------------------------------------------


def _format_validation(exc: ValidationError, indent: str = "  ") -> str:
    return "\n".join(
        f"{indent}{'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}"
        for err in exc.errors()
    )


def load_spec(path: Path) -> VideoSpec:
    """Rung 1 (schema), including each beat's component params.

    VideoSpec alone accepts any `component` string and any `params` dict; the
    component's own Params model is what knows they are wrong. Checking both
    here keeps "spec invalid" distinct from "layout refused" -- the geometric
    probe would otherwise fold a bad param into a build_error finding.
    """
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise SpecInvalid(f"cannot read spec {path}: {exc.strerror}") from exc

    try:
        spec = VideoSpec.model_validate_json(text)
    except ValidationError as exc:
        raise SpecInvalid(f"{path} is not a valid spec:\n"
                          f"{_format_validation(exc)}") from exc

    problems = []
    try:
        get_theme(spec.theme)
    except KeyError as exc:
        problems.append(f"  theme: {exc.args[0]}")
    for beat in spec.beats:
        try:
            get_component(beat.component).Params.model_validate(beat.params)
        except KeyError as exc:
            problems.append(f"  {beat.id}.component: {exc.args[0]}")
        except ValidationError as exc:
            problems.append(f"  {beat.id} ({beat.component}) params:\n"
                            f"{_format_validation(exc, indent='    ')}")
    if problems:
        raise SpecInvalid(f"{path} is not a valid spec:\n" + "\n".join(problems))
    return spec


def carry_in_recipes(spec: VideoSpec) -> dict[str, tuple[ArtifactRecipe, ...]]:
    """Each beat's carried artifacts (SCENE_SPEC.md §6), as the probe and the
    render both need them. A carry-in the producing component cannot rebuild
    is a spec error (CarryInError), not a layout one."""
    try:
        return resolve_carry_in(spec)
    except CarryInError as exc:
        raise SpecInvalid(f"carry_in: {exc}") from exc


def check_layout(spec: VideoSpec) -> list[Report]:
    """Rung 3 (geometric). Runs before speech, so the probe uses the
    validator's placeholder duration: geometry does not depend on run time.
    Beats are probed with their carried artifacts on screen, as rendered."""
    reports = validate_specs(list(spec.beats), theme=spec.theme,
                             recipes=carry_in_recipes(spec))
    failed = [r for r in reports if not r.ok]
    if failed:
        raise LayoutRefused("\n".join(str(r) for r in failed))
    return reports


@contextmanager
def manim_scratch(work_dir: Path, verbose: bool = False) -> Iterator[None]:
    """Point every file Manim writes at `work_dir` and quiet its logging.

    Scoped with tempconfig rather than set globally; worker.render_beat opens
    its own tempconfig inside this one, which copies -- and so inherits --
    the logging settings. The worker sets media_dir and video_dir itself, from
    the work dir it is handed (`{work_dir}/manim`, the same media_dir as here,
    so Manim's text and LaTeX caches are shared with validation). Covers
    validation too: building Text writes Pango SVGs to `{media_dir}/texts`.
    """
    with tempconfig({
        "media_dir": str(Path(work_dir) / "manim"),
        "verbosity": "INFO" if verbose else "WARNING",
        "progress_bar": "display" if verbose else "none",
    }):
        yield


def validate(spec_path: Path, work_dir: Path = DEFAULT_WORK_DIR,
             verbose: bool = False) -> VideoSpec:
    """Rungs 1 and 3, with no speech and no render."""
    spec = load_spec(spec_path)
    with manim_scratch(work_dir, verbose):
        check_layout(spec)
    return spec


def render(
    spec_path: Path,
    quality: Quality = Quality.DRAFT,
    out: Path | None = None,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    work_dir: Path = DEFAULT_WORK_DIR,
    verbose: bool = False,
) -> RunResult:
    """Spec file -> finished MP4. Every stage is cached per beat (D-004), so a
    re-run after editing one narration line rebuilds only that beat."""
    work_dir = Path(work_dir)
    spec = validate(spec_path, work_dir, verbose)
    out = Path(out) if out else Path("out") / f"{spec.video_id}-{quality.value}.mp4"

    cache = Cache(cache_dir)
    ctx = BuildContext(quality=quality)
    video = Video.from_spec(spec)
    recipes = carry_in_recipes(spec)
    # The resolved voice (platform default backend, backend default voice) is
    # what tts.synthesize hashes, so the cached/synth report must use it too.
    try:
        voice = resolve_voice(spec.voice)
    except TTSError as exc:
        raise SpeechFailed(str(exc)) from exc

    # Speech for every beat before any render: render keys need durations.
    speech_cached = {}
    for beat in video.beats:
        speech_cached[beat.id] = cache.slot(
            "tts", tts_key(beat.spec.narration, voice), ".wav").exists
        try:
            synthesize_beat(beat, voice, cache)
        except Exception as exc:
            raise SpeechFailed(f"{beat.id}: {exc}") from exc

    result = RunResult(output=out)
    # The worker owns Manim scratch under this dir: one video dir per beat,
    # removed once the clip is committed to the cache.
    manim_dir = work_dir / "manim"
    beats_dir = cache.root / "beats"
    with manim_scratch(work_dir, verbose):
        for beat in video.beats:
            # The render key (font-resolved theme, tier, repair plan, carried
            # artifacts) is the worker's to build; a beat was cached iff the
            # clip the worker recorded on it was in the cache before the call.
            before = set(beats_dir.glob("*.mp4"))
            try:
                render_beat(beat, spec.theme, ctx, cache, manim_dir, recipes[beat.id])
            except Exception as exc:
                raise RenderFailed(f"{beat.id} ({beat.spec.component}): {exc}") from exc
            cached = beat.render_path in before

            outcome = BeatOutcome(beat.id, speech_cached[beat.id], cached,
                                  beat.duration)  # type: ignore[arg-type]
            result.beats.append(outcome)
            print(f"  {beat.id}  speech {'cached' if outcome.speech_cached else 'synth '}"
                  f"  render {'cached ' if cached else 'rebuilt'}"
                  f"  {outcome.duration:6.2f}s  {beat.spec.component}")

    print(f"  beats: {len(result.beats) - len(result.rebuilt)} cached, "
          f"{len(result.rebuilt)} rebuilt")

    assembly_dir = work_dir / "assemble" / f"{spec.video_id}-{quality.value}"
    try:
        assemble(video, out, assembly_dir, verbose=verbose)
    except Exception as exc:
        raise AssemblyFailed(str(exc)) from exc
    shutil.rmtree(assembly_dir, ignore_errors=True)
    return result
