"""Phase 0 pipeline: one hand-written spec file in, one finished MP4 out.

Stage order is the architecture, not a convenience (ARCHITECTURE.md §1-2):

  load      -- JSON -> VideoSpec, plus every beat's component params
  validate  -- schema, semantic, then geometric (SCENE_SPEC.md §8 rungs 1-3),
               BEFORE any speech or render. Fail early: a spec that cannot
               lay out must not cost a TTS call.
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
from chalkdust.core.models import (
    BeatSpec,
    BuildContext,
    CarryInError,
    Quality,
    Video,
    VideoSpec,
)
from chalkdust.render.assemble import assemble
from chalkdust.render.worker import render_beat
from chalkdust.scenes.components import get_component
from chalkdust.scenes.components.raw_scene import (
    USAGE_LOG_NAME,
    RawScene,
    RawSceneError,
    RawSceneParams,
    check_code,
    degrade_spec,
)
from chalkdust.scenes.theme import get_theme
from chalkdust.speech.base import TTSError
from chalkdust.speech.tts import resolve_voice, synthesize_beat
from chalkdust.validate.geometric import Report, validate_specs
from chalkdust.validate.semantic import validate_semantic

DEFAULT_CACHE_DIR = Path(".cache")
# Manim scratch, assembly intermediates. Git-ignored; never cwd/media.
DEFAULT_WORK_DIR = Path("work")


# --- failures ---------------------------------------------------------------


class PipelineError(Exception):
    """A known failure, already phrased for the operator."""


class SpecInvalid(PipelineError):
    """Rung 1: the file is not a valid spec. Nothing ran."""


class SemanticRefused(PipelineError):
    """Rung 2: content no layout can carry -- narration too short for the
    animation or too long for one beat, more text than the regions hold, LaTeX
    that does not compile. Nothing was synthesised."""


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
    # A RawScene beat that fell back to BulletReveal (D-010, SCENE_SPEC.md §7).
    degraded: bool = False


@dataclass
class RunResult:
    output: Path
    beats: list[BeatOutcome] = field(default_factory=list)

    @property
    def rebuilt(self) -> list[str]:
        return [b.beat_id for b in self.beats if not b.render_cached]

    @property
    def degraded(self) -> list[str]:
        return [b.beat_id for b in self.beats if b.degraded]


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


@dataclass(frozen=True)
class CheckedBeat:
    """What rungs 2 and 3 check for one beat: the spec the worker will build.

    For a library component that is the beat as written. A RawScene beat is
    routed as the worker routes it (worker.render_beat -> render_raw_beat):
    it never builds in a host scene -- RawScene.build() raises -- so the
    geometric probe cannot run on it; its layout is asserted after every
    play() inside its own out-of-process render (SCENE_SPEC.md §7). When it
    is statically certain to degrade, what renders is the BulletReveal
    fallback, and that is what gets checked.
    """

    spec: BeatSpec
    geometric: bool
    note: str | None = None


def raw_scene_refusal(beat: BeatSpec) -> RawSceneError | None:
    """Why render_raw_beat will certainly degrade this RawScene beat, decided
    without running it; None when only the render can tell. The same checks
    in the same order as render_raw_beat: a carry-in, then the static
    allowlist (check_code). Params already passed rung 1."""
    if beat.carry_in:
        return RawSceneError(
            f"carries in {beat.carry_in}; RawScene renders out of process and "
            "cannot draw carried artifacts", kind="carry_in")
    try:
        check_code(RawSceneParams.model_validate(beat.params).code)
    except RawSceneError as exc:
        return exc
    return None


def checked_beats(spec: VideoSpec) -> list[CheckedBeat]:
    """Every beat, as rungs 2 and 3 should see it (CheckedBeat)."""
    checked = []
    for beat in spec.beats:
        if beat.component != RawScene.name:
            checked.append(CheckedBeat(beat, geometric=True))
            continue
        problem = raw_scene_refusal(beat)
        if problem is None:
            checked.append(CheckedBeat(beat, geometric=False, note=(
                "RawScene: layout is asserted in its own render, not probed here")))
        else:
            # A degraded beat is always better than a failed video (D-010):
            # say so now, and check the fallback that will actually render.
            checked.append(CheckedBeat(degrade_spec(beat), geometric=True, note=(
                f"RawScene will degrade to BulletReveal ({problem.kind}: "
                f"{problem}); checking the fallback")))
    return checked


def check_semantics(spec: VideoSpec,
                    checked: list[CheckedBeat] | None = None) -> list[Report]:
    """Rung 2 (semantic). Cheap, and before the geometric probe because what
    it catches is not a layout problem: no mechanical repair fixes it
    (ARCHITECTURE.md §9).

    Runs before speech, so durations are estimated from the narration
    (semantic.estimate_seconds); each beat sees the artifacts registered by
    the beats before it."""
    checked = checked if checked is not None else checked_beats(spec)
    registered: set[str] = set()
    reports = []
    for beat, check in zip(spec.beats, checked, strict=True):
        reports.append(validate_semantic(check.spec, registered))
        if beat.registers is not None:
            registered.add(beat.registers)
    failed = [r for r in reports if not r.ok]
    if failed:
        raise SemanticRefused("\n".join(str(r) for r in failed))
    return reports


def check_layout(spec: VideoSpec,
                 checked: list[CheckedBeat] | None = None) -> list[Report]:
    """Rung 3 (geometric). Runs before speech, so the probe uses the
    validator's placeholder duration: geometry does not depend on run time.
    Beats are probed with their carried artifacts on screen, as rendered;
    a RawScene beat is probed only as its fallback (CheckedBeat)."""
    checked = checked if checked is not None else checked_beats(spec)
    probed = [c.spec for c in checked if c.geometric]
    # A degraded fallback drops its carry_in, so it is probed carrying nothing.
    recipes = carry_in_recipes(spec)
    reports = validate_specs(
        probed, theme=spec.theme,
        recipes={s.id: recipes.get(s.id, ()) for s in probed if s.carry_in})
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
    """Rungs 1, 2 and 3, with no speech and no render."""
    spec = load_spec(spec_path)
    carry_in_recipes(spec)  # an unrebuildable carry-in is a spec error
    checked = checked_beats(spec)
    for check in checked:
        if check.note:
            print(f"  {check.spec.id}  {check.note}")
    # The semantic rung compiles LaTeX, which writes under Manim's media dir.
    with manim_scratch(work_dir, verbose):
        check_semantics(spec, checked)
        check_layout(spec, checked)
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
                                  beat.duration,  # type: ignore[arg-type]
                                  degraded=beat.degraded)
            result.beats.append(outcome)
            shown = beat.spec.component
            if beat.degraded:
                # The reason is in the RawScene usage log, one line per call.
                shown += (" -> DEGRADED to BulletReveal (reason in "
                          f"{manim_dir / USAGE_LOG_NAME})")
            print(f"  {beat.id}  speech {'cached' if outcome.speech_cached else 'synth '}"
                  f"  render {'cached ' if cached else 'rebuilt'}"
                  f"  {outcome.duration:6.2f}s  {shown}")

    summary = (f"  beats: {len(result.beats) - len(result.rebuilt)} cached, "
               f"{len(result.rebuilt)} rebuilt")
    if result.degraded:
        summary += f", {len(result.degraded)} degraded ({', '.join(result.degraded)})"
    print(summary)

    assembly_dir = work_dir / "assemble" / f"{spec.video_id}-{quality.value}"
    try:
        assemble(video, out, assembly_dir, verbose=verbose)
    except Exception as exc:
        raise AssemblyFailed(str(exc)) from exc
    shutil.rmtree(assembly_dir, ignore_errors=True)
    return result
