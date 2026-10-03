"""Semantic validation (validation rung 2, SCENE_SPEC.md §8).

Cheap checks that need only the spec and the component's params -- no scene is
built, nothing is laid out. They run before the geometric probe because they
are cheaper, and because what they catch is not a layout problem: narration
too short for the animation, a carry-in nobody registered, more text than any
legible arrangement could hold. Mechanical repair cannot fix any of these, so
per ARCHITECTURE.md §9 they fail the beat with a specific message instead of
entering the repair loop.

Findings reuse the Report/Finding types from geometric.py so the repair loop
dispatches on one vocabulary across rungs.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping
from itertools import combinations
from pathlib import Path
from typing import Any

from chalkdust.core.models import BeatSpec, Region
from chalkdust.scenes.components import Component, make_component
from chalkdust.scenes.components.raw_scene import RawScene
from chalkdust.scenes.regions import MIN_FONT_SIZE, LayoutError, region_rect
from chalkdust.scenes.theme import DEFAULT, check_latex_source, math
from chalkdust.validate.geometric import Finding, Report, probe_media

# --- duration ---------------------------------------------------------------

# Narration pace used to estimate a beat's length before speech has run.
# Explainer narration sits around 140-160 wpm; we take the brisk end so the
# estimate errs short, which keeps the min_seconds() check honest -- a slow
# estimate would invent time the real audio will not have. Once the speech
# stage has run, pass the measured duration instead (D-002).
WORDS_PER_MINUTE = 160.0

# One idea per beat; past this the visual sits static too long
# (SCENE_SPEC.md §2, PRD.md §7). models.MAX_NARRATION_WORDS is only a sanity
# bound and defers the real limit to here.
MAX_BEAT_SECONDS = 25.0


def estimate_seconds(narration: str, wpm: float = WORDS_PER_MINUTE) -> float:
    """Spoken length of `narration` at `wpm`, before any audio exists."""
    return len(narration.split()) * 60.0 / wpm


def check_duration(component: Component, seconds: float,
                   measured: bool = False) -> list[Finding]:
    """Narration length against the component's animation budget."""
    source = "measured" if measured else f"estimated at {WORDS_PER_MINUTE:.0f} wpm"
    findings = []
    need = component.min_seconds()
    if seconds < need:
        findings.append(Finding(
            "duration",
            f"narration runs {seconds:.1f}s ({source}) but {component.name} "
            f"needs at least {need:.1f}s to animate these params without "
            f"rushing. Lengthen the narration or give the component fewer steps.",
        ))
    if seconds > MAX_BEAT_SECONDS:
        findings.append(Finding(
            "duration",
            f"narration runs {seconds:.1f}s ({source}); one beat carries one "
            f"idea in at most {MAX_BEAT_SECONDS:.0f}s. Split this beat.",
        ))
    return findings


# --- continuity -------------------------------------------------------------


def check_carry_in(carry_in: Iterable[str],
                   registered: Collection[str]) -> list[Finding]:
    """Every carry-in must name an artifact an earlier beat registered.

    The artifact registry itself lives with the compiler (SCENE_SPEC.md §6);
    this check takes only the set of names available at this beat, so it does
    not care how they were registered.
    """
    return [
        Finding(
            "carry_in",
            f"carry_in {name!r} is not a registered artifact; available: "
            f"{sorted(registered) or 'none'}",
        )
        for name in carry_in
        if name not in registered
    ]


def check_carried_targets(component: Component,
                          carry_in: Collection[str]) -> list[Finding]:
    """Every artifact the component acts on must be carried into its beat.

    A Callout whose `target_id` names something the beat does not carry in
    would raise CarryInError from inside build(); caught here it is refused
    by name, as a spec error, before anything is built.
    """
    return [
        Finding(
            "carry_in",
            f"{component.name} acts on {name!r}, which this beat does not "
            f"carry in; carry_in: {sorted(carry_in) or 'none'}. Add it to the "
            f"beat's carry_in or point at an artifact the beat carries.",
        )
        for name in component.carried_targets()
        if name not in carry_in
    ]


# --- regions ----------------------------------------------------------------


def check_region_conflicts(
    claims: Mapping[str, Iterable[Region]],
) -> list[Finding]:
    """No two simultaneously active claimants may claim intersecting regions.

    Compared by geometry, not by name: STAGE and STAGE_LEFT are different
    regions that occupy the same space (SCENE_SPEC.md §4).
    """
    findings = []
    for (a, a_regions), (b, b_regions) in combinations(claims.items(), 2):
        clashes = sorted(
            f"{ra.value}/{rb.value}"
            for ra in set(a_regions)
            for rb in set(b_regions)
            if region_rect(ra).intersects(region_rect(rb))
        )
        if clashes:
            findings.append(Finding(
                "region_conflict",
                f"{a} and {b} are active together but claim the same space: "
                f"{', '.join(clashes)}",
            ))
    return findings


# --- text volume ------------------------------------------------------------

# An upper bound on how many glyphs fit legibly in an area. Measured on this
# box with Manim 0.21 Text at font_size 22 (MIN_FONT_SIZE): line pitch is
# 0.298 units for every font tried (Manim's own line spacing), and the
# narrowest face tried (Times New Roman) averages 0.160 units per non-space
# glyph -- 0.54 of the pitch. We use 0.4 so the bound stays an upper bound for
# condensed faces too. The check is meant to catch only content that no
# arrangement could hold; anything subtler is the geometric probe's job.
LINE_PITCH_PER_FONT_SIZE = 0.298 / 22.0
GLYPH_WIDTH_PER_PITCH = 0.4


def text_volume(params: Any) -> int:
    """Non-whitespace characters across every string in the params.

    Counts all string values, including enum-like ones ("sequential") -- they
    are a rounding error against the generous capacity bound.
    """
    if isinstance(params, str):
        return sum(not c.isspace() for c in params)
    if isinstance(params, Mapping):
        return sum(text_volume(v) for v in params.values())
    if isinstance(params, (list, tuple)):
        return sum(text_volume(v) for v in params)
    return 0


def region_capacity(regions: Iterable[Region]) -> int:
    """Most glyphs the union of `regions` could hold at the legibility floor."""
    rects = [region_rect(r) for r in set(regions)]
    # Regions are either nested (STAGE holds both halves) or disjoint, so the
    # union is the sum of the rects no other claimed rect contains.
    outer = [r for r in rects if not any(o != r and o.contains(r) for o in rects)]
    area = sum(r.area for r in outer)
    pitch = MIN_FONT_SIZE * LINE_PITCH_PER_FONT_SIZE
    return int(area / (pitch * pitch * GLYPH_WIDTH_PER_PITCH))


def check_capacity(component: Component) -> list[Finding]:
    """Text volume against what the component's regions can hold legibly.

    Not applied to RawScene: its params are a rationale and Manim source,
    neither of which is drawn, so counting them measures code length, not
    text on screen -- a typical generated scene (~3k glyphs) would exceed the
    ~2.7k bound for the whole safe area. Its legibility is asserted after
    every play() inside its own render (SCENE_SPEC.md §7).
    """
    if isinstance(component, RawScene):
        return []
    volume = text_volume(component.params.model_dump(mode="json"))
    capacity = region_capacity(component.regions())
    if volume <= capacity:
        return []
    return [Finding(
        "capacity",
        f"{component.name} params carry {volume} characters of text but its "
        f"regions hold at most ~{capacity} at the minimum legible size. "
        f"No layout can fit this; split the beat.",
    )]


# --- LaTeX ------------------------------------------------------------------


def check_latex(component: Component,
                media_dir: Path | str | None = None) -> list[Finding]:
    """Compile each of the component's LaTeX strings standalone.

    First as written, through theme.check_latex_source: MathTex repairs
    unbalanced braces and an unpaired \\left before compiling, so only a
    compile of the raw source sees them (an empty-denominator \\frac{1}{
    would otherwise pass here and render). Then through theme.math, the
    constructor build() uses, so the expression is
    compiled exactly as the render will compile it, a success lands in
    Manim's Tex cache for the real build, and a failure is the same refusal
    the geometric rung would raise: kind "invalid_latex"
    (theme.refuse_invalid_latex). One vocabulary across rungs, so the repair
    loop dispatches on one kind whichever rung caught the bad LaTeX.

    Compiling writes .tex/.svg files under Manim's media_dir. They go where
    the geometric probe's go (geometric.probe_media): `media_dir` when given,
    else a caller's own redirect (the pipeline's work dir), else
    work/manim -- never ./media in the cwd.

    A failure that is not the expression's fault (the TeX toolchain itself
    failing) is a build_error, not invalid_latex: regenerating the spec
    cannot fix it.
    """
    findings = []
    with probe_media(media_dir):
        for i, source in enumerate(component.latex_strings()):
            what = f"{component.name} latex_strings()[{i}]"
            try:
                check_latex_source(source, what=what)
                math(source, DEFAULT, what=what)
            except LayoutError as exc:
                findings.append(Finding(exc.kind, str(exc)))
            except Exception as exc:
                first_line = (str(exc).strip().splitlines() or [""])[0]
                findings.append(Finding(
                    "build_error",
                    f"{component.name} latex_strings()[{i}] {source!r} could "
                    f"not be compiled ({type(exc).__name__}: {first_line})",
                ))
    return findings


# --- entry point ------------------------------------------------------------


def validate_semantic(
    spec: BeatSpec,
    registered_artifacts: Collection[str] = (),
    duration: float | None = None,
    concurrent: Mapping[str, Iterable[Region]] | None = None,
    media_dir: Path | str | None = None,
) -> Report:
    """Run every rung-2 check on one beat. Never raises.

    registered_artifacts -- names registered by earlier beats (SCENE_SPEC.md §6).
    duration             -- measured audio length; estimated from the
                            narration when the speech stage has not run.
    concurrent           -- regions claimed by anything else on screen during
                            this beat, by name. Today one component owns a
                            beat, so this is empty unless the compiler layers
                            something over it.
    media_dir            -- where compiling LaTeX writes its scratch (see
                            check_latex); never the cwd.
    """
    report = Report(beat_id=spec.id)
    try:
        component = make_component(spec.component, spec.params)
    except Exception as exc:
        report.findings.append(Finding("build_error", f"{type(exc).__name__}: {exc}"))
        return report

    measured = duration is not None
    seconds = duration if measured else estimate_seconds(spec.narration)
    report.findings += check_duration(component, seconds, measured=measured)
    report.findings += check_carry_in(spec.carry_in, registered_artifacts)
    report.findings += check_carried_targets(component, spec.carry_in)
    report.findings += check_region_conflicts(
        {spec.component: component.regions(), **(concurrent or {})}
    )
    report.findings += check_capacity(component)
    report.findings += check_latex(component, media_dir)
    return report
