"""Visual identity: palette, typography, and text constructors.

Components never specify colours or font sizes directly -- they ask the theme
for a role ("heading", "accent"). That is what lets a channel's visual identity
be swapped by config, and it is why `theme` is a cache-key input (D-004).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from manim import MathTex, Mobject, Tex, Text, VMobject, config
from manim.utils.tex import TexTemplate
from manim.utils.tex_file_writing import make_tex_compilation_command

from chalkdust.core.cache import content_hash
from chalkdust.scenes.regions import (
    INVALID_LATEX,
    UNRENDERABLE_TEXT,
    LayoutError,
    tag_font_size,
)


@dataclass(frozen=True)
class Palette:
    bg: str
    fg: str          # primary text
    muted: str       # secondary text, labels
    accent: str      # the one colour that means "look here"
    accent_alt: str  # second emphasis, used sparingly
    success: str
    danger: str


@dataclass(frozen=True)
class Typography:
    """Font families plus sizes in Manim font_size units.

    If a font is not installed, Pango substitutes silently -- your render will
    succeed and look wrong. `check_fonts()` below surfaces that early.
    """

    heading_font: str
    body_font: str
    mono_font: str

    title: float = 60.0
    heading: float = 44.0
    body: float = 32.0
    caption: float = 24.0
    mono: float = 28.0


@dataclass(frozen=True)
class Theme:
    name: str
    palette: Palette
    type: Typography
    # Motion language: how fast things move in this channel's videos.
    fade_time: float = 0.4
    write_time: float = 0.8


DEFAULT = Theme(
    name="default",
    palette=Palette(
        bg="#0E1116",
        fg="#E6EDF3",
        muted="#8B949E",
        accent="#58A6FF",
        accent_alt="#F0883E",
        success="#3FB950",
        danger="#F85149",
    ),
    type=Typography(
        heading_font="Archivo",
        body_font="Inter",
        mono_font="JetBrains Mono",
    ),
)

THEMES: dict[str, Theme] = {"default": DEFAULT}


def get_theme(name: str) -> Theme:
    if name not in THEMES:
        raise KeyError(f"unknown theme {name!r}; available: {sorted(THEMES)}")
    return THEMES[name]


def check_fonts(theme: Theme) -> list[str]:
    """Return theme fonts that are not installed.

    Call this once at startup. A missing font does not raise -- Pango falls
    back -- so this is the only way to notice before the render looks wrong.
    """
    available = _installed_fonts()
    if not available:
        return []  # can't check; don't block the render
    wanted = {theme.type.heading_font, theme.type.body_font, theme.type.mono_font}
    return sorted(f for f in wanted if f not in available)


# --- text constructors ------------------------------------------------------
# Every text mobject in the system should come from one of these. They apply
# the theme and tag the font size so the legibility check in regions.py works,
# and they refuse text the resolved font cannot draw (check_renderable), so no
# component lays out an inkless or tofu-filled Text (SCENE_SPEC.md §11 rule 1).


def title_text(s: str, theme: Theme, color: str | None = None, *,
               what: str = "text") -> Text:
    return _text(s, theme.type.heading_font, theme.type.title,
                 color or theme.palette.fg, "BOLD", what)


def heading_text(s: str, theme: Theme, color: str | None = None, *,
                 what: str = "text") -> Text:
    return _text(s, theme.type.heading_font, theme.type.heading,
                 color or theme.palette.fg, "SEMIBOLD", what)


def body_text(s: str, theme: Theme, color: str | None = None, *,
              what: str = "text") -> Text:
    return _text(s, theme.type.body_font, theme.type.body,
                 color or theme.palette.fg, "NORMAL", what)


def caption_text(s: str, theme: Theme, color: str | None = None, *,
                 what: str = "text") -> Text:
    return _text(s, theme.type.body_font, theme.type.caption,
                 color or theme.palette.muted, "NORMAL", what)


def mono_text(s: str, theme: Theme, color: str | None = None, *,
              what: str = "text") -> Text:
    return _text(s, theme.type.mono_font, theme.type.mono,
                 color or theme.palette.fg, "NORMAL", what)


def _text(s: str, font: str, size: float, color: str, weight: str,
          what: str) -> Text:
    """Build one themed Text, or refuse it as kind "unrenderable_text".

    Three guards, each for a way Pango turns content into something that is
    not the content, and each grounded in what Manim actually does rather
    than in a character count -- a glyph count is not a character count
    once ligatures merge "fi" into one glyph, or combining marks compose:

      1. check_renderable first: every distinct character the font cannot
         draw by itself refuses, before anything is built or laid out.
      2. Manim's own "rendered fewer glyph(s)" ValueError (raised only when
         glyphs are mapped back to characters) becomes the typed refusal
         instead of a build_error.
      3. A built Text with no points at all refuses: it is a zero-size
         mobject that crashed layouts (no centre to align on, a zero width
         to divide by) or sat in the frame as a silent gap. This catches what
         the per-character probe skips on purpose: a string made only of
         format characters (U+200B, U+2060, a soft hyphen).
    """
    check_renderable(s, font, weight=weight, what=what)
    try:
        t = Text(s, font=font, font_size=size, color=color, weight=weight)
    except ValueError as exc:
        if "rendered fewer glyph" not in str(exc):
            raise
        raise LayoutError(
            f"{what} {s!r} cannot be drawn in font {font!r}: it shapes to "
            f"fewer glyphs than it has characters. Rewrite it with characters "
            f"that font draws.", kind=UNRENDERABLE_TEXT) from exc
    if not any(len(m.points) for m in t.get_family()):
        raise LayoutError(
            f"{what} {s!r} draws nothing in font {font!r}: it has no character "
            f"that font can draw (only spaces, zero-width or format "
            f"characters, or a script the font lacks). Write the text the "
            f"viewer should read.", kind=UNRENDERABLE_TEXT)
    return tag_font_size(t, size)


# --- glyph coverage -----------------------------------------------------------
# Pango never fails on a character the font lacks. Depending on the character
# it draws nothing (right-to-left letters and emoji under the fallback fonts
# on Windows -- measured: Hebrew letters and U+1F600 in Arial shape to no
# glyph) or a missing-glyph box with the code point in hex (CJK and
# private-use in Arial or Courier New). Either way the render "succeeds" and is wrong, and an
# inkless label crashed several layouts. Asking the font about each character
# alone is the one probe that does not depend on ligatures or shaping context.

# Characters the per-character probe does not ask about, by Unicode category:
# combining marks (M*) draw only on a base -- alone, Pango adds a dotted
# circle, which would read as two glyphs -- and format characters (Cf: ZWNJ,
# ZWJ, bidi marks, soft hyphen) are legitimately invisible. Whitespace is
# skipped by str.isspace(). A string made ONLY of these draws nothing, which
# _text's no-points guard refuses.
_UNPROBED_CATEGORIES = ("Mn", "Mc", "Me", "Cf")


@lru_cache(maxsize=8192)
def _glyph_paths(ch: str, font: str, weight: str) -> int:
    """How many paths Pango draws for `ch` alone in `font`; -1 if it refuses.

    Manim gives every shaped glyph its own submobject. One means the font
    drew the character; zero means nothing was drawn; a missing-glyph box
    is the box plus one path per hex digit (5 for a BMP code point). Cached
    per process: coverage is a property of the installed font, and a font
    installed mid-run would not reach the render key until the next process
    anyway (see _installed_fonts).
    """
    try:
        return len(Text(ch, font=font, weight=weight).submobjects)
    except Exception:
        # NUL, a lone surrogate: Pango or the encoder refuses the string.
        return -1


def unsupported_characters(text: str, font: str, *,
                           weight: str = "NORMAL") -> list[tuple[str, str]]:
    """Distinct characters of `text` that `font` cannot draw, with why.

    In order of first appearance. A character passes when it draws as exactly
    one path, or as no more paths than its canonical decomposition has
    characters (HarfBuzz may draw a precomposed letter the font lacks as base
    plus mark). Whitespace, combining marks and format characters are not
    probed (see _UNPROBED_CATEGORIES).
    """
    out: list[tuple[str, str]] = []
    for ch in dict.fromkeys(text):
        if ch.isspace() or unicodedata.category(ch) in _UNPROBED_CATEGORIES:
            continue
        n = _glyph_paths(ch, font, weight)
        if n == 1 or 1 < n <= len(unicodedata.normalize("NFD", ch)):
            continue
        if n < 0:
            why = "cannot be shaped at all"
        elif n == 0 and unicodedata.bidirectional(ch) in ("R", "AL"):
            why = "draws nothing (a right-to-left letter this font does not shape)"
        elif n == 0:
            why = "draws nothing"
        else:
            why = "draws a missing-glyph box"
        out.append((ch, why))
    return out


def check_renderable(text: str, font: str, *, weight: str = "NORMAL",
                     what: str = "text") -> None:
    """Refuse `text` as kind "unrenderable_text" if `font` cannot draw it.

    The reusable form of the theme's glyph guard. The text constructors call
    it on everything they build; a component that builds text some other way
    (CodeWalk through Manim's Code) calls it on that text with the resolved
    font it passes along, e.g. check_renderable(source, theme.type.mono_font).
    The message names the resolved font, because the fix is either the text
    or the font installed on the render machine -- right-to-left scripts in
    particular draw nothing with the fallback fonts and are refused, not
    reshaped.
    """
    bad = unsupported_characters(text, font, weight=weight)
    if not bad:
        return
    shown = "; ".join(f"{_describe(ch)} {why}" for ch, why in bad[:6])
    more = f"; and {len(bad) - 6} more" if len(bad) > 6 else ""
    raise LayoutError(
        f"{what} {text!r} cannot be drawn in font {font!r}: {shown}{more}. "
        f"Rewrite it with characters that font draws.",
        kind=UNRENDERABLE_TEXT)


def _describe(ch: str) -> str:
    name = unicodedata.name(ch, "")
    return f"U+{ord(ch):04X}" + (f" {name}" if name else "")


# --- maths ------------------------------------------------------------------

# The environment theme.math compiles in. MathTex's own default, named and
# passed explicitly so the standalone check below and the real build cannot
# drift apart: a check in plain math mode would refuse every aligned
# derivation that uses &.
MATH_ENVIRONMENT = "align*"

# How long one LaTeX run may take. A cold MiKTeX compile that downloads
# packages measured ~23 s on this project's machines, but under parallel load
# a valid label has taken longer than 120 s (register G4b-N6), so running past
# it says nothing about the expression: the run is retried once
# (LATEX_CHECK_ATTEMPTS), and a second timeout raises LatexToolchainError --
# never invalid_latex, and never cached. A macro that never terminates
# (`\def\a{\a}\a`) still ends there, named by its source.
LATEX_CHECK_TIMEOUT = 120.0
LATEX_CHECK_ATTEMPTS = 2


class LatexToolchainError(RuntimeError):
    """The TeX toolchain did not finish a compile in time, on any attempt.

    A toolchain condition, not a verdict on the spec: the machine is loaded
    or TeX is hung. Typed so the rungs report it as kind "toolchain" and the
    pipeline as a toolchain failure with its own exit code, rather than as a
    refusal the repair loop would answer by rewriting valid maths.
    """


def check_latex_source(source: str, *, what: str = "maths",
                tex_template: TexTemplate | None = None,
                environment: str = MATH_ENVIRONMENT) -> None:
    """Compile `source` exactly as written; refuse it as "invalid_latex" if
    LaTeX reports an error.

    Why a second compile: MathTex rewrites its source before compiling it.
    It closes unbalanced braces (`\\frac{1}{` draws a fraction with an empty
    denominator), turns an unpaired `\\left(` into `\\big(`, and wraps a stray
    `}}` so that `}} = 1` draws "= 1". Each of those compiles, renders, and
    is not what the spec says. Compiling the raw source with the same
    TexTemplate and the same environment MathTex uses (align*) is the one
    check that sees the source as the author wrote it. Used by theme.math at
    build and by the semantic rung (validate.semantic.check_latex), so both
    rungs refuse the same strings with the same kind.

    Cached by hash(source, template, environment), in memory and on disk
    under `{media_dir}/latex/` -- media_dir is always the run's work dir
    (pipeline.manim_scratch, the render worker, geometric.probe_media), so
    the cache lives with Manim's own Tex cache, never in the cwd. A verdict
    is cached only when it is the expression's: a toolchain failure or a
    timeout is retried next time.

    A failure that is not the expression's fault -- no `latex` on PATH, or
    a template that does not compile even around "x" -- raises RuntimeError,
    which the rungs report as build_error: regenerating the spec cannot fix
    it. A compile still running after LATEX_CHECK_TIMEOUT on every attempt
    raises LatexToolchainError, which the rungs report as kind "toolchain".
    """
    template = tex_template or config["tex_template"]
    key = content_hash("latex-check", source, template.body,
                       template.tex_compiler, template.output_format,
                       environment)
    verdict = _latex_verdicts.get(key)
    record = _latex_dir() / f"{key}.json"
    if verdict is None and record.exists():
        try:
            verdict = json.loads(record.read_text(encoding="utf-8"))["error"]
            _latex_verdicts[key] = verdict
        except (OSError, ValueError, KeyError):
            verdict = None  # unreadable record: compile again
    if verdict is None and key not in _latex_verdicts:
        verdict = _compile_raw(source, key, template, environment, what)
        _latex_verdicts[key] = verdict
        try:
            tmp = record.with_name(f".tmp-{os.getpid()}-{record.name}")
            tmp.write_text(json.dumps({"error": verdict}), encoding="utf-8")
            os.replace(tmp, record)
        except OSError:
            pass  # a cache write failing must not fail the check
    if verdict:
        raise refuse_invalid_latex(what, source, "does not compile",
                                   RuntimeError(verdict))


# key -> None (compiles) or LaTeX's first error line. Per process, in front of
# the on-disk records.
_latex_verdicts: dict[str, str | None] = {}


def _latex_dir() -> Path:
    d = Path(config.media_dir) / "latex"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _compile_raw(source: str, key: str, template: TexTemplate,
                 environment: str, what: str) -> str | None:
    """Run the template's compiler once on the raw source.

    Returns None when it compiles, else LaTeX's first error line (the
    expression's fault). Raises LatexToolchainError when LaTeX times out on
    every attempt and RuntimeError when the toolchain is otherwise at fault;
    neither is cached.
    """
    error = _run_latex(template.get_texcode_for_expression_in_env(source, environment),
                       key, template, what, source)
    if error is not None and _run_latex(
            template.get_texcode_for_expression_in_env("x", environment),
            content_hash("latex-baseline", template.body, environment),
            template, "the TeX template", "x") is not None:
        raise RuntimeError(
            f"the TeX toolchain cannot compile its own template around 'x' "
            f"({error}); check the LaTeX install, not the spec")
    return error


def _compile(command: list[str]) -> subprocess.CompletedProcess:
    """One TeX run, bounded by LATEX_CHECK_TIMEOUT. Its own function so a
    test can stand in for a slow toolchain without a slow test."""
    return subprocess.run(command, stdin=subprocess.DEVNULL,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                          timeout=LATEX_CHECK_TIMEOUT)


def _run_latex(texcode: str, key: str, template: TexTemplate, what: str,
               source: str) -> str | None:
    compiler = template.tex_compiler
    if not isinstance(compiler, str):
        compiler = compiler[0]
    for attempt in range(1, LATEX_CHECK_ATTEMPTS + 1):
        # A private scratch dir per run: two processes checking the same
        # source at once must not share a .log or .dvi (Windows locks open
        # files). Per attempt too: a timed-out run's TeX may still hold its
        # files, and a retry tripping over them would read as a LaTeX error.
        work = _latex_dir() / f".run-{key}-{os.getpid()}-{attempt}"
        work.mkdir(parents=True, exist_ok=True)
        tex_file = work / f"{key}.tex"
        tex_file.write_text(texcode, encoding="utf-8")
        command = make_tex_compilation_command(
            compiler, template.output_format, tex_file, work)
        try:
            cp = _compile(command)
            break
        except FileNotFoundError as exc:
            shutil.rmtree(work, ignore_errors=True)
            raise RuntimeError(f"{compiler!r} is not on PATH; LaTeX cannot be "
                               f"checked or rendered") from exc
        except subprocess.TimeoutExpired as exc:
            shutil.rmtree(work, ignore_errors=True)
            if attempt < LATEX_CHECK_ATTEMPTS:
                continue  # slow is not wrong: run it again before judging
            raise LatexToolchainError(
                f"the TeX toolchain did not finish compiling {what} {source!r} "
                f"within {LATEX_CHECK_TIMEOUT:.0f} s, {LATEX_CHECK_ATTEMPTS} "
                f"times: LaTeX is slow or hung (a heavily loaded machine, or a "
                f"macro that never terminates). This is not a verdict on the "
                f"spec; run again on a quieter machine, and if it persists, "
                f"look for a self-referencing macro in that source") from exc
    error = None
    if cp.returncode != 0:
        error = _first_tex_error(tex_file.with_suffix(".log"))
    shutil.rmtree(work, ignore_errors=True)
    return error


def _first_tex_error(log: Path) -> str:
    """LaTeX's first '! ...' line, with the 'l.N ...' context if present."""
    try:
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return "LaTeX failed and wrote no log"
    for i, line in enumerate(lines):
        if line.startswith("! "):
            where = next((ln.strip() for ln in lines[i + 1:i + 12]
                          if ln.startswith("l.")), "")
            return f"{line[2:].strip()}" + (f" at {where}" if where else "")
    return "LaTeX failed without an error line"


def math(s: str, theme: Theme, size: float | None = None,
         color: str | None = None, *, what: str = "maths") -> MathTex:
    """LaTeX maths. Note MathTex font_size behaves differently from Text --
    the same numeric value renders visually smaller, so we bump it.

    Bad LaTeX refuses as LayoutError kind "invalid_latex" (see
    refuse_invalid_latex); `what` names the content in that message, e.g.
    "step[2]", so the refusal points at the spec field to regenerate. The
    raw source is compiled first (check_latex_source), because MathTex silently
    repairs unbalanced braces and an unpaired \\left before compiling.
    """
    size = size or theme.type.heading
    check_latex_source(s, what=what)
    try:
        m = MathTex(s, font_size=size * 1.2, color=color or theme.palette.fg,
                    tex_environment=MATH_ENVIRONMENT)
    except ValueError as exc:
        raise refuse_invalid_latex(what, s, "does not compile", exc) from exc
    if m.width == 0 and m.height == 0:
        # Compiles, draws nothing (e.g. only spacing commands). Placing it
        # would leave a silent gap where the viewer expects maths.
        raise refuse_invalid_latex(what, s, "renders nothing")
    return tag_font_size(m, size)


def refuse_invalid_latex(what: str, source: str, reason: str,
                         cause: Exception | None = None) -> LayoutError:
    """The one way scene code refuses bad LaTeX: kind "invalid_latex".

    Manim reports a LaTeX failure as a bare ValueError from deep inside its
    compile step. Typing it here gives the repair loop a refusal it can
    dispatch on (regenerate the spec, SCENE_SPEC.md §9) instead of a crash,
    and the probe reports it as a finding of that kind rather than as a
    build_error. Every maths constructor routes its failures through this.
    """
    detail = ""
    if cause is not None:
        first = (str(cause).strip().splitlines() or [""])[0]
        detail = f" ({first})" if first else ""
    return LayoutError(f"{what} is not valid LaTeX, {reason}: {source!r}{detail}",
                       kind=INVALID_LATEX)


def emphasize(mob: VMobject, theme: Theme) -> VMobject:
    """Recolour to the accent. The one gesture that means 'this matters'."""
    return mob.set_color(theme.palette.accent)

# --- font resolution --------------------------------------------------------
# Pango substitutes missing fonts silently, so a render succeeds and looks
# wrong. Resolving explicitly at startup makes the substitution loud.

FALLBACKS = {
    "heading": ["Archivo", "Helvetica Neue", "Arial"],
    "body": ["Inter", "Helvetica Neue", "Arial"],
    "mono": ["JetBrains Mono", "Menlo", "Courier New"],
}


# Substitutions already announced in this process, as (role, wanted, used).
_WARNED: set[tuple[str, str, str]] = set()


@lru_cache(maxsize=1)
def _installed_fonts() -> frozenset[str]:
    """The system's font families, enumerated once per process.

    resolve_fonts runs on every scene construction -- the validation probe,
    the repair probe and the render of every beat -- and enumerating fonts
    costs ~0.3 s a call on Windows. Fonts installed mid-run are not seen;
    a run is minutes long and the render key would not notice them anyway
    until the next process.
    """
    try:
        import manimpango

        return frozenset(manimpango.list_fonts())
    except Exception:
        return frozenset()


def resolve_fonts(theme: Theme, warn: bool = True) -> Theme:
    """Return a copy of `theme` with any missing font swapped for a fallback.

    Called on every scene construction -- every probe, repair attempt and
    render -- so each substitution is printed once per process, not once per
    scene: a validation pass over a video would otherwise repeat the same
    line hundreds of times and bury everything else. If nothing in the
    fallback chain is installed we leave the original name and let Pango
    decide -- but by then the warning has already been printed.
    """
    available = _installed_fonts()
    if not available:
        return theme

    def pick(role: str, wanted: str) -> str:
        if wanted in available:
            return wanted
        for candidate in FALLBACKS[role]:
            if candidate in available:
                if warn and (role, wanted, candidate) not in _WARNED:
                    _WARNED.add((role, wanted, candidate))
                    print(f"[theme] {wanted!r} missing, using {candidate!r} for {role}")
                return candidate
        return wanted

    t = theme.type
    return Theme(
        name=theme.name,
        palette=theme.palette,
        type=Typography(
            heading_font=pick("heading", t.heading_font),
            body_font=pick("body", t.body_font),
            mono_font=pick("mono", t.mono_font),
            title=t.title,
            heading=t.heading,
            body=t.body,
            caption=t.caption,
            mono=t.mono,
        ),
        fade_time=theme.fade_time,
        write_time=theme.write_time,
    )


# --- font metrics -----------------------------------------------------------
# Manim bounding boxes include descenders, so they vary with the letters in a
# string. Anything aligned to them drifts row to row. Cap height is a property
# of the font at a given size, so it is a stable layout unit.


@lru_cache(maxsize=64)
def _cap_height(font: str, size: float) -> float:
    """Height of a capital H -- i.e. cap height -- in Manim units."""
    return float(Text("H", font=font, font_size=size).height)


def body_cap_height(theme: Theme) -> float:
    return _cap_height(theme.type.body_font, theme.type.body)
