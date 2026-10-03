# CLAUDE.md

Guide for Claude sessions working in this repo. Short on purpose: the docs in `docs/`
are the law, and this file points at them rather than restating them.

## What ChalkDust is

A Python pipeline that turns a declarative JSON spec into a Manim explainer video with a
synced voiceover. Three ideas hold it up:

- **The LLM never writes Manim.** It picks hand-written, layout-safe components and
  fills in their params; a deterministic compiler builds the scene (D-001,
  SCENE_SPEC.md §1).
- **Speech first, measured, animation stretches to fit.** TTS runs as its own cached
  stage, `ffprobe` measures the audio, and that duration drives every run time
  (D-002, D-003).
- **Every artifact is cached by the hash of its inputs.** Editing one beat re-renders
  one beat (D-004). Beats render independently; continuity across beats is only via
  `carry_in` (D-005).

## Read these first, in this order

1. `docs/SCENE_SPEC.md`: the spec contract, regions, component catalog, design rules (§11)
2. `docs/ARCHITECTURE.md`: stages, caching, module layout
3. `docs/ROADMAP.md`: phases and what is deliberately not built yet
4. `docs/DECISIONS.md`: D-001 to D-012, each with "revisit if"
5. `docs/PRD.md`: goals, non-goals, the definition of a good video
6. `docs/OPERATIONS.md`: channels, publishing, quota, policy

Don't change a decision in these docs on your own. If you disagree, say so and give a
recommendation; the owner decides.

## Layout rules (non-negotiable)

- **No absolute positioning.** Place content with `fit_to_region(mob, Region.X)` or
  relative to another mobject (`scenes/regions.py`, SCENE_SPEC.md §4). Regions:
  `TITLE_BAR`, `STAGE` (= `STAGE_LEFT` + `STAGE_RIGHT`), `LOWER_THIRD`, all inside the
  safe area.
- **Theme-only styling.** Build text with the `scenes/theme.py` constructors
  (`title_text`, `heading_text`, `body_text`, `caption_text`, `mono_text`, `math`) and
  colour with palette roles. A component never takes a coordinate, colour or font size
  from the spec (SCENE_SPEC.md §3, §11).
- **Derived timing.** Split `scene.beat_duration` with `scene.budget(*weights)`. Never
  hardcode a run time (D-002, SCENE_SPEC.md §11 rule 4).
- **Render correctly or raise.** Too much content is a `LayoutError` (`overflow`,
  `illegible`, `overlap`, `out_of_bounds`), never shrunken or off-screen text. The fix
  for overflow is splitting the beat. Call `scene.settle()` at points a viewer will
  look at; mark things that must not overlap with `scene.exclusive(...)`.

## Adding a component

Subclass `Component` in `chalkdust/scenes/components/<name>.py`: set `name` and
`Params` (a `ComponentParams` subclass, which is frozen and `extra="forbid"`),
implement `regions()` and `build()`, provide `examples()` (must validate clean) and
`stress()` (about 3x realistic volume; must fit or refuse with a `LayoutError`), decorate
with `@register`, and import the module in `components/__init__.py`.
`tests/test_layout.py` walks the registry, so a registered component gets layout tests
with no new test file.

## Cache-key discipline (D-004)

- Keys come from `core/cache.py`: `tts_key(narration, voice)` and
  `beat_render_key(spec, duration, ctx)`. Every input that changes an artifact must be
  in its key, and nothing else. A missing input serves stale files, the worst bug class
  here.
- Version strings travel in `BuildContext`; don't pass them loose. When behaviour
  changes, bump the matching constant in `core/version.py` following SCENE_SPEC.md §10
  (component library, theme, Manim pin).
- Spec objects (`BeatSpec`, `VideoSpec`, component params) are frozen and are the only
  things that get hashed. State objects (`Beat`, `Video`) are never hashed.
- Cache writes go through `cache.slot(...)`: write to `slot.tmp`, then `slot.commit()`.

## Running tests

Windows (PowerShell, from the repo root):

```powershell
.venv\Scripts\python -m pytest
```

LaTeX (`MathTex`) needs MiKTeX's `latex` and `dvisvgm` on PATH. A per-user install
lives in `%LOCALAPPDATA%\Programs\MiKTeX\miktex\bin\x64`. The first compile downloads
packages and is slow. If a LaTeX test fails on a missing package, retry once; never skip
or weaken it.

macOS:

```bash
.venv/bin/python -m pytest
```

In a git worktree, set `PYTHONPATH` to the worktree root and confirm
`python -c "import chalkdust; print(chalkdust.__file__)"` points into it. An editable
install from another checkout will otherwise test the wrong code.

Point Manim output at a git-ignored directory (`work/` or `media/`) and keep it quiet
(`config.verbosity = "WARNING"`). To check a render, extract a frame with ffmpeg and look
at it.

## Platform notes

- The default voice backend `macos_say` exists only on macOS. On Windows the speech stage
  raises `TTSError` until another backend is wired (`speech/backends/`).
- ffmpeg/ffprobe must be on PATH everywhere: duration probing, mux, concat, loudnorm.
- The `chalkdust` console script points at `chalkdust.cli:main`, which doesn't exist
  yet. Don't document CLI flags until it does.

## Conventions

- Docstrings explain *why* and cite the law: `(D-004)`, `(SCENE_SPEC.md §11)`.
- One focused test per pinned behaviour in `tests/`. Scratch scripts never become tests.
- No LICENSE file. That is the owner's decision.
