# ChalkDust - Manim Video (with voiceover) Generation Pipeline

> *"There is no spoon."*<br>
> *--The Matrix*

There is no animation either, its just math if you look at it that way.

An educational video generator. Topic in, finished Manim-animated explainer with synced
AI voiceover out. Designed to eventually run unattended across a fleet of YouTube channels.

> **Codename is a placeholder.** Find-and-replace `CHALKDUST` when you pick a real name.

## Document set

| File | What it answers |
|---|---|
| `docs/PRD.md` | What we're building, for whom, what "good" means, what we refuse to build |
| `docs/ARCHITECTURE.md` | How the system is put together and why it's shaped this way |
| `docs/SCENE_SPEC.md` | The contract between the LLM and the renderer — the most important doc here |
| `docs/ROADMAP.md` | Build order, phase exit criteria, what to deliberately not build yet |
| `docs/OPERATIONS.md` | Fleet config, publishing, quota, policy risk, cost model |
| `docs/DECISIONS.md` | Architecture decision log — why we chose X over Y, and when to revisit |

## The one-paragraph version

LLMs write Manim code that compiles but looks broken — text off-screen, objects
overlapping, labels colliding. The model can't see the frame it's producing. So we don't
ask it to write Manim. We give it a **library of hand-written, layout-safe scene
components** and ask it to select and parameterize them via a validated JSON spec. A
deterministic compiler turns that spec into Manim. Audio is generated first and measured,
then animations are stretched to fit. Everything is content-addressed and cached so
editing one beat re-renders one beat.

## Reading order

If you're picking this up cold: `PRD.md` → `SCENE_SPEC.md` → `ARCHITECTURE.md` → `ROADMAP.md`.

`SCENE_SPEC.md` before `ARCHITECTURE.md` is intentional. The spec is the load-bearing
abstraction; the architecture is mostly plumbing around it.

## Install

What every platform needs:

- **Python 3.11+** (`pyproject.toml`). Manim is pinned to exactly `0.21.0`; don't
  upgrade it casually, the version is part of every render cache key (D-004).
- **ffmpeg and ffprobe on PATH.** Manim itself no longer shells out to ffmpeg, but
  ChalkDust does: `ffprobe` measures narration length (the number every animation
  run time derives from, D-002) and `ffmpeg` does mux, concat and loudness.
  Each beat's mux and the joined file carry uncompressed PCM audio (`.mov`); AAC is
  encoded once, at the end. `tests/test_mux.py` also writes PCM into `.mp4`, which
  ffmpeg 8.1 does; an older build may refuse it.
- **LaTeX with `latex` and `dvisvgm` on PATH**, for anything built with `MathTex`
  (`scenes/theme.py: math()`): `EquationDerivation`, `UnitBreakdown`, `GraphPlot` tick
  labels and friends. The semantic rung compiles every LaTeX string before any speech
  or render, so a spec with maths needs the toolchain even for `validate`.
- **Only render specs you trust.** LaTeX can read files. Every TeX compile runs with
  the engine's file-access restriction and no shell escape, in an empty working
  directory (`scenes/theme.py`). On MiKTeX that refuses braced `\input{}` and
  `\include{}`, but MiKTeX does not gate TeX's primitive readers. So a spec, or a
  RawScene `Tex`, can still put the contents of a file on the render machine into the
  video, and validation reports ok. TeX Live's `openin_any=p` is set too and is meant
  to gate those readers, but that was not measured. Specs are hand-written today; this
  matters once specs are generated.

### Windows

Verified 2026-10-05 on Windows 11 (build 26300) with Python 3.13.12, in a fresh venv:
`pip install -e ".[dev]"` took 56 seconds. Commands are PowerShell from the repo root.

```powershell
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -e ".[dev]"
```

Pango and Cairo need no separate install on Windows: `manimpango` and `pycairo` arrive
as prebuilt `win_amd64` wheels with the libraries bundled.

System tools (the winget packages on the verified machine):

```powershell
winget install Gyan.FFmpeg        # ffmpeg + ffprobe (8.1 verified)
winget install MiKTeX.MiKTeX      # latex + dvisvgm (MiKTeX 25.12 verified)
```

- Open a new terminal afterwards so both land on PATH. A per-user MiKTeX lives in
  `%LOCALAPPDATA%\Programs\MiKTeX\miktex\bin\x64`; every run documented here had that
  directory **first** on PATH, so `latex` and `dvisvgm` were MiKTeX's:
  `$env:Path = "$env:LOCALAPPDATA\Programs\MiKTeX\miktex\bin\x64;" + $env:Path`
- Leave MiKTeX's on-the-fly package install on (`initexmf --show-config-value=[MPM]AutoInstall`
  prints `1`). The first `MathTex` compile is slow while it downloads packages; after
  that it's quick.
- MiKTeX prints `latex: major issue: So far, you have not checked for MiKTeX updates`
  to stderr on compiles until you run an update check from MiKTeX Console. Harmless,
  but it makes a PowerShell `2>&1` capture look like an error.
- **The work dir must have no `~` in its full path.** TeX reads `~` as an active
  character, so every `MathTex` fails under such a path. The 8.3 short names Windows
  puts in `%TEMP%` (`C:\Users\LOKAVY~1\...`) are fine: ChalkDust expands them before
  handing Manim the directory (`render/worker.py: long_path`). A `~` that survives the
  expansion is refused up front with exit code 9.
- Write specs as UTF-8. PowerShell 5.1's `>` and `Out-File` write UTF-16 by default,
  which `chalkdust` refuses with a message naming the fix (`Set-Content -Encoding utf8`).
- **Voice works out of the box.** A spec that names no `backend` gets `windows_sapi`
  (the built-in voice `Microsoft David Desktop`, through Windows PowerShell 5.1; no
  install). See [Voice](#voice).
- **Theme fonts are optional.** The default theme asks for Archivo, Inter and JetBrains
  Mono. If one isn't installed you'll see `[theme] 'Archivo' missing, using 'Arial' for
  heading` (Inter falls back to Arial, JetBrains Mono to Courier New) and the render
  proceeds. Install the three fonts if you want renders to look like the theme intends.
  The fonts that *resolve* are part of the render cache key (D-004): the same spec on a
  box with and without the fonts draws different pixels and never shares a cached clip.

### macOS

Not re-run as part of the Windows verification above; these follow from the code and
from how the project was developed on a Mac.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

- Install ffmpeg (which includes ffprobe) and a LaTeX distribution that provides `latex`
  and `dvisvgm`, both on PATH. If `pip` can't build `manimpango` or `pycairo`, follow
  the macOS section of the Manim Community install guide for the system libraries.
- A spec that names no `backend` gets `macos_say`, the built-in `say` command. Nothing
  to install. List voices with `say -v '?'`; the default is `Daniel`.
- Theme fonts fall back to Helvetica Neue and Menlo before Arial and Courier New
  (`scenes/theme.py: FALLBACKS`).

### Voice

One rule, in `speech/tts.py: resolve_voice`, the same on every platform. Only the fields
a spec actually sets count:

1. no `voice.backend` is the platform's zero-install voice: `macos_say` on macOS,
   `windows_sapi` on Windows; on any other OS the speech stage stops (exit code 5) and
   asks you to name one;
2. no `voice.voice_id` is that backend's default voice;
3. `voice.rate` is a speed multiplier, 1.0 unless set.

The TTS cache is keyed on the *resolved* voice, so a spec that says nothing never
shares cached audio between macOS and Windows. A spec headed for publication names
`backend` and `voice_id` itself. `kokoro` is the intended production backend, but its
extra (`pip install -e ".[kokoro]"`) does not install on Python 3.13 (pip reports
`Requires-Python >=3.10,<3.13` for kokoro 0.8.1 through 0.9.4), so asking for it on a 3.13
interpreter fails with exit code 5. Details, voice-listing commands and the 3.13 status
are in [`docs/VOICE.md`](docs/VOICE.md).

## Running the tests

```powershell
# Windows
.venv\Scripts\python -m pytest
```

```bash
# macOS
.venv/bin/python -m pytest
```

`pyproject.toml` already sets `addopts = "-q --tb=short"`. **Don't add another `-q`:** two
of them (`-qq`) drop the `N passed` summary line and print only dots. To pass your own
flags, replace the defaults: `python -m pytest -o addopts="" -q -p no:cacheprovider --tb=short`.
There is no `pytest-xdist` in the dev extra, so the suite runs in one process. It builds
and probes real Manim scenes and compiles LaTeX, so the full run is slow (about 17 minutes on
the verified machine, with other renders running alongside); one file (`python -m pytest tests/test_cache.py`) takes seconds.

On the verified Windows machine (Python 3.13.12, MiKTeX first on PATH, the three theme fonts
installed), after the render pool merged (which added the nine tests in
`tests/test_render_pool.py`), the full run ends
`2162 passed, 5 skipped, 4 warnings in 1029.13s (0:17:09)`, exit code 0. The five skips are expected there: `RawScene` declares no examples (two tests), `say`
exists only on macOS, the Kokoro extra needs Python below 3.13, and Inter draws `U+E000`.

If you run tests from a git worktree with an interpreter that was installed from a
different checkout, set `PYTHONPATH` to the worktree root first and check
`python -c "import chalkdust; print(chalkdust.__file__)"` points where you think it does.

### Component snapshots

`tests/snapshots/<Component>.json` records what each component builds for its
`examples()` (`SCENE_SPEC.md` §11 rule 6). Each file has a structure half that every
machine compares, and a layout-and-geometry half kept per *font fingerprint*
(`platform|heading font|body font|mono font`, e.g. `win32|Archivo|Inter|JetBrains Mono`)
that is compared only on a machine whose fingerprint was recorded and skipped, with the
reason stated, elsewhere. Regenerate only on purpose, with a reason, which is stored in
the file:

```powershell
.venv\Scripts\python -m chalkdust.validate.snapshot --reason "why it changed" --component TitleCard
```

`--reason` is required (the command refuses without one), `--component` repeats, and
omitting it rewrites every snapshotted component. To record the fallback-fonts baseline
from a box that has the theme fonts, hide them: `--hide-font Archivo --hide-font Inter
--hide-font "JetBrains Mono"` records the `win32|Arial|Arial|Courier New` fingerprint.
`RawScene` is exempt and refused. A new component ships its snapshot in the same change
as the component; `tests/test_snapshots.py` fails without it. `--dir <path>` writes
somewhere else, which is how to try the command without touching `tests/snapshots/`.

## Usage

The `chalkdust` console script (or `python -m chalkdust`, identical) has two verbs. Both
take a spec file: hand-written JSON in the format of `docs/SCENE_SPEC.md`. Five runnable
specs are in `examples/`. Every command below was run on the verified Windows machine.

```powershell
# Rungs 1-3: schema, content and layout. No speech, no render.
.venv\Scripts\chalkdust validate examples\binary_search.json
# examples\binary_search.json: ok (4 beats)

# Spec -> MP4. Draft is the default.
.venv\Scripts\chalkdust render examples\binary_search.json --quality draft
.venv\Scripts\chalkdust render examples\binary_search.json --quality final
```

`render` validates first (a spec that cannot lay out never costs a TTS call), then runs
speech for every beat, renders each beat from its measured duration, and assembles. It
prints one line per beat saying what the cache did, plus a line naming the pool when it
starts one:

```
rendering examples\binary_search.json at final
  rendering 4 beat(s) across 4 processes
  b01  speech cached  render rebuilt   10.03s  TitleCard
  ...
  beats: 0 cached, 4 rebuilt
wrote out\cs-binary-search-intro-final.mp4
```

| Option | Applies to | Default |
|---|---|---|
| `--quality draft\|final` | `render` | `draft` = 854x480 at 15 fps; `final` = 1920x1080 at 60 fps |
| `--out PATH` | `render` | `out/<video_id>-<quality>.mp4` |
| `--cache-dir DIR` | `render` | `.cache` |
| `-j`, `--jobs N` | `render` | one process per beat that needs rendering, at most the CPU count; a draft render with fewer than 5 beats stays in one process. `--jobs 1` renders one after another in this process; `0` is refused (exit 2) |
| `--work-dir DIR` | both | `work` |
| `-v`, `--verbose` | both | off; shows Manim's own logging and progress bars |

**Parallel render (`--jobs`).** `render` renders the beats that miss the cache across
N processes (`chalkdust/render/pool.py`, ARCHITECTURE.md §4). Cache hits never start a
worker. With no `--jobs`, it starts one process per beat to render, at most the CPU count,
except that a draft render of a spec with fewer than five beats (or with fewer than five
to render) stays in one process: there the pool's start-up costs more than it saves.
`--jobs 1` renders them one after another in a single process. `RawScene` beats always
render in the main process, after the pool, since a degraded one's cache slot is known
only once it renders. The clips, cache keys and finished MP4 are the same whichever you
pick: here, `cs_hash_table` rendered cold with `--jobs 4` and with `-j 1` into two cache
dirs gave the same nine keys and byte-identical clips (run as `python -m chalkdust`, which
is the same entry point).

```powershell
.venv\Scripts\chalkdust render examples\cs_hash_table.json --jobs 4
#   rendering 9 beat(s) across 4 processes
.venv\Scripts\chalkdust render examples\cs_hash_table.json -j 1
```

### The examples

| Spec | Beats | Components | Notes |
|---|---|---|---|
| `examples/binary_search.json` | 4 | TitleCard, BulletReveal | no LaTeX; the quickest end-to-end check |
| `examples/completing_the_square.json` | 5 | adds EquationDerivation | needs LaTeX |
| `examples/cs_hash_table.json` | 9 | BoxFlow, DataStructureViz, StepTrace, CodeWalk, Callout, ZoomHighlight, ... | carries an artifact between beats (`registers` / `carry_in`) |
| `examples/jee_incline_pulley.json` | 10 | ProblemStatement, SolutionStep, FreeBodyDiagram, AnswerBox, ... | JEE Advanced worked solution |
| `examples/science_orbits.json` | 9 | VectorField, SplitCompare, UnitBreakdown, GraphPlot, ... | science explainer |

None sets a `voice`, so each speaks in the platform's default voice. Measured after the
render pool merged, whole command, draft, cold caches and a fresh work dir, Windows SAPI
voice, default `--jobs`, on a machine that was also running other jobs (one load sample
right after read 43% CPU), so read them as rough: `binary_search` 20.6 s (4 beats, one process),
`completing_the_square` 38.8 s, `cs_hash_table` 58.3 s, `science_orbits` 66.9 s,
`jee_incline_pulley` 98.3 s (10 beats across 10 processes). Speech for every beat is
synthesised, one beat at a time, before any render starts. The `binary_search`
render at `final` took 41.1 s with its speech already cached (4 beats across 4 processes).
Re-running `binary_search` at draft with nothing changed took 12.7 s, every beat cached.

### RawScene beats

`RawScene` is the escape hatch: a beat that carries Manim code instead of picking a
library component (`docs/SCENE_SPEC.md` §7, `chalkdust/scenes/components/raw_scene.py`).
The code is checked against an import and name allowlist, then rendered in a
subprocess with a **120 s timeout** (`raw_scene.TIMEOUT_S`), with the usual layout
assertions after every `play()`. Any failure (a forbidden import or name, a crash, the
timeout, a layout violation, a carried-in artifact) **degrades** the beat to a
`BulletReveal` of its narration instead of failing the video (D-010), so the render still
exits 0. The beat's line says so and gives the full path of the usage log that holds
the reason (shortened here):

```
  b01  speech synth   render rebuilt    5.71s  RawScene -> DEGRADED to BulletReveal (reason in ...\work\manim\raw_scene_usage.jsonl)
  beats: 0 cached, 1 rebuilt, 1 degraded (b01)
```

Only a RawScene that rendered is cached. **A failed one is not**, since a timeout or a
crash may be the machine's fault, so its code runs again on every render. The fallback
`BulletReveal` is cached under its own key, so on a re-run the line reads
`render cached` even though the RawScene was tried again first. A RawScene that hangs
therefore costs the full 120 s on every render, not just the first: here, a one-beat
spec whose code is `while True: pass` took 128.8 s cold and 123.7 s re-run with its
speech and fallback cached. `validate` never runs the code, so it cannot see a hang. For
a failure it can see statically (a syntax or allowlist problem, a `carry_in`) it prints
`RawScene will degrade to BulletReveal (<reason>)` and checks the fallback instead; in
neither case does it refuse the beat for its code. To stop paying for a failing
RawScene, fix the code or swap the beat for a library component.

### What gets written where

| Path (all git-ignored) | Holds |
|---|---|
| `out/<video_id>-<quality>.mp4` | the finished video (`--out` to move it) |
| `.cache/tts/<key>.wav` | normalised narration, keyed on the text and the *resolved* voice |
| `.cache/beats/<key>.mp4` | one rendered clip per beat, keyed on everything that determines it: component, params, carry-ins, measured duration, build context, the font-resolved theme, resolution and frame rate, and the mechanical repair plan |
| `work/manim/` | Manim's scratch, including its text and LaTeX caches (`texts/`, `Tex/`); while a `--jobs` pool runs, `pool/` holds each worker's private copy of those two, removed when the pool is done |
| `work/manim/raw_scene_usage.jsonl` | the RawScene usage log: one JSON line per RawScene beat per render, with its rationale, outcome (`rendered` or `degraded`), the degradation reason and whether the clip came from the cache. Appended to, never rotated |
| `work/assemble/<video_id>-<quality>/` | mux and concat intermediates, deleted after a successful render |

Edit one beat's narration and re-run: only that beat's speech and render are rebuilt
(measured: `beats: 3 cached, 1 rebuilt`). Draft and final never share clips, since the
tier is in the key. Deleting `.cache/` is always safe; it only costs time.

### Exit codes

From `chalkdust/cli.py`. A calling script can tell a spec to fix from a render to retry.
A failure goes to stderr as `chalkdust: <what>: <detail>`.

| Code | Meaning | Raised when |
|---|---|---|
| 0 | ok | |
| 2 | usage error (argparse) | a missing `spec` argument or unknown option; observed |
| 3 | spec invalid | unreadable or non-UTF-8 file, schema error, unknown component or bad params, bad `video_id`, a broken `carry_in` reference; observed (unknown component, missing file, UTF-16 file) |
| 4 | layout refused | a component cannot fit its content legibly even after mechanical repair (rung 3); observed with six long bullets |
| 5 | speech failed | backend unknown, missing or unusable: Kokoro without its extra, no default voice on this OS, a `rate` SAPI cannot speak. A per-beat failure names the beat (`speech failed: b01: ...`); a voice that cannot be resolved does not (`docs/VOICE.md`); observed (Kokoro, unknown backend) |
| 6 | render failed | Manim raised while rendering a beat, or a `--jobs` worker process died (the message names the beats that were in flight); not reproduced |
| 7 | assembly failed | the ffmpeg mux, concat or loudness step failed; not reproduced |
| 8 | semantic refused | rung 2: narration too short for the animation or too long for one beat (`duration`: over 25 s estimated at 160 words per minute, so at most 66 words; the schema's 80-word cap is only a sanity bound: past it, rung 1 refuses first with exit 3), more text than the regions hold (`capacity`), LaTeX that does not compile (`invalid_latex`); observed (`duration`, `invalid_latex`) |
| 9 | directory unusable | `--work-dir` or `--cache-dir` is a file or cannot be created, or the work dir's full path contains `~`; observed |
| 10 | toolchain failed | TeX did not finish a compile in time on every attempt: retry, the spec is not at fault; not reproduced |

Codes 6, 7 and 10 are read from `cli.py` and `pipeline.py`, not reproduced.

A `RawScene` beat never sets an exit code of its own: a failure, the 120 s timeout
included, degrades it and the render exits 0 ([RawScene beats](#rawscene-beats)). Only
the fallback `BulletReveal` render can fail, as any other beat does.
