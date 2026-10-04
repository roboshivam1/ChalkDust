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
- **LaTeX with `latex` and `dvisvgm` on PATH**, for anything built with `MathTex`
  (`scenes/theme.py: math()`). The current test suite doesn't touch LaTeX; the maths
  components (`EquationDerivation` and friends) will.

### Windows

Verified 2026-10-03 on Windows 11 (build 26300) with Python 3.13.12, in a fresh venv.
The pip install took about 70 seconds. Commands are PowerShell from the repo root.

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
  `%LOCALAPPDATA%\Programs\MiKTeX\miktex\bin\x64` if you need to add it by hand.
- Leave MiKTeX's on-the-fly package install on (`initexmf --show-config-value=[MPM]AutoInstall`
  prints `1`). The first `MathTex` compile is slow while it downloads packages; after
  that it's quick.
- MiKTeX prints `major issue: So far, you have not checked for MiKTeX updates` until
  you run an update check from MiKTeX Console. Harmless.

Two things that are true on Windows today:

- **No voice backend works yet.** The default `VoiceConfig` backend is `macos_say`,
  which needs macOS's `say`; on Windows the speech stage raises
  `TTSError: 'say' not found on PATH`. The Kokoro backend is a stub. Layout validation
  and the tests don't need speech.
- **Theme fonts fall back.** The default theme asks for Archivo, Inter and JetBrains
  Mono. If they aren't installed you'll see `[theme] 'Archivo' missing, using 'Arial'`
  (and the same for Inter → Arial, JetBrains Mono → Courier New). Install the three
  fonts if you want renders to look like the theme intends.

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
- The `macos_say` voice backend uses the built-in `say` command. Nothing to install.
  List voices with `say -v '?'`; the default is `Daniel`.

## Running the tests

```powershell
# Windows
.venv\Scripts\python -m pytest
```

```bash
# macOS
.venv/bin/python -m pytest
```

`22 passed` on the verified Windows machine. If you run tests from a git worktree with an
interpreter that was installed from a different checkout, set `PYTHONPATH` to the
worktree root first and check `python -c "import chalkdust; print(chalkdust.__file__)"`
points where you think it does.

## Usage

> **The CLI lands with the pipeline PR.** `pyproject.toml` already declares a
> `chalkdust` command pointing at `chalkdust.cli:main`, but `chalkdust/cli.py` doesn't
> exist yet, so the installed `chalkdust` command fails with
> `ModuleNotFoundError: No module named 'chalkdust.cli'`. This section gets real
> commands when that PR merges.

**Parallel render (`--jobs`).** `python -m chalkdust render <spec> --jobs N` renders the
beats that miss the cache across N processes (ARCHITECTURE.md §4). The default is one
process per beat that needs rendering, at most the CPU count, except that a draft render
with fewer than five beats to render stays in one process (measured: there the pool's
start-up costs more than it saves); `--jobs 1` renders them one after another in a
single process. Cache hits never start a worker, and RawScene
beats always render in the main process, after the pool (a degraded one's cache slot is
only known once it renders). The clips, cache keys and finished MP4 are the same
whichever you pick.
