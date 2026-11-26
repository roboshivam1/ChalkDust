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
recommendation; the owner decides. Where the code has moved past a factual description
in them (the module layout in ARCHITECTURE.md §5 is the main one), say so; don't edit
the doc to match.

Usage, the CLI, exit codes and the examples are in `README.md`; voice backends in
`docs/VOICE.md`. Don't restate them here.

## Layout rules (non-negotiable)

- **No absolute positioning.** Place content with `fit_to_region(mob, Region.X)` or
  relative to another mobject (`scenes/regions.py`, SCENE_SPEC.md §4). Regions:
  `TITLE_BAR`, `STAGE` (= `STAGE_LEFT` + `STAGE_RIGHT`), `LOWER_THIRD`, all inside the
  safe area.
- **Theme-only styling.** Build text with the `scenes/theme.py` constructors
  (`title_text`, `heading_text`, `body_text`, `caption_text`, `mono_text`, `math`) and
  colour with palette roles. A component never takes a coordinate, colour or font size
  from the spec (SCENE_SPEC.md §3, §11). Text that bypasses the constructors (CodeWalk
  through Manim's `Code`) must call `theme.check_renderable(text, font)` itself.
- **Derived timing, in whole frames.** Split the beat with `scene.budget(*weights)` and
  never hardcode or compute a run time (D-002, SCENE_SPEC.md §11 rule 4). `budget()`
  returns whole-frame run times that sum to exactly `scene.beat_frames`
  (`ceil(audio * fps)`), and the scene snaps every `play()` and `wait()` to whole frames,
  so a clip is exactly that long. A run time that does not come from `budget()` makes the
  beat drift from its audio by up to a frame per segment (`tests/test_budget_frames.py`
  counts the frames of real encoded clips).
- **Render correctly or raise.** Too much content is a typed `LayoutError`, never
  shrunken or off-screen text. The fix for overflow is splitting the beat. Call
  `scene.settle()` at points a viewer will look at; mark things that must not overlap
  with `scene.exclusive(...)`. The vocabulary is next.

## Refusals: the typed vocabulary

A refusal carries a `kind`; the repair loop and the exit codes dispatch on it
(`LayoutError.kind`, `validate/geometric.Finding`, SCENE_SPEC.md §9).

| Kind | Raised by | Rung |
|---|---|---|
| `overflow`, `illegible` | `fit_to_region` and friends: content cannot fit at `MIN_FONT_SIZE` (22) | 3 |
| `out_of_bounds`, `overlap` | `scene.settle()` / `exclusive()`, including a component drawing over a carried artifact it does not act on | 3 |
| `invalid_latex` | `theme.math()` and `semantic.check_latex`: LaTeX that does not compile or draws nothing; `check_latex_source` compiles the raw source, because `MathTex` silently repairs unbalanced braces | 2 and 3 |
| `unrenderable_text` | the theme text constructors via `check_renderable`: a character the resolved font draws as nothing or as a missing-glyph box (right-to-left, emoji, CJK under the fallback fonts), or a string that draws nothing | 3 |
| `capacity`, `duration` | `semantic.check_capacity` / `check_duration`: more text than the regions can hold, narration too short for the animation or over 25 s | 2 |
| `carry_in`, `region_conflict` | a bad carry-in reference, or no free STAGE region for a carried artifact (next section) | 2 |
| `toolchain` | TeX still running after its timeout, every attempt: the machine, not the spec | 2 and 3 |
| `build_error` | any other exception: a bug, never a legitimate refusal | |

`CLEAN_REFUSALS` (`tests/test_layout.py`) is what a stress case may refuse with:
`overflow`, `illegible`, `invalid_latex`, `unrenderable_text`. Anything else from a
stress case, `build_error` above all, fails the suite. A new guard gets its own kind
(`regions.py`), one place that raises it, and a line in `CLEAN_REFUSALS` only if it is
a clean refusal of overloaded content. Spec text is also checked at rung 1: control
characters and private-use, unassigned and surrogate code points are refused
(`pipeline.load_spec`, `core/models.py`).

## Carry-in placement

A component that does not act on a carried artifact would draw over it, so the artifact
goes, dimmed, into the STAGE half the component does not claim (`continuity.carry_region`):
`STAGE_RIGHT` beside a `STAGE_LEFT` component and the reverse, all of STAGE for one that
claims only TITLE_BAR or LOWER_THIRD, in `carry_in` order as rows if there are several.
A component that claims STAGE or both halves leaves none: the beat is refused as
`region_conflict` at rung 2 (the CLI reports it as spec invalid, exit 3). A *consumer*
(`carried_targets()` non-empty: Callout, ZoomHighlight) gets the artifacts centred in
STAGE and lays out around them. Declare `carried_targets()` and `fixture_carry_in()` on a
new consumer; `tests/test_layout.py` builds it with the fixtures on screen.

## Adding a component

Subclass `Component` in `chalkdust/scenes/components/<name>.py`: set `name` and
`Params` (a `ComponentParams` subclass, which is frozen and `extra="forbid"`),
implement `regions()` and `build()`, provide `examples()` (must validate clean) and
`stress()` (about 3x realistic volume; must fit or refuse with a clean kind above),
decorate with `@register`, and import the module in `components/__init__.py` (the
registry is `scenes/components/base.py`). `tests/test_layout.py` walks the registry, so a
registered component gets layout tests with no new test file. If it adds a component, bump
`COMPONENT_LIBRARY_VERSION` (minor; SCENE_SPEC.md §10).

**Merge-time snapshot step.** A component without `tests/snapshots/<Name>.json` fails
`tests/test_snapshots.py`; shipping the snapshot is part of shipping the component. When
you merge a component branch, record it in the same step, from the repo root:

```powershell
python -m chalkdust.validate.snapshot --reason "<Name> merged" --component <Name>
```

A snapshot has two halves. **Structure** (the timeline and the tree of what is on screen,
with measured decisions such as where a line breaks reduced to the fact they were taken)
is compared on every machine. **Layout and geometry** are recorded per *font fingerprint*,
`platform|heading|body|mono` (`validate/snapshot.fingerprint()`, e.g.
`win32|Archivo|Inter|JetBrains Mono`), and compared only where the fingerprint matches;
elsewhere the test skips and says which fingerprint to record. Pixels are never
snapshotted. Regenerate only on purpose, with a reason (`--reason` is required and is kept
in the file); a changed snapshot in a diff needs its reason in the commit message. To add
the fallback-fonts baseline from a box that has the theme fonts, record again with
`--hide-font Archivo --hide-font Inter --hide-font "JetBrains Mono"` (the
`Arial|Arial|Courier New` fingerprint). `RawScene` is exempt and refused. A snapshot
that fails is Manim or a font shifting, or a real change: find out which before
regenerating; never regenerate to make red go green.

## Animations (run 3)

An *animation* (`scenes/components/animation.py`; contract in `docs/research/ANIMATIONS.md`)
is one picture that plays over several consecutive beats, one *phase* per beat. Its
params are a `phase`, optional `cues`, and a *setup* (everything else) that every phase
repeats unchanged.

- **Phases continue.** Phase k+1's first frame equals phase k's last. Both are rebuilt
  from the setup (D-005 holds: no state crosses beats). The `continuity` checks
  (`tests/test_animation.py`) compare the last frame actually written with the next
  phase's entry state and its first written frame. Neither sees the hold before a cued
  first moment, where LimitAtHole once blanked its readouts. So a new animation also
  passes `research/verify_video.py` on an encoded 1080p render with cued first moments.
- **Moments land on words.** A phase is a sequence of `Moment`s. `scene.cue_timeline(*moments)`
  times them from real Kokoro word timings. The anchor is the cue phrase, else sentence k,
  else the moment's weight, and every window gets at least its floor. A cue phrase not in the
  narration, or said twice where the cue could fall, is refused at rung 2 as kind `cue`
  (`core/cues.py`). Cues are in the render key.
- **Everything drawn is computed.** Numbers on screen come from the setup through the
  component's own functions (e.g. `secant_tangent.slope_plan`, `limit_hole._analyse`,
  `riemann_sum.riemann_sum`). A setup that would draw something false is refused as a
  param error that names the fix.
- **Writing one.** Follow `docs/research/ANIMATIONS.md`: one file per component, examples
  and stress cases, a fresh reviewer, and the snapshot recorded at merge. Then add its card
  to `research/animate.py`, or `research/tests/test_animate.py` fails.
- **Quality.** `--quality standard` is 1920x1080 at 30 fps, the delivery tier. On the
  run 3 pilot, 60 fps cost 1.45x the whole render and about 1.9x the render stage
  (`docs/research/RUN3.md`), so 30 fps is the default.
- **Voice.** Kokoro `af_heart` is the default delivery voice. `python -m research.voice_check`
  measures a clip: duration, internal gaps, clipping, and the word error rate of a local
  faster-whisper pass. Run it with `.lab/venv-k312`.

## Cache-key discipline (D-004)

- Keys come from `core/cache.py`: `tts_key(narration, resolved_voice)` and
  `beat_render_key(spec, duration, ctx, theme, tier, repair, *, carried=None)`. Every
  input that changes an artifact must be in its key, and nothing else. A missing input
  serves stale files, the worst bug class here. Narration and `transition` are
  deliberately out of the render key.
- The render key's `theme` is the **font-resolved** theme (`resolve_fonts`), so the
  fonts actually drawn are in the key: a box without Archivo or Inter keys differently
  from one with them. `tier` is the concrete resolution and frame rate (`render/worker.TIERS`),
  `repair` is the mechanical repair plan itself (so repair-code changes need no version
  bump), `carried` is `continuity.carry_in_fingerprint`. The TTS key uses the **resolved**
  voice (`speech/tts.resolve_voice`).
- Version strings travel in `BuildContext`; don't pass them loose. When behaviour
  changes, bump the matching constant in `core/version.py` following SCENE_SPEC.md §10
  (component library, theme, Manim pin).
- Spec objects (`BeatSpec`, `VideoSpec`, component params) are frozen and are the only
  things that get hashed. State objects (`Beat`, `Video`) are never hashed.
- Cache writes go through `cache.slot(...)`: write to `slot.tmp`, then `slot.commit()`.
- Each beat renders in its own `video_dir` (`videos/beat_<key>`, `worker._render`), so
  two renders never overwrite each other's partial movie files. That is what makes the
  render pool safe.

## Render pool

`render --jobs N` (`render/pool.py`, `pipeline.render_stage_pooled`; ARCHITECTURE.md §4):

- Cache hits never start a worker; only beats that miss are pooled, one per cache slot.
- With no `--jobs`: one process per beat to render, at most the CPU count. A draft render
  of a spec with fewer than 5 beats (`pool.DRAFT_MIN_POOLED`), or with fewer than 5 to
  render, stays in-process: the pool's start-up costs more than it saves there.
- `--jobs 1` is the sequential path, in this process.
- `RawScene` beats render in the main process, after the pool: a degraded one's cache slot
  is known only once it renders.
- Output is identical either way: same keys, same clips, same MP4.
- Workers are spawned (never forked) and each gets its own copy of Manim's `texts/` and
  `Tex/` caches under `{media_dir}/pool/`: concurrent writes there tore SVGs. A worker
  failure comes back as data and becomes the same `RenderFailed`/`ToolchainFailed` a
  sequential render raises; a dead worker is `RenderFailed` naming the beats in flight.
- Anything you hand the pool must pickle by reference (`pool.can_ship`); a closure (a test
  double) silently falls back to the in-process path.

## Running tests

Windows (PowerShell, from the repo root):

```powershell
$env:Path = "$env:LOCALAPPDATA\Programs\MiKTeX\miktex\bin\x64;" + $env:Path   # MiKTeX FIRST
.venv\Scripts\python -m pytest -o addopts="" -q -p no:cacheprovider --tb=short
```

macOS: `.venv/bin/python -m pytest`.

- **`pytest -q` doubling.** `pyproject.toml` already sets `addopts = "-q --tb=short"`, so
  a CLI `-q` makes `-qq`, which prints only dots and no summary line. Either pass nothing
  extra or override the defaults as above. Quote the real summary line (`N passed in ...`)
  and the exit code when you report a result; dots are not a result.
- No `pytest-xdist` is installed; don't pass `-n`. The full suite (2167 tests) is slow
  because it builds real Manim scenes and compiles LaTeX; iterate on the files you touched
  and run everything before you merge.
- LaTeX (`MathTex`) needs MiKTeX's `latex` and `dvisvgm` on PATH, with MiKTeX's directory
  ahead of any other TeX. The first compile downloads packages and is slow. If a LaTeX
  test fails on a missing package, retry once; never skip or weaken it.
- In a git worktree, set `PYTHONPATH` to the worktree root and confirm
  `python -c "import chalkdust; print(chalkdust.__file__)"` points into it. An editable
  install from another checkout will otherwise test the wrong code.
- Point Manim output at a git-ignored directory (`work/` or `media/`) and keep it quiet
  (`config.verbosity = "WARNING"`). To check a render, extract a frame with ffmpeg and look
  at it.

## Platform notes

- **Windows and `~`.** TeX reads `~` as an active character, so Manim must never get a
  media or work dir whose full path contains one. 8.3 short names (`C:\Users\LOKAVY~1\...`,
  which is what `%TEMP%` looks like) must be expanded first: `render/worker.long_path`
  does it and every dir handed to Manim goes through it. Code that calls Manim directly
  (a test, a script) does the same. A real `~` in `--work-dir` is exit 9.
- Specs are UTF-8. PowerShell 5.1's `>` and `Out-File` write UTF-16, which the loader
  refuses with the fix named.
- MiKTeX prints `major issue: ... not checked for MiKTeX updates` on stderr per compile;
  it is noise, and it turns a PowerShell `2>&1` capture into a `NativeCommandError`.
- Voice resolution: a spec with no `voice.backend` gets `macos_say` on macOS and
  `windows_sapi` on Windows; elsewhere it must name one (`docs/VOICE.md`). `kokoro`
  does not install on Python 3.13.
- Theme fonts (Archivo, Inter, JetBrains Mono) are optional: a missing one falls back
  (`theme.FALLBACKS`) with a `[theme] ... missing, using ...` line, and the resolved fonts
  enter the render key and the snapshot fingerprint.
- ffmpeg/ffprobe must be on PATH everywhere: duration probing, mux, concat, loudnorm.

## Research harness (`research/`): small local models

Experiments on making small local LLMs fill ChalkDust specs well. It is not part of the
app: `chalkdust/` never imports `research/`, and the harness only calls the validator
and the CLI. Outcome and numbers: `docs/research/FINDINGS.md`. Method:
`docs/research/METHODOLOGY.md`. Background research: the four `docs/research/*-2026-11.md`
notes.

- **Environments.** Use `.lab/venv-harness` for generation, validation, analysis and
  rendering (chalkdust editable install, sympy, antlr4). Use `.lab/venv-train` for
  training (torch cu132, transformers, peft, TRL, bitsandbytes). `.lab/` is git-ignored
  and holds the GGUFs, HF weights, run traces, datasets and videos. Its `STATE.md`
  lists what was installed.
- **Run and read.** Run with `python -m research.run --exp <id> --models <keys>
  --methods <m> --topics dev|heldout|train`. Summarise with
  `python -m research.analyze <id>`, which re-scores content from the saved specs with
  the current checkers.
- **Results are append-only.** Never edit `research/results/*.jsonl`. A fixed bug gets
  a new experiment id, and the old file stays as the record. Every record carries the
  commit and `prompt_hash`; an SFT set is only valid for one prompt hash.
- **Topics are frozen.** In `research/topics.py`, add topics with new ids. HELDOUT never
  goes near training. `research/train_topics.py` asserts at import that no training
  family is an evaluation family or a near-twin of one.
- **Thinking is on by default.** Qwen3.5/3.6, Granite 4.2 and Gemma 4 all think under
  llama-server unless `enable_thinking=false` is sent (`models.py` `chat_extra`). Train
  with the same template kwargs (`train_sft.py --template-kwargs`), and confirm with
  `research/check_template.py` that the trained prompt is the served prompt. Every model
  gets a probe request before an experiment; a failed probe skips the model rather than
  scoring it.
- **Schema order is generation order.** Under grammar-constrained decoding, put any
  reasoning field (`working`, `answer_calc`) before the field it should determine
  (`final_answer`).
- **No closed-model outputs as training targets.** Teachers are open-weight Apache-2.0
  or MIT models run locally.
- **One GPU job at a time.** Queue them with `research/gpu_queue.py` (jobs in
  `.lab/queue/jobs.json`, re-read before each job; finished names in `done.txt`). It waits
  before each job until the GPU is free of other programs and 8 GiB of RAM is free.
  - A job that exits non-zero stops the queue and is left out of `done.txt`.
  - So is a job the memory guard kills: available RAM under 1.5 GiB or commit headroom
    under 2 GiB for 60 s.
  - Status lines go to `.lab/queue/queue.log`.
  - Run it in its own PowerShell window, not as a Claude Code background task, which
    is stopped when memory runs low.
  - One queue per queue directory: a second `python -m research.gpu_queue` on the same
    `QUEUE_DIR` logs `REFUSED` and exits 2. Importing the module starts nothing.
  - A gate job waits for a marker file (for example `.lab/queue/go-sft`) or its
    timeout. Use one to hold a step until a check outside the queue is done.
  - `research.run` exits non-zero, with no record for that video, if llama-server
    dies mid-run. Restart the queue; `--resume` redoes the video.
- **Checks freeze before held-out output.** Fix the answer checks (`research/content.py`)
  on DEV, TRAIN and saved runs only, never in response to T2 or T3 held-out output
  (METHODOLOGY section 3). Pin each fix with a case in `research/tests/test_train_checks.py`.
- **Field leaks.** `research/leaks.py` flags JSON and label text that leaked into a
  component's text field (the TitleCard kicker, FINDINGS item 10). `analyze` and
  `report` show it as the `leak` column. `build_sft` trims leaked kickers and marks
  those rows `"edited": "kicker_trim"`; no other teacher output is edited.
- **Host RAM.** `research/llm.py` turns llama-server's prompt cache off
  (`--cache-ram 0`). Under Gemma 4 a saved slot state is about 1 GB, and the default
  cache is up to 8 GiB. Scoring workers are about 1.7 GB each; the queue sets
  `SCORE_WORKERS=2`.
  - On Windows, llama-server's private bytes sit near its VRAM use even fully offloaded,
    so judge memory by growth after load and by system commit.
  - A 32 GB laptop with desktop apps open has run out twice.
- **Animations in the harness (`anim=1`, run 3).** `research/animate.py` holds the
  animation cards. With `anim=1` an animation is one plan item that code expands to one beat
  per phase. The setup is filled once and carried; later phases get the on-screen numbers
  as FACTS. With `calc=1` the calculator, which also takes `derivative(f, a)`,
  `limit(f, a)` and `integral(f, a, b)`, decides the answer shown and spoken. A plan's
  calculation that is only a number decides nothing. If the lesson has exactly one
  animation that computes a derivative, limit or integral, the calculator computes it from
  that animation's setup instead (`generate._decide_from_animation`, R3-anim-c). `anim=1`
  records carry `PROMPT_HASH_ANIM`; `PROMPT_HASH` is unchanged for every other method.
- **Tests:** `.lab/venv-harness/Scripts/python -m pytest research/tests -o addopts="" -q -p no:cacheprovider`.

## Conventions

- Docstrings explain *why* and cite the law: `(D-004)`, `(SCENE_SPEC.md §11)`.
- One focused test per pinned behaviour in `tests/`. Scratch scripts never become tests.
- Never delete, skip or weaken a test to get green.
- No LICENSE file. That is the owner's decision.
