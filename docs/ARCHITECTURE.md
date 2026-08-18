# CHALKDUST — Architecture

**Status:** Draft v0.1
**Prerequisite reading:** `SCENE_SPEC.md`

---

## 1. Shape of the system

A content-addressed pipeline of pure-ish stages. Each stage takes an artifact, produces a
new artifact, and writes it to a cache keyed by the hash of its inputs. Re-running any
stage with unchanged inputs is free.

```
  topic + channel config
          │
     [1] RESEARCH ────────── grounded facts, sources
          │
     [2] SCRIPT ──────────── beat sheet: narration + component choices
          │
     [3] ══ REVIEW GATE ══   human (early) → automated (later)
          │
     [4] SPEECH ──────────── wav per beat + duration + word timings
          │
     [5] COMPILE ─────────── spec → Manim scene objects
          │
     [6] VALIDATE ────────── geometric + visual checks ⇄ repair loop
          │
     [7] RENDER ──────────── parallel, per beat, 1080p
          │
     [8] ASSEMBLE ────────── concat, mux, normalize, captions, chapters
          │
     [9] PACKAGE ─────────── title, description, thumbnail, tags
          │
    [10] ══ PUBLISH GATE ══  approval + fact verification
          │
    [11] UPLOAD ──────────── YouTube API, per-channel credentials
```

Two gates, deliberately. Reviewing a script takes ninety seconds; diagnosing a broken
eight-minute video takes twenty minutes. Push judgment as early in the pipeline as it
will go.

## 2. Why audio comes before rendering

The classic mistake is rendering the animation and then trying to fit narration to it.
That gives you drift, dead air, and visuals that finish before the sentence does.

Invert it. **Generate speech first, measure it, then stretch the animation to fit.**

```python
beat.audio = tts(beat.narration)          # cached by hash(text, voice, params)
beat.duration = probe(beat.audio)         # ffprobe, exact
scene.run_time = beat.duration            # component derives its own timing
```

`manim-voiceover` implements this pattern inside the Scene. We keep TTS as a **separate
cached stage** instead, because otherwise every re-render hits the TTS backend again —
which is slow, costs money on hosted voices, and reintroduces nondeterminism into a
pipeline we want to be reproducible.

Sub-beat sync (highlight a word exactly as it's spoken) needs word-level timings. Run
forced alignment over the generated audio — Whisper-based alignment works well and you
already have that tooling. The word timing table also gives you SRT captions for free.

## 3. Caching

Content-addressed, on disk. Every artifact filename is the hash of everything that
determines it.

```
beat_key = sha256(
    narration_text,
    component_spec_json,
    style_theme_version,
    component_library_version,
    manim_version,
    render_quality,
)
```

```
.cache/
  tts/<hash>.wav              text + voice + params
  latex/<hash>.svg            LaTeX source + preamble
  beats/<hash>.mp4            full beat key
  frames/<hash>/              draft frames for validation
  research/<hash>.json        topic + query set
```

Editing beat 7's narration changes beat 7's key. Nothing else moves. This single decision
is what turns "iterate on a video" from a 20-minute cycle into a 40-second one, and it's
worth building correctly in Phase 0 even though it feels premature.

**LaTeX caching matters more than it sounds.** LaTeX compilation is a large fraction of
Manim runtime on equation-heavy content, and equations repeat constantly across a
derivation.

## 4. Rendering strategy

**Two-tier quality.** Everything validates at 480p15. Only after all gates pass does the
1080p60 render run. Draft renders are roughly an order of magnitude faster, so the repair
loop iterates cheaply and full-quality compute is only spent on content that's already
known good.

**Parallel by beat.** Beats are independent by construction (`SCENE_SPEC.md` §2), so they
render across a process pool and concatenate with ffmpeg. On an 8-core box this is the
single largest speedup available — bigger than any Manim-level optimization.

```
render_pool(beats) → [b01.mp4, b02.mp4, ...] → ffmpeg concat → video.mp4
```

Concat requires identical codec params across segments. Pin them in one place; a mismatch
here produces a file that plays fine locally and breaks on upload.

**Transitions** happen at the assembly layer via ffmpeg filters, not inside Manim. Doing
them in Manim would couple adjacent beats and destroy independent rendering.

## 5. Module layout

```
chalkdust/
  core/
    models.py           Pydantic: Video, Beat, ComponentSpec, ChannelConfig
    cache.py            content-addressed store
    pipeline.py         stage orchestration, resume-from-stage
    errors.py           typed failures the repair loop can dispatch on

  research/
    gather.py           web search + retrieval
    ground.py           claim → source mapping

  script/
    outline.py          topic → beat sheet
    narrate.py          beat → narration text
    specify.py          beat → ComponentSpec
    critic.py           second-pass review of the generated script

  speech/
    tts.py              backend-agnostic interface
    align.py            forced alignment → word timings
    backends/           kokoro.py, elevenlabs.py, ...

  scenes/
    base.py             ChalkdustScene, region system, fit helpers
    regions.py          layout grid, safe area
    theme.py            palettes, typography, motion language
    components/         one file per component
    registry.py         name → class, with version

  validate/
    schema.py           rung 1
    semantic.py         rung 2
    geometric.py        rung 3 — bounding-box assertions
    visual.py           rung 4 — VLM on sampled frames
    factual.py          rung 5 — sympy, answer keys
    repair.py           the bounded loop

  render/
    compile.py          spec → Scene
    worker.py           single-beat render subprocess
    pool.py             parallel orchestration
    assemble.py         concat, mux, normalize, captions

  package/
    metadata.py         title, description, tags, chapters
    thumbnail.py        generated, channel-themed

  publish/
    youtube.py          resumable upload, per-channel OAuth
    scheduler.py        fleet cadence

  fleet/
    channels/           one config file per channel
    topics.py           backlog, dedupe against published titles

  cli.py
```

Module boundaries follow pipeline stages. Nothing imports across a boundary except
through `core.models`. This is the same modular-monolith discipline as the MCS platform —
one process, hard internal seams, so extracting a stage later (say, rendering onto a
separate box) is mechanical.

## 6. Data model

```python
class ChannelConfig(BaseModel):
    slug: str
    vertical: Literal["science", "cs", "jee"]
    theme: str                    # visual identity
    voice: VoiceConfig            # backend, voice id, rate, register
    pacing: PacingProfile         # target beat length, cut rate, density
    opening_formula: str          # channel's signature hook structure
    target_duration: tuple[int, int]
    component_preferences: list[str]
    youtube: YouTubeCredentials
    cadence: CronExpr

class Beat(BaseModel):
    id: str
    narration: str
    component: str
    params: dict                  # validated against the component's model
    carry_in: list[str] = []
    transition: TransitionKind = "cut"
    # populated by later stages
    audio_path: Path | None = None
    duration: float | None = None
    word_timings: list[WordTiming] = []
    render_path: Path | None = None
    degraded: bool = False
```

`ChannelConfig` carries real differentiation, not just a name and a color. That's a
product requirement (`PRD.md` §8) expressed as a data structure — if two channels can be
described by the same config with different strings, they will look like the same channel
to YouTube.

## 7. State and jobs

SQLite for job state. Same durable-queue pattern as JARVIS MK3: jobs are rows, stages are
transitions, crashes resume from the last completed stage rather than restarting.

```sql
jobs(id, channel, topic, stage, status, attempts, created_at, updated_at)
artifacts(job_id, stage, path, hash, created_at)
events(job_id, ts, level, stage, message)
publishes(job_id, video_id, url, published_at)
```

Postgres if this ever runs on more than one machine. It won't for a long time. Don't
build for that.

## 8. LLM usage

Three distinct calls with different requirements:

| Call | Needs | Notes |
|---|---|---|
| Outline | Strong reasoning, pedagogy | Most quality-determining call in the system |
| Narration | Voice consistency, register | Channel config drives tone; few-shot from prior videos |
| Spec | Structured output discipline | Constrained JSON; retry on schema failure is cheap |

**Split them.** A single call that produces outline + narration + spec optimizes for
none of them and fails opaquely. Separate calls fail specifically.

**Add a critic pass.** After the script is generated, a second call reviews it against the
quality bar in `PRD.md` §7 — hook quality, one-idea-per-beat, visual buildup. Cheap
relative to rendering, and it catches the "technically correct but boring" failure that
no automated check will.

**Cache aggressively.** Prompt + model + params → response. During development you re-run
the same topic dozens of times.

## 9. Failure philosophy

Fail early, fail loud, degrade late.

- Schema and semantic failures kill the job immediately with a specific message.
- Geometric failures enter the repair loop.
- Repair exhaustion degrades the beat and continues.
- Only assembly and upload failures kill a job that's made it past validation.

**The pipeline never produces a broken video.** It produces a good video, a degraded
video with flagged beats, or a clear error. There is no fourth outcome.

## 10. Stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.11+ | Manim is Python; no argument here |
| Animation | Manim Community, pinned | Pin hard. Manim breaks across versions. |
| Validation | Pydantic v2 | Same pattern as the SSG; fast, good errors |
| TTS | Pluggable; Kokoro self-hosted default | Cost at fleet scale; hosted for flagship |
| Alignment | Whisper-based forced alignment | Already in your toolchain |
| Video | ffmpeg | Concat, mux, loudnorm, transitions |
| State | SQLite | Single box, durable, zero ops |
| Scheduling | cron or APScheduler | Airflow is absurd for this |
| Upload | YouTube Data API v3 | Resumable uploads, per-channel OAuth |

Deliberately boring. The interesting problem is the component library, and every hour
spent on infrastructure novelty is an hour not spent there.

## 11. What this architecture makes hard

Worth stating explicitly, so it's a choice rather than a surprise.

- **Continuous animation across beat boundaries.** Independent rendering forbids it.
  Cross-beat continuity is limited to carry-ins and ffmpeg transitions.
- **Genuinely novel visuals.** Everything routes through the component library or the
  escape hatch. There's a ceiling on visual originality that grows only as fast as the
  library does.
- **Interactive iteration.** This is a batch pipeline. There's no scrubbing timeline.

If any of these becomes the binding constraint, that's the signal to revisit — not to
patch around it.
