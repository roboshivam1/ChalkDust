# CHALKDUST — Roadmap

**Principle:** every phase ends with something independently usable. You have several
active projects; an interruption should never strand this one mid-abstraction.

**Estimates are in focused sessions (~3 hours), not calendar time.** Calendar depends
entirely on what else is running.

---

## Phase 0 — Skeleton end-to-end
**~4 sessions**

The goal is a working spine, not a good video. Hand-written JSON spec, three components,
one voice, sequential rendering. Ugly is fine. Broken is not.

- Repo, pinned Manim, project config
- `core/models.py` — Video, Beat, ComponentSpec
- `core/cache.py` — content-addressed store (build this now, not later)
- `scenes/base.py` — ChalkdustScene, region system, `fit_to_region`
- Components: `TitleCard`, `BulletReveal`, `EquationDerivation`
- TTS stage with one backend, duration probing
- Sequential render + ffmpeg concat + mux

**Exit:** a hand-authored 5-beat JSON produces a watchable MP4 with synced audio. Edit
one narration line, re-run, and only that beat re-renders.

**Deliberately not yet:** LLM anywhere in the pipeline.

---

## Phase 1 — Layout safety
**~3 sessions**

The phase that decides whether this project is different from your v1. Everything else is
downstream of layout reliability.

- Formalize regions and safe area
- Geometric validator: bounding-box assertions at settle points
- Mechanical repair: auto-scale, auto-nudge, text wrap
- Two-tier render (480p15 draft → 1080p final)
- Snapshot tests per component
- Stress fixtures: 3× expected content volume through every component

**Exit:** deliberately abusive specs (30-word titles, 12 bullets, 40-character equations)
either render correctly or raise a clear, specific error. Nothing renders broken.

---

## Phase 2 — Script generation
**~4 sessions**

- `script/outline.py` — topic → beat sheet
- `script/narrate.py` — beat → narration, channel voice applied
- `script/specify.py` — beat → validated ComponentSpec
- `script/critic.py` — quality-bar review pass
- LLM response caching
- CLI review gate: show the beat sheet, allow edit, approve or reject

**Exit:** `chalkdust make "why gradient descent works"` produces a script you'd approve
with light edits, then a rendered video. Layout validation passes on first attempt >70%
of beats.

---

## Phase 3 — Component library
**~6 sessions, ongoing after**

Expand to ~15 components. **Drive this from the escape-hatch log, not from imagination** —
build what actual generated scripts keep reaching for.

- Vertical-specific sets (science, CS, JEE per `SCENE_SPEC.md` §5)
- Theme system: palette, typography, motion language per channel
- Carry-in continuity
- `RawScene` escape hatch with sandbox + timeout + degradation

**Exit:** across 10 generated videos in one vertical, escape hatch fires on <10% of beats
and first-attempt layout validation exceeds 90%.

---

## Phase 4 — Efficiency
**~3 sessions**

Only now, because premature optimization here would have been optimizing code you were
still redesigning.

- Parallel beat rendering across process pool
- LaTeX and TTS cache hardening
- Full repair loop: mechanical → regenerate → degrade
- Visual validation rung (VLM on sampled draft frames)
- Resume-from-stage on crash

**Exit:** an 8-minute video renders in under 15 minutes on 8 cores. A narration edit
followed by re-render completes in under 60 seconds.

---

## Phase 5 — Accuracy
**~3 sessions**

Blocking prerequisite for the JEE vertical. Not optional.

- Symbolic verification via sympy for derivations and final answers
- Answer-key cross-check for problem videos
- Dual-model agreement on factual claims
- Source grounding in the research stage
- Publish gate blocks on any verification failure

**Exit:** 20 JEE problems through the pipeline with zero incorrect final answers reaching
the render stage.

---

## Phase 6 — Publishing, one channel
**~4 sessions**

- Metadata generation: title, description, tags, chapters from beat timings
- SRT captions from word timings
- Thumbnail generation, channel-themed, legible at 168×94
- YouTube OAuth + resumable upload
- Publish approval gate

**Exit:** one real channel, ten published videos, human-approved. Watch the retention
data. **This is where you find out whether the thesis holds.**

---

## Phase 7 — Fleet
**~5 sessions — gated on Phase 6 results**

**Do not start this until one channel is actually working.** If Phase 6 videos don't
retain viewers, multiplying them multiplies nothing.

- ChannelConfig with real differentiation (theme, voice, pacing, opening formula)
- Topic backlog per channel, dedupe against published titles
- Scheduler, quota-aware
- Multi-account credential management
- Operations dashboard: job states, failures, degraded beats
- Cost tracking per video

**Exit:** three channels publishing on schedule, no manual intervention for a week, no
policy flags.

---

## Phase 8 — Quality flywheel
**Ongoing**

- Retention analytics fed back into pacing and hook selection
- Thumbnail and title A/B testing
- Component usage stats → library priorities
- Comment sentiment monitoring for the "AI slop" signal
- Per-channel quality review, monthly, honest

---

## Sequencing rationale

**Why cache in Phase 0:** retrofitting content-addressing into a working pipeline means
rewriting every stage boundary. It's a day now and a week later.

**Why layout safety before the LLM:** if you add generation first, you'll spend Phase 2
debugging whether bad output is a prompt problem or a layout problem. Make layout
unimpeachable, and every remaining failure is unambiguously a generation problem.

**Why efficiency in Phase 4:** parallelism and aggressive caching freeze interfaces.
Phases 1–3 are exactly when those interfaces are still moving.

**Why accuracy before publishing:** the ordering is obvious in retrospect and easy to
skip when the video looks good.

**Why fleet last:** it's the least interesting engineering and the highest product risk.
It's also the part that's tempting to build first because it's the part that sounds like
a business.

---

## Kill criteria

Worth writing down before you're invested. Stop or substantially rethink if:

- Escape hatch usage stays above 25% after Phase 3 — the component-library bet is wrong,
  and freeform generation with heavy validation may be the better architecture
- Phase 6 videos average under 30% retention across 10 videos — the output isn't good
  enough and the fleet is worthless
- A policy strike lands on the first channel — the automated-fleet thesis needs rethinking
  before any scaling
- Per-video cost exceeds what the channel could plausibly earn, with no path down

---

## The thing to resist

The fleet layer is roughly 15% of the work, entirely mechanical, and by far the most fun
to imagine. Building it before the core is good just means automating the production of
mediocre videos at scale.

**One video you'd genuinely put your name on. Then worry about the second.**
