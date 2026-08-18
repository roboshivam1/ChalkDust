# CHALKDUST — Decision Log

Each entry records what was decided, what it costs, and what would make it wrong. The
last field is the important one — it's the difference between a decision and a dogma.

---

### D-001 — Component library over freeform Manim generation
**Decided:** Phase 0

The LLM selects and parameterizes hand-written scene components via a validated spec. It
does not author Manim code.

**Because:** an LLM can't see the frame it produces. Layout correctness becomes a
property of the library, verified once, instead of a property of each generation,
verified never.

**Costs:** hard ceiling on visual expressiveness. Every new visual idea is a code change.
The library becomes the bottleneck on content variety.

**Revisit if:** escape-hatch usage stays above 25% after Phase 3, or if models become
reliably good at spatial layout — which would be a genuine architectural shift, not a
prompt improvement.

---

### D-002 — Audio generated before animation
**Decided:** Phase 0

TTS runs as an upstream stage. Measured duration drives animation run times.

**Because:** fitting audio to rendered video produces drift and dead air. The inverse is
exact.

**Costs:** narration can't respond to how the animation actually turned out. Script edits
invalidate audio.

**Revisit if:** never, realistically. This one is settled.

---

### D-003 — TTS as a separate stage, not inside the Scene
**Decided:** Phase 0

We use the `manim-voiceover` timing pattern but keep TTS out of the renderer.

**Because:** TTS inside the Scene re-hits the backend on every render — slow, costly on
hosted voices, and nondeterministic in a pipeline we want reproducible.

**Costs:** slightly more plumbing than using the library as intended.

**Revisit if:** TTS becomes free and instant, which is close enough to true for local
models that this is worth re-examining if the plumbing ever gets annoying.

---

### D-004 — Content-addressed caching from day one
**Decided:** Phase 0

Every artifact is keyed by the hash of its determining inputs.

**Because:** retrofitting content-addressing into a working pipeline means rewriting
every stage boundary. It's a day now, a week later. It also converts iteration from a
20-minute cycle to under a minute, which changes how much you'll actually iterate.

**Costs:** hash-key discipline everywhere. Forgetting to include an input in the key
produces stale artifacts, which is a confusing class of bug.

**Revisit if:** never. Just be rigorous about key completeness.

---

### D-005 — Beats render independently
**Decided:** Phase 0

No beat may depend on mobject state left by the previous beat. Continuity is declarative
via `carry_in`.

**Because:** this is what makes parallel rendering and incremental re-render possible.
Both depend on it entirely.

**Costs:** no continuous animation across beat boundaries. Transitions happen at the
ffmpeg layer.

**Revisit if:** cross-beat continuity becomes the dominant quality complaint. It would
mean giving up parallelism, so the bar is high.

---

### D-006 — Two-tier rendering
**Decided:** Phase 1

Validate at 480p15, render final at 1080p60 only after all gates pass.

**Because:** the repair loop can iterate cheaply, and full-quality compute is spent only
on content already known good.

**Costs:** some visual issues only appear at full resolution — fine text rendering,
antialiasing artifacts.

**Revisit if:** resolution-dependent bugs start slipping through. Fix would be a final
sampled check at full quality, not abandoning the tier split.

---

### D-007 — SQLite for job state
**Decided:** Phase 0

**Because:** single machine, durable, zero operational overhead. Same durable-queue
pattern already proven in JARVIS MK3.

**Costs:** single-writer. Won't scale across machines.

**Revisit if:** rendering moves to a separate box. Postgres then, and the module
boundaries in `ARCHITECTURE.md` §5 make it mechanical.

---

### D-008 — Separate LLM calls per script stage
**Decided:** Phase 2

Outline, narration, spec, and critique are four calls, not one.

**Because:** each has different requirements — reasoning, voice, structural discipline,
judgment. A combined call optimizes for none and fails opaquely. Separate calls fail
specifically, which matters more than the token savings.

**Costs:** more calls, more latency, more state to pass between them.

**Revisit if:** cost becomes material — but at these volumes, rendering dominates.

---

### D-009 — Human review gate stays longer than technically necessary
**Decided:** Phase 2

Script approval remains manual well past the point where automated quality checks pass.

**Because:** it's the cheapest quality intervention in the system (90 seconds vs. 20
minutes diagnosing a bad render), and it's the main mitigation against the policy risk in
`PRD.md` §8.

**Costs:** blocks true unattended operation, which was a stated goal.

**Revisit if:** the critic pass demonstrably matches human judgment across 50 videos.
Measure this rather than assuming it.

---

### D-010 — Degrade rather than fail
**Decided:** Phase 3

A beat that can't render correctly falls back to the simplest component carrying its
narration. The video completes with the beat flagged.

**Because:** a flagged beat in a finished video is reviewable. A failed pipeline run at
beat 34 of 40 wastes everything upstream.

**Costs:** silent quality degradation if nobody reads the flags.

**Revisit if:** degraded beats start reaching publication. Fix is a hard gate on degraded
count, not abandoning graceful degradation.

---

### D-011 — Deliberately boring infrastructure
**Decided:** Phase 0

Python, ffmpeg, SQLite, cron. No orchestration framework, no message broker, no
containers until there's a reason.

**Because:** the interesting problem is the component library. Infrastructure novelty is
time not spent there.

**Costs:** some manual work that a framework would handle.

**Revisit if:** operational overhead exceeds the cost of adopting a framework. It won't
at three channels.

---

### D-012 — JEE gets a separate pipeline branch
**Decided:** Phase 5

Problem-solution content is not just another channel config.

**Because:** absolute accuracy requirement, rigid content template, narrow visual needs,
and mandatory symbolic verification. Forcing it through the general pipeline compromises
both.

**Costs:** two code paths to maintain.

**Revisit if:** the branches converge naturally — but the verification requirement alone
probably keeps them apart permanently.
