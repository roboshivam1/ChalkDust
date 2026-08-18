# CHALKDUST — Product Requirements

**Status:** Draft v0.1
**Owner:** Shivam Kapoor

---

## 1. Problem

Producing a good educational explainer video takes 8–20 hours: scripting, storyboarding,
animating, recording, editing, thumbnailing. That cost is why most technical topics have
either no video or a bad one. Manim collapses the animation cost for people who can write
Manim, but writing Manim is itself slow and requires visual iteration.

A previous scrappy version of this system proved the concept works and failed on
reliability: generated code compiled but produced visually broken frames, audio drifted
out of sync with animation, and every change forced a full re-render.

## 2. What we're building

A pipeline that takes a topic and a channel style config and produces a publish-ready
video: Manim animations, AI voiceover synced to the frame, captions, chapters, thumbnail,
title, and description.

Then a thin orchestration layer that runs this on a schedule across multiple YouTube
channels, each with its own subject vertical, visual identity, and topic backlog.

## 3. Goals

**G1 — Visual correctness by construction.** A rendered video must never contain
off-screen text, overlapping mobjects, or unreadably small type. This is enforced
structurally, not by hoping the LLM gets it right.

**G2 — Frame-accurate audio sync.** Narration and animation are locked. No drift, no
awkward silence, no visual finishing four seconds before the sentence does.

**G3 — Incremental cost.** Changing one narration line re-renders that beat only.
Full re-renders are an exception, not the workflow.

**G4 — Factual reliability.** Especially for problem-solution content, where a wrong
answer is unrecoverable. Verified before render, not after upload.

**G5 — Unattended operation.** A channel can produce and publish daily without a human
in the loop — after that channel's quality has been established with a human in the loop.

## 4. Explicit non-goals

- **Not a general video editor.** No timeline UI, no manual keyframing.
- **Not a Manim replacement.** Anything outside the component library's expressive range
  is out of scope for automated generation.
- **Not live-action, stock footage, or talking-head.** Manim, text, and generated
  diagrams only.
- **Not multi-language at launch.** English first. The beat model makes dubbing tractable
  later; don't build for it now.
- **Not maximum throughput.** Deliberately. See §8.

## 5. Users

**Primary — operator (you).** Runs the pipeline, reviews scripts, approves publishes,
tunes channel configs. Wants: low babysitting, fast failure signals, cheap iteration.

**Secondary — viewer.** Student or curious person on YouTube. Wants: correct information,
clear visuals, a voice that isn't grating, and a video that respects their time. Does not
know or care that it's generated — and if the output makes that obvious, we've failed.

## 6. Content verticals (initial)

| Vertical | Shape | Special requirements |
|---|---|---|
| Science explainers | 6–10 min, concept-driven | Diagram-heavy components, physical intuition |
| CS fundamentals | 5–8 min, mechanism-driven | Code display, data-structure visualization, step traces |
| JEE Advanced solutions | 4–7 min, problem-driven | **Symbolic answer verification mandatory**, rigid template |

JEE gets its own pipeline branch. The content is more templated (given → approach →
derivation → answer), the accuracy bar is absolute, and the visual needs are narrower.
Treating it as "just another channel config" is a mistake.

## 7. Definition of a good video

This is the acceptance bar. A video ships only if all of these hold.

**Correctness**
- [ ] Every factual claim in narration is defensible; sources logged
- [ ] Every equation renders correctly and is mathematically valid
- [ ] For problem videos: final answer verified symbolically or against a key

**Visual**
- [ ] No element crosses the safe-area boundary
- [ ] No unintended overlap between visual elements
- [ ] All text ≥ minimum legible size at 1080p and at mobile thumbnail scale
- [ ] Consistent palette, typography, and motion language per channel

**Audio**
- [ ] Narration and animation aligned within ±150 ms at every beat boundary
- [ ] No dead air > 1.5 s, no clipped or rushed segment
- [ ] Consistent loudness (target −14 LUFS integrated)

**Pedagogical**
- [ ] Opens with a concrete hook, not a definition
- [ ] One idea per beat; no beat exceeds ~25 s of narration
- [ ] Builds visually — new elements reference what's already on screen
- [ ] Ends with a resolution, not a trailing thought

**Packaging**
- [ ] Title is honest and searchable; not clickbait
- [ ] Thumbnail readable at 168×94 px
- [ ] Chapters and captions generated from beat timings

## 8. On scale, honestly

YouTube's inauthentic-content policy explicitly targets mass-produced, repetitive,
templated content. A fleet of channels running the same pipeline with a swapped config
file is exactly the pattern being enforced against. Confirm current policy language
before building the fleet layer.

The implication is not "don't build this." It's that **throughput is not the moat.**
Each channel needs genuine editorial identity — distinct visual language, real
pedagogical structure, human judgment on script selection — or it gets demonetized and
the entire fleet layer becomes worthless.

Design consequence: channel configs carry substantial differentiation (visual theme,
narration voice and register, pacing profile, component preferences, opening formula),
and human script review stays in the loop far longer than is technically necessary.

**One excellent channel is the goal. The fleet is the stretch.**

## 9. Success metrics

**Phase-gate metrics (build quality)**
- Layout validation pass rate on first attempt: target > 90%
- Beats requiring the freeform Manim escape hatch: target < 10%
- Full-video render wall time (8 min video, 8 cores): target < 15 min
- Cache hit rate on iterative edits: target > 95%
- Human script edits per video: trending toward zero

**Product metrics (does anyone watch)**
- Average view duration > 50%
- No community-guideline or monetization strikes
- Comment sentiment not dominated by "this is AI slop"

The last one is the real test. Track it manually and honestly.

## 10. Constraints

- **Compute:** single workstation initially. CPU-bound rendering. No GPU dependency.
- **Budget:** student budget. Self-hosted TTS is preferred over per-character APIs at
  fleet scale; verify current pricing for any paid tier before committing.
- **YouTube API quota:** uploads are expensive against the default daily project quota —
  verify current numbers, but expect a single-digit ceiling of uploads per day per
  project. This binds the fleet before compute does.
- **Time:** this is one project among several. Every phase must produce something
  independently usable, so an interruption doesn't strand the work.

## 11. Key risks

| Risk | Impact | Mitigation |
|---|---|---|
| YouTube policy action on automated fleet | Kills the fleet thesis | Channel differentiation from day one; one channel proven before scaling |
| Wrong answer published in a solution video | Reputational, unrecoverable | Symbolic verification gate; no publish without it |
| LLM output outside component expressiveness | Quality ceiling | Escape hatch + component library growth driven by observed misses |
| Generated videos read as obviously synthetic | No audience | Human review gate; retention as ground truth; kill channels that don't land |
| Manim CE version drift breaking components | Build breakage | Pin version; components have snapshot tests |
| Scope creep into a general video tool | Never ships | Non-goals in §4 are binding |

## 12. Open questions

- Does a single narration voice per channel, or a fixed pair, read better long-term?
- Should the script gate be CLI-only or a small local web review UI?
- Is Kokoro's quality sufficient for a flagship channel, or is it draft-tier only?
- How much visual variety is needed before a channel starts feeling repetitive to a
  subscriber who watches 10 videos?
- Do JEE solution videos need a human subject-matter check regardless of symbolic
  verification?
