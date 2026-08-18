# CHALKDUST — Operations

Fleet configuration, publishing mechanics, and the constraints that bind before compute
does.

> **Verify all API numbers and policy language before building against them.** Quotas,
> pricing, and enforcement policy all change. The figures below are directional, not
> authoritative.

---

## 1. Channel configuration

A channel is not a color swap. Per `PRD.md` §8, differentiation is a survival
requirement, so the config carries real weight.

```yaml
slug: cs_fundamentals
name: "Under the Hood"
vertical: cs

identity:
  theme: cs_dark_mono           # palette, typography, motion language
  opening_formula: concrete_bug_first
  closing_formula: one_line_takeaway
  register: precise_casual      # narration voice and diction

voice:
  backend: kokoro
  voice_id: am_michael
  rate: 1.0
  pause_profile: technical      # longer pauses after definitions

pacing:
  target_beat_seconds: [10, 20]
  target_duration_minutes: [5, 8]
  density: high                 # information per minute

components:
  preferred: [CodeWalk, DataStructureViz, StepTrace, BoxFlow]
  avoid: [VectorField, GeometryConstruct]

publishing:
  cadence: "0 14 * * 1,3,5"     # Mon/Wed/Fri 14:00 IST
  visibility: public
  playlist_strategy: by_series
```

**Test for adequate differentiation:** put two channels' videos side by side with the
audio off. If a viewer couldn't tell them apart, the configs aren't different enough.

## 2. Topic backlog

Each channel keeps a backlog with state. Sources: manual entry (best early), search-volume
research, gaps in existing coverage, viewer comments once there are any.

```yaml
- topic: "Why hash maps degrade to O(n)"
  status: queued          # queued | scripted | rendered | published | rejected
  priority: 3
  angle: "collision chains, worst case, and why it matters in practice"
  prerequisites: [hashing_basics]
```

**Dedupe against published titles** using embedding similarity, not string matching. The
failure mode is publishing a near-duplicate three months apart — which is both bad for
viewers and exactly the repetitive-content signal you don't want to emit.

## 3. YouTube API constraints

**This binds the fleet before compute does.** Uploads are expensive against the daily
project quota — expect a single-digit ceiling of uploads per day per Google Cloud project.

Key point: **the quota is per project, not per channel.** Attaching ten channels to one
project doesn't multiply it.

Options when you hit it:
- Request a quota increase — a real audit process, slow, often denied for automated
  upload use cases
- Spread channels across separate GCP projects — check current Terms of Service before
  relying on this; it can read as quota circumvention
- Publish less per channel — usually the right answer anyway

**Implementation notes**
- Use resumable uploads. Non-resumable uploads on a home connection will fail eventually.
- Per-channel OAuth refresh tokens. Store encrypted, never in the repo.
- Refresh tokens expire on password change and inactivity. Build a health check that
  fails loudly rather than silently skipping a publish.
- Uploads process asynchronously — a successful API response is not a live video. Poll
  processing status before marking the job complete.

## 4. Policy risk

YouTube's inauthentic-content policy targets mass-produced and repetitive content. A
fleet of channels sharing a pipeline is squarely in scope. Read the current policy text
directly before the fleet phase; enforcement language has shifted more than once.

**What reduces exposure**
- Genuine per-channel visual and editorial identity
- Human review on scripts, kept in the loop past the point of technical necessity
- Real pedagogical structure rather than a fixed template with substituted nouns
- Channel-level disclosure that content is AI-assisted, where appropriate
- Varied video structure within a channel

**What increases it**
- Identical intro/outro across channels
- High-volume publishing on a rigid schedule
- Topics selected purely by search volume
- Zero human touch anywhere in the loop

**Operational stance:** treat each channel as independently expendable. Shared
infrastructure, isolated identity, isolated credentials. One channel's strike shouldn't
be able to take down the rest.

## 5. Cost model

Estimate per 8-minute video. **Fill in real numbers once you've measured — these are
placeholders for the shape of the model, not quotes.**

| Item | Estimate | Notes |
|---|---|---|
| LLM (outline, narration, spec, critic) | $__ | Caching cuts dev cost sharply |
| TTS | $0 self-hosted / $__ hosted | Dominant variable cost on hosted voices |
| Research/search | $__ | Small |
| Compute | electricity | Sunk if it's your own machine |
| **Total** | **$__** | |

Track actual cost per video from Phase 4 onward. At 3 channels × 3 videos/week, small
per-video costs compound into a real monthly number, and you should know it before it
surprises you.

Self-hosted TTS is the biggest lever. Consider hosted voices for a flagship channel only.

## 6. Monitoring

Minimum viable, added at Phase 7:

**Per job:** stage, duration, cache hit rate, degraded beat count, escape-hatch usage,
total cost.

**Per channel:** publish success rate, credential health, backlog depth, retention
trend.

**Alerts that matter**
- Publish failed
- OAuth credentials expired
- Degraded beats exceeded threshold in a published video
- Cost per video above budget
- Any policy notification from YouTube

Alerts go somewhere you'll actually see them. You already have a Telegram-based
notification path; reuse it rather than building a dashboard nobody opens.

## 7. Runbook stubs

**Upload failed** → check credential health, then processing status; retry is safe due to
resumable upload; if quota-exceeded, requeue for the next window.

**Video published with a visual bug** → unlist immediately, fix the component, re-render
the affected beats only, re-upload as a new video, delete the original. Log the failure
mode in the component's stress fixtures so it can't recur.

**Factual error reported in a published video** → unlist immediately. Correct, re-render,
re-publish with a note. Add the case to the verification suite. **Do not leave it up while
you decide** — especially on JEE content, where a wrong answer costs someone real marks.

**Policy strike** → stop all publishing on all channels. Read the specific violation.
Assume it applies to every channel until proven otherwise.

## 8. Manual override

Every automated stage needs a manual entry point. You will need to hand-write a script,
force a specific component, or publish something the gate rejected.

```
chalkdust script --edit <job_id>
chalkdust render --beats b03,b07 --force
chalkdust publish --skip-gate <job_id>    # logged, requires confirmation
```

An automation you can't intervene in is an automation you'll eventually abandon.
