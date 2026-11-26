# Run 3: real animations, real voice (2026-11-26)

Branch `research/run3-animations`. Brief: `.lab/run3/BRIEF.md` (not in git).

## Outcome

PENDING-OUTCOME

## 1. Voice

Kokoro `af_heart` is the default voice, at 1080p30. Every one of the 13 English Kokoro voices read the same
maths-heavy script ("x-squared plus 3x ... negative 1.5 ... 8 thirds, about 2.67"). Each clip was measured
for length, the longest silence, peak level and clipping, then transcribed locally with faster-whisper and
scored as word error rate after normalising spoken numbers.

| voice | length s | longest gap s | peak dBFS | clipped | WER |
|---|---|---|---|---|---|
| **af_heart** (default) | 14.4 | 0.64 | -3.7 | 0 | 0.067 |
| af_bella / af_aoede / af_kore / af_sarah | 14.1-14.8 | 0.36-0.68 | -6.3 to -3.7 | 0 | 0.067 |
| af_nicole | 22.4 | **1.68** | -5.8 | 0 | 0.067 |
| am_michael | 15.5 | 0.74 | -5.1 | 0 | **0.133** ("a half" heard as "one by two") |
| am_fenrir / am_puck / bf_emma / bf_isabella / bm_george / bm_fable | 12.7-15.7 | 0.34-0.72 | -8.6 to -2.1 | 0 | 0.067 |

No voice clips. The one shared word error is the transcriber's, not the voice's. af_nicole is slow and
breathy. Clips: `.lab/run3/voices/*.mp3`, numbers in `voices.json`.

Narration is written as spoken words ("x squared over four, plus one", "five and a quarter"). Every final
video is checked the same way (`research/verify_video.py`):
- Kokoro's word timings are compared with whisper's on the encoded audio.
- Each cue is checked to change the picture on its word.
- Frame count is compared with the audio.
- Every beat join is compared pixel by pixel.

## 2. The animation contract

`docs/research/ANIMATIONS.md` is the contract. CLAUDE.md "Animations (run 3)" summarises it. In short:
- An animation is one component with **phases**. Each phase is one beat. The model, or a spec, picks only
  the setup (function, range, point...) and which words cue each moment. Nothing positional or stylistic
  comes from the spec.
- Validated params refuse a setup whose picture would be false, with a message that names a fix that works.
- Every number on screen is computed by code from the setup.
- Each phase rebuilds its start state from the setup, so the last frame of one phase and the first frame
  of the next match to the pixel.
- Cues are words from the narration. Moments land on those words' Kokoro timings (`cue_timeline`).
- Registry tests cover every animation: examples, stress cases, phase seams, cue timing, snapshots.

New this run:
- The SecantToTangent pilot proved the contract at full quality first. Its encoded 1080p video has 1197/1197
  frames, 0 seam pixels over threshold, and every cue on its word.
- A shared `readout()` puts digits on the label's baseline. All three reviews found digits 5-6 px low.
- A seam test now checks the first frame each phase writes. That still cannot see a frame written during the
  hold before a cued first moment. Only the encoded-video check caught LimitAtHole blanking there, so that
  check is the acceptance gate.

## 3. The animations

Three animations, each written from scratch (no 3b1b code) and reviewed by a fresh reviewer who tried to
break it:

| component | phases (moments) | review | what round 1 caught |
|---|---|---|---|
| `SecantToTangent` | pose (curve, point, question) · secant (second, line, rise_run) · shrink (slide, tangent) · name (limit, value) | round 1 FAIL, round 2 **PASS** | wrong slopes (sin(1000x) at 0.005 read 275.27, want 283.66); a slide that stopped before the readout showed f'(a); a curve that left the frame over a peak; a corner slope drawn as a derivative |
| `RiemannSum` | pose (curve, interval, question) · boxes (rects, sum, gap) · refine (split, more, settle) · name (sigma, integral) | round 1 FAIL, round 2 REVIEW-RS | a false integral was drawn for poles and narrow spikes; the "gap" number was not the red area drawn; Σ = 5.20 beside ∫ = 5.25 read as a contradiction |
| `LimitAtHole` | pose (curve, hole, question) · approach (points, slide, agree) · name (limit, value) | round 1 FAIL, round 2 REVIEW-LH | `sin(pi/x)` showed a false "lim = 0"; some refusal messages were untrue |

The integrator found one more problem in the encoded 1080p video: at each approach → name join of LimitAtHole,
both f-readouts blanked for more than a second. Its writer fixed it in round 2.

Each component refuses setups whose picture would be false, and each refusal names a fix that works. Examples:
- **RiemannSum:** f negative or undefined on [a, b]; a pole or narrow spike; rectangles thinner than 2 px; values
  that would read 0.
- **LimitAtHole:** no hole at a; a jump; oscillation ("does not settle"); "cancels too finely"; "still drifting".

Stated limits that the reviews accepted or judged: REVIEW-LIMITS.

**Not built:** the other ~9 animations in the brief's dozen. Each animation cost about 1% of the weekly allowance
across writer and review rounds, and the brief stops at 70%. The contract, the writer and reviewer briefs
(`.lab/run3/WRITER_BRIEF.md`, `REVIEWER_BRIEF.md`) and the seam/cue checks are ready for the next fan-out.

## 4. Videos: my specs vs the local model

PENDING-VIDEOS

## 5. Speed

All timings are from this laptop at full quality (1080p30). Each run's command and log are in
`.lab/run3/speed/*.json`. The stage split comes from `.lab/run3/tools/stage_time.py`, which wraps the pipeline's
stages with timers and changes no engine code. **Unconfirmed:** the 3-4 minute numbers below were taken with
review agents and the GPU model run working at the same time, so they are upper bounds.

**Per video (3-4 minutes, 12-14 beats):**

| stage | time | notes |
|---|---|---|
| model (local 4B, best-of-4 + repair) | 144-246 s | the model itself is 48-73 s; checking the candidate beats is 95-172 s at 2 score workers |
| validate spec | 9-18 s | layout probes, TeX |
| voice (Kokoro, CPU) | about 50 s cold, 0.6 s cached | all beats are voiced before any beat renders |
| render (4 workers) | 45-63 s | |
| assembly | 26-32 s | per-beat mux plus concat |
| **render total** | 84-109 s with voice cached; 142-153 s cold | |

**Time to first playable beat** is the whole model stage, then validation, then voicing every beat, then
rendering beat 1: about 4-5 minutes from request. Once the spec exists, beat 1 is ready 24-35 s into the
render with voice cached. The cheap win not taken is to stream: render and voice beat 1 while the model writes
beat 2. That changes the harness/pipeline contract, so it is left for a decision.

**Cheap wins taken, measured on the 40 s pilot:**

| change | before | after |
|---|---|---|
| lossless parallel mux (86f3ee0) | 12.0 s all cached | 9.7 s |
| 4 render workers instead of 1 | 41.4 s | 35.1 s |

**Frame rate: 30 fps.** 60 fps cost 60.0 s against 41.4 s for the pilot (1.45x the whole render; about 1.9x the
render stage). In the full-size frame review, 30 fps showed no stepping in the slides and refinements. That
was judged by eye; per-object motion per frame was not measured.

Biggest remaining costs, in order:
1. Checking the model's candidate beats: it renders probes for every candidate. Use more score workers when
   the machine is free.
2. Voicing every beat before any render.
3. Assembly time growing with video length.

## 6. Agents

The lead ran on Opus 5.5 at xhigh effort. Every agent was given its model and effort explicitly, and each one
reported what it ran on.

| agent | job | model, effort |
|---|---|---|
| writer-riemann | RiemannSum, rounds 1-2 | Opus 5.5, medium |
| writer-limit | LimitAtHole, rounds 1-2 | Opus 5.5, medium |
| reviewer-secant | SecantToTangent review, rounds 1-2 | Opus 5.5, medium |
| reviewer-riemann | RiemannSum review, rounds 1-2 | Opus 5.5, medium |
| reviewer-limit | LimitAtHole review, rounds 1-2 | Opus 5.5, medium |

No Sonnet or Haiku agents were needed. The mechanical work (renders, measurement, queue) ran as scripts.
There were no worktrees: each writer owned one component file, and the lead owned the shared files and every
commit.

## 7. Follow-ups and watch list

- **The calculator checks only the final answer.** The model's intermediate claims are unchecked. In R3-anim,
  a SplitCompare beat showed "f'(1) = 1/4" in a lesson whose answer is 1/2. In R3-anim-b, an EquationDerivation
  showed "= 3.75" for ∫₀³ (x²/4 + 1) dx. Each beat's own maths needs checking against the calculator.
- **EquationDerivation accepts English words as steps.** The model wrote derivation steps as sentences, and the
  validator passed them. `chalkdust/scenes/components/equation_derivation.py`.
- **The plan's calculation was never a computation.** All three R3-anim-b plans wrote a bare number. 412102e
  now computes the answer from the animation's setup. A plan with no animation, or with two, still echoes the
  model.
- **The structure check `beat_count_5_to_9` fails by design under anim=1.** It has 8-11 beats plus expanded
  phases. `research/content.py`.
- **`_glyph_paths` caches a failure.** A torn SVG left by a crashed run produced a spurious "U+0031 cannot be
  shaped" until the work dir was cleared. `chalkdust/scenes/theme.py:207`.
- **Agents leave TeX caches behind.** Reviewers' renders left `media/` and `work/manim` caches in the repo root.
  They are git-ignored, but they grow.
- **Voice-stack licences, noted for later as the brief asks.** `kokoro` 0.9.4 and `misaki` 0.9.4 are Apache-2.0,
  but `phonemizer-fork` 3.3.2 is GPL and `espeakng-loader` 0.2.4 ships `espeak-ng.dll`, which is GPL-3.0. This
  is from installed package metadata, and nothing has been checked beyond it.
- **The phase-seam unit test cannot see the hold before a cued first moment.** Only the encoded-video check
  (`research/verify_video.py`) caught LimitAtHole's blank readouts there. Every new animation should pass
  that check on a cued render.
- **The cue metric misses slow fades.** A 6-frame window at a 24-level threshold flagged 9 of 59 cues. All 9
  were checked over 30 frames and land on their words. Widen the metric before relying on it unattended.
