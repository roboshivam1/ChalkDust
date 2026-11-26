# Run 3: real animations, real voice (2026-11-26)

Branch `research/run3-animations`. Brief: `.lab/run3/BRIEF.md` (not in git).

## Outcome

**Done:**
- **Two Calc 1 animations reach the bar:** `SecantToTangent` and `RiemannSum`. Each passed independent review
  on round 2.
- **A third, `LimitAtHole`, is left out and written up (§3).** It can print a wrong last digit for a slowly
  converging limit. It failed round 2, then failed again after I changed its precision rule. Its code and tests
  stay. The limit lessons' own limits (2, 1, 1/4) are exact and verified. The model is no longer offered it
  (`research/animate.py` `HELD_BACK`).
- **Three 3-4 minute Calc 1 videos at 1080p30 with Kokoro `af_heart`, made twice** (my spec and the local 4B
  model). All six are checked on the encoded video, and all six show and say the right answer.
- **A contract for animation components** that a fan-out can follow. The checks caught real defects: two
  seams, one of them only in the encoded video, plus false pictures in every pilot.

**Not done:** the other ~9 animations of the dozen. At about 1% of the weekly allowance per animation (writer,
reviewer and a second round each), they did not fit under the 70% stop. The fan-out is set up for a next run.

**The model's videos are clearly weaker than mine.** Its final answers and its animation narration are right,
because code decides them. Its other beats contain false maths that nothing checks: the integral video says
"exactly 3.75" before showing 5.25. That is the next thing to fix (§7).

**Speed:** about 3.4-5 minutes from request to first playable beat. Of that, 2-3 minutes are the model; the rest
is validation, voicing every beat, and rendering beat 1. A whole video renders in 2.3-3 minutes cold. 30 fps
stays: 60 fps cost 1.45x.

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

Three animations were written from scratch (no 3b1b code), and each was reviewed by a fresh reviewer who tried to
break it:

| component | phases (moments) | review | what round 1 caught |
|---|---|---|---|
| `SecantToTangent` | pose (curve, point, question) · secant (second, line, rise_run) · shrink (slide, tangent) · name (limit, value) | round 1 FAIL, round 2 **PASS** | wrong slopes (sin(1000x) at 0.005 read 275.27, want 283.66); a slide that stopped before the readout showed f'(a); a curve that left the frame over a peak; a corner slope drawn as a derivative |
| `RiemannSum` | pose (curve, interval, question) · boxes (rects, sum, gap) · refine (split, more, settle) · name (sigma, integral) | round 1 FAIL, round 2 **PASS** (4 minors) | a false integral was drawn for poles and narrow spikes; the "gap" number was not the red area drawn; Σ = 5.20 beside ∫ = 5.25 read as a contradiction |
| `LimitAtHole` | pose (curve, hole, question) · approach (points, slide, agree) · name (limit, value) | round 1 FAIL, round 2 FAIL, round 3 (my re-approach) FAIL: **left out** | `sin(pi/x)` showed a false "lim = 0"; some refusal messages were untrue |

The integrator found one more problem in the encoded 1080p video: at each approach → name join of LimitAtHole,
both f-readouts blanked for more than a second. Its writer fixed it in round 2.

**Why LimitAtHole is left out.**
- **Round 2 (writer's fix):** fixed every round-1 finding, including oscillations now refused. It failed on a
  finding round 1 missed: the readout decimals came from the window height alone. A tall window printed
  "lim ≈ 0" for 1/3 and "lim ≈ 1" for ln 2.
- **Round 3:** under the brief's rule, I changed the precision approach in 17f6622: always show 3 significant
  figures of a limit that is clearly not 0. That fixed those cases but showed digits finer than the computed
  limit's own error. Under √|x| convergence, `(5.5^sqrt(abs(x)) - 1)/sqrt(abs(x))` reads "≈ 1.71" for
  ln 5.5 = 1.7047. That happened in 33 of 299 setups in the reviewer's sweep (`.lab/run3/agents/review_lh/`).
- **What would fix it:** print only as many digits as the computed L's error bound supports (the reviewer
  suggests 0.5·10^-dy > the agreement bound), and refuse what cannot be shown. Then run one fresh review.
- **Two other open items:** some refusal messages for a tiny L on a steep slope name a fix that fails; and the
  approach → name join moves the readout rows 1-2 px after very deep slides.

Each component refuses setups whose picture would be false, and each refusal names a fix that works. Examples:
- **RiemannSum:** f negative or undefined on [a, b]; a pole or narrow spike; rectangles thinner than 2 px; values
  that would read 0.
- **LimitAtHole:** no hole at a; a jump; oscillation ("does not settle"); "cancels too finely"; "still drifting".

Known limits, as the final reviews judged them (none blocking):
- **RiemannSum.** One minor is fixed (156ac1c): a gap that read 0.000000 beside drawn slivers is now refused.
  Three remain:
  - A spike about 0.01 px wide passes every sampled check. `1 + 1000*exp(-3e9*(x - 1.2001)^2)` shows
    ∫ = 2.00 where the exact value is 2.0324. Fix: use sympy's exact integral where one exists.
  - `1e15*(x^2+1)` is refused with the wrong reason. The cause is double precision, so it should say
    "scale f down".
  - The settle pulse on the red slivers at n = 64 is barely visible.
- **LimitAtHole.**
  - An oscillation below a pixel (e.g. 3e-5·sin(pi/x)) is shown as its limit.
  - A jump hidden under a tiny oscillation can pass, but the hold check keeps it below the last digit shown,
    so no readout is wrong.
  - `exp(300x)/exp(300x)` at 1 is labelled "f(1) is undefined", because a part above 1e100 counts as
    undefined. **This one draws a false sentence.**
  - Two refusal messages suggest the wrong fix: `(exp(x)-1)/x` on [-1e-5, 1e-5], and
    `sin(x)/x + exp(-1e-18/x^2)`.
- **Beat joins in the encoded video.** At full 1080p the first 2 frames of a beat show encoder speckle (up to
  ~2,000 px at a max of 64 levels) where the drawing itself is identical. Each beat's clip starts on a fresh
  keyframe. The seam check measures at 640×360, where this reads 0.

**Not built:** the other ~9 animations in the brief's dozen. Each animation cost about 1% of the weekly allowance
across writer and review rounds, and the brief stops at 70%. The contract, the writer and reviewer briefs
(`.lab/run3/WRITER_BRIEF.md`, `REVIEWER_BRIEF.md`) and the seam/cue checks are ready for the next fan-out.

## 4. Videos: my specs vs the local model

Three Calc 1 lessons, each made twice:
- **My spec:** `examples/calculus/ref_*.json`.
- **The local model:** `qwen3.5-4b`, untuned, thinking off, best-of-4 plus one repair round, the calculator on,
  `anim=1`. That is experiment **R3-anim-c** at 412102e, records in `research/results/R3-anim-c.jsonl`.

All six videos render at 1080p30 with Kokoro `af_heart` and are checked on the encoded MP4. The videos are in
`.lab/run3/videos/` and are not in git. Side-by-side sheets (8 evenly spaced frames, my spec on top, the model
below) are in `.lab/run3/stills/pairs/pair_{derivative,limit,integral}.png`.

| | length | beats (animation) | answer shown / spoken | phase joins (px > 24) | cues on their word | Kokoro vs whisper word start, median |
|---|---|---|---|---|---|---|
| derivative, mine | 3:44 | 14 (8) | 0.5 / 0.5 | 0 ×6 | 18/18 | 0.003 s |
| derivative, model | 3:22 | 11 (4) | 0.5 / 0.5 | 0 ×3 | 7/7 | 0.018 s |
| limit, mine | 3:07 | 14 (9) | 2 / 2 | 0 ×6 | 21/21 | 0.003 s |
| limit, model | 3:00 | 10 (3) | 2 / 2 | 0 ×2 | 6/6 | 0.030 s |
| integral, mine | 3:03 | 12 (8) | 5.25 / 5.25 | 0 ×6 | 20/20 | 0.015 s |
| integral, model | 3:29 | 11 (4) | 5.25 / 5.25 | 0 ×3 | 10/10 | 0.018 s |

The "cues on their word" column was checked twice:
- by `research/verify_video.py`;
- by hand for slow fades: 12 cues the 6-frame metric flagged, each checked over 30 frames.

In every model video, every beat scores level 3 and clean, and every spec passes the CLI.

**What the model does well.** It picked the right animation for each lesson and a sensible setup (the
question's own function, point and interval). It also narrated the animations with the numbers the screen
shows: "six point one six", "a gap of zero point nine one", "one point zero zero down to zero point five zero",
"two point zero zero". Those numbers come to it as FACTS.

**Where it falls short of my specs:**
1. **False intermediate claims** that no check catches. The integral video says "evaluating at three and zero
   confirms the total area is exactly 3.75" two beats before the answer box shows 5.25. It also says the curve
   "starts at zero and ends at two and a half" (it runs from 1 to 3.25). Its derivation steps use the wrong
   sample points, `3i/n + 3`. The derivative video's last derivation is English sentences with false algebra
   ("simplifies to four h over four h").
2. **One animation per lesson, and fewer cues.** The model used one animation per lesson (3-4 phase beats); my
   specs use 2-3 runs of the animation (8-9 beats). The model cued 6-10 moments against my 18-21. Uncued
   moments fall back to sentence timing.
3. **Filler.** Examples: "where the derivative lives and breathes", "for precision today in calculus class
   right now".

**The answer rule needed a fix to be real.** Under R3-anim-b, the plan's "calculation" was the model's own
number in all three lessons ("1/2", "2", "3.75"). The integral video said and showed 3.75 while its RiemannSum
showed 5.25. Since 412102e, the calculator computes the lesson's quantity from the animation's setup, e.g.
`integral(x^2/4 + 1, 0, 3)`. In R3-anim-c the plan wrote 3.75 again, and the calculator overrode it.

**The harness picks up new components.** `research/tests/test_animate.py` fails if a registered animation has
no card. All three animations reached the model through their cards, and the model used each one.

## 5. Speed

All timings are from this laptop at full quality (1080p30). Each run's command and log are in
`.lab/run3/speed/*.json`. The stage split comes from `.lab/run3/tools/stage_time.py`, which wraps the pipeline's
stages with timers and changes no engine code. **Unconfirmed:** the 3-4 minute numbers below were taken with
review agents and the GPU model run working at the same time, so they are upper bounds.

**Per video (3-4 minutes, 12-14 beats):**

| stage | time | notes |
|---|---|---|
| model (local 4B, best-of-4 + repair) | 115-195 s | R3-anim-c. The model itself is 47-61 s; checking the candidate beats is 67-134 s at 2 score workers |
| validate spec | 9-35 s | layout probes and TeX: 9-18 s on my specs, 27-35 s on the model's (new TeX) |
| voice (Kokoro, CPU) | 46-57 s cold, 0.6 s cached | all beats are voiced before any beat renders |
| render (4 workers) | 41-67 s | |
| assembly | 24-32 s | per-beat mux plus concat |
| **render total** | 139-178 s cold (model specs); 84-109 s with voice cached | |

**Time to first playable beat** is the model stage, then validation, then voicing every beat, then rendering
beat 1: 115-195 s + 88-105 s, or **3.4-5 minutes from request**. The model stage is from R3-anim-c; the render
half is the model's own specs with cold voice (`.lab/run3/speed/model_*_stages.json`). The cheap win not taken is to stream: render and voice beat 1 while the model writes
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
- **A `latex.exe` can hang without writing a log.** One hung for 34 minutes during a review's test run, until
  it was killed. Nothing times out a TeX call.
