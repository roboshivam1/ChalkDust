# Methodology: making small local models good at ChalkDust videos

Written 17 November 2026 on branch `research/small-local-models`. This is the method;
`FINDINGS.md` has the numbers that support or kill each part of it. The code is in
`research/` and every claim about behaviour points at the module that implements it.

## The idea in one paragraph

A small model is bad at long, open-ended writing but can be good at short, tightly
specified filling, if three things are true. It never sees more than it needs for one
step. Its output is constrained to a shape that cannot be malformed. And a
deterministic checker, not the model, decides what is good. ChalkDust already has the
checker: the validator ladder (schema, semantic, geometric probe). So the method splits
the video into small jobs, puts the validator inside the generation loop rather than at
the end, and moves anything that must be *true* (the final number, the maths) away from
the model's free text and into places a checker or a calculator can see. The same
validator that picks the best sample at inference is the filter for training data and
the reward for RL. One signal is used three times.

## 1. How the job is split

```
student request
   │
   ▼
[1 PLAN]  one call (or k calls + vote) ── grammar: plan_schema
   │      sees: system prompt with lesson rules + one-line card per allowed component
   │            + the student's request
   │      makes: title, objective, key_insight, misconception, final_answer,
   │             (answer_calc), beats[ {purpose, idea, component} ] (5-9)
   ▼
[2 CALC]  no model: research/generate.safe_calc evaluates answer_calc with an
   │      AST allowlist (research/safe_math.py) -> a number the beats are told
   ▼
[3 FILL]  one call per beat, n samples in parallel ── grammar: beat_schema(component)
   │      sees: the request, the plan header, the beat outline with this beat marked,
   │            the previous beat's narration, this beat's purpose and idea,
   │            the FULL card of this one component only, (validator feedback)
   │      makes: {narration, params}
   ▼
[4 VALIDATE] every sample, in a process pool (research/score.py)
   │      rung 1  BeatSpec + the component's Params model + control characters
   │      rung 2  validate_semantic: duration at 160 wpm, capacity, LaTeX compiles,
   │              regions, carry_in
   │      rung 3  repair_beat: geometric probe, mechanical repair
   │      flags   (research harness, not in the validator yet): LaTeX/symbols in
   │              narration (the TTS reads them aloud), LaTeX drawn literally in a
   │              prose field, a stray $ in prose
   ▼
[5 SELECT] best sample by (level, fewest flags, content floor, fewest findings,
   │       fewest escape repairs) (score.selection_key)
   ▼
[6 REPAIR] if the best is not clean: the validator's typed refusal, rung-1 errors
   │       and flags go back to the model as text; n new samples; up to R rounds
   ▼
[7 ASSEMBLE] VideoSpec from the chosen beats -> `chalkdust validate` (the real CLI,
   │         whole-video checks) -> `chalkdust render`
   ▼
 MP4
```

Why this split:

- **The plan is the reasoning job; the fill is the formatting job.** The plan decides
  what the lesson says and what the answer is. A fill only has to express one idea
  through one component. Different skills, so they can be done by different models
  (split pipeline, section 5).
- **A fill sees one component card, not nineteen.** The full catalog with every Params
  schema is about 21.6k characters. A fill prompt carries one card. That keeps the
  context short for small models and stops one component's fields leaking into
  another's.
- **Grammar-constrained decoding per component.** `research/catalog.beat_schema` turns
  the component's own pydantic Params model into a JSON schema, and llama.cpp compiles
  it to a GBNF grammar. The model cannot emit an unknown field, a missing required
  field, or broken JSON. That is not the same as valid: pydantic validators (lengths,
  cross-field rules, LaTeX that compiles) are still checked at rung 1-2, which is why
  the validator stays in the loop.
- **One-shot is kept as the baseline.** `generate_oneshot` writes the whole video in
  one call against a union schema. If staged generation does not beat it, the split
  is not worth its extra calls.

## 2. Where the validator sits

The validator is used at four points, and it is the same code each time:

| Point | What it decides | Code |
|---|---|---|
| Inner loop, per sample | which of n samples is kept | `score.score_beat`, `selection_key` |
| Repair | what the model is told is wrong | `generate._feedback` |
| Whole video | whether the assembled spec passes the real CLI | `score.validate_spec_cli` |
| Training data | which teacher samples become SFT targets | `research/build_sft.py` |
| RL (planned) | reward = validator gate x content floor | section 6 |

The validator is deterministic and cheap compared with generation: about 1-5 s per
novel beat, mostly LaTeX compile, cached after that (measured, FINDINGS.md). That is
what makes best-of-n affordable.

**What the validator cannot see.** Found in this run, with examples in FINDINGS.md:

1. LaTeX in narration. The validator only checks the text it renders; narration goes
   to TTS, which reads `\frac{1}{6}` aloud as "backslash frac". The harness flags it.
2. LaTeX in prose fields, drawn literally on screen. Only ProblemStatement supports
   inline `$...$`. The harness flags it.
3. Maths that is wrong but valid: a correct-looking derivation with a wrong unit, or
   an AnswerBox that disagrees with the narration. The validator checks that the maths
   *renders*, not that it is *true*. The content checks in section 3 cover the
   answer; nothing yet covers every intermediate step.

Items 1 and 2 are cheap to move into the validator (`validate/semantic.py`) as new
refusal kinds. That is a recommendation for the owner, not done here: this run does
not edit the `chalkdust/` package.

## 3. How quality is measured

Hard checks decide; model judgement does not. Every number in FINDINGS.md comes from
one of these, recomputed from the saved spec by `research/analyze.py`:

| Metric | What it is | Code |
|---|---|---|
| beat valid | passes rungs 1-3 | `score.BeatScore.ok` |
| beat clean | valid and no quality flag (LaTeX or `^`/superscripts read aloud; LaTeX drawn literally in a prose field, judged per component by its own `latex_strings()`; stray `$`); re-flagged at analysis time so every run is judged by the same flags | `score.quality_flags`, `analyze.rescore` |
| video clean | every beat clean | `analyze` |
| CLI valid | `chalkdust validate` exit 0 on the assembled spec | `score.validate_spec_cli` |
| answer shown | the value the AnswerBox (else the last working line) STATES -- its final side, so `6/36 = 1/5` states 1/5 -- equals the sympy-computed key within 2%; a formula with no value is skipped, never scored; `unparsed` counts as not correct. A shown set must equal the key's roots. When the key names a unit and the line holds several statements, a statement in another unit is set aside (`I = 1.2 A, V_6 = 7.2 V` states 1.2 for a current) | `content.check_answer` |
| answer spoken | the last two narrations' final stated value (the last one after 'is/be/about/so...', numbers copied from the request ignored unless asserted) equals the key, however written (digits, fractions, LaTeX, number words, scientific notation); `n/a` for expression answers and when the key is in the request. With a unit-bearing key, a value in another unit is set aside and a value in the key's unit counts as asserted. A set must be the closing statement exactly | `content.check_spoken` |
| derivation | for solve-for-x topics, each working line holds at a true root | `content.derivation_consistency` |
| structure | 8 checks: 5-9 beats, 20-66 words a beat, at least 3 components, none over half, opens with hook or problem, states the answer, names the misconception, no filler greeting | `content.structure_checks` |
| leak | the video draws a leaked field: a TitleCard kicker holding the subtitle, a field label or JSON punctuation (`JEE Main - Calculus", `). Lower is better. Added 2026-11-21, before any T2 output, as a separate column: changing `quality_flags` would change best-of-N selection and the GRPO reward mid-study | `leaks.beat_leaks`, `analyze.rescore` |
| speed | LLM seconds (max over concurrent samples), validator seconds, wall, tok/s from llama.cpp's own timings | `llm.chat`, `run.summarise` |
| memory | VRAM peak minus pre-load baseline; server RSS; layers on GPU as chosen by `--fit` | `llm.GpuSampler`, `LlamaServer.placement` |

On Windows, llama-server's private bytes sit near its VRAM use even with every
layer on the GPU: 10.1 GB after loading Gemma 4 12B, against 0.6 GB of host
buffers in its load log. We infer that Windows charges GPU allocations to system
commit; this is not tested. Read server memory as growth after load, never as an
absolute figure.

Topic sets are frozen in `research/topics.py`: 10 DEV topics for comparisons and 6
HELDOUT topics that no training step sees. Every key is computed by sympy, never typed.
Training topics (`research/train_topics.py`) come from 43 problem families. None is an
evaluation family or a near-twin, and the module asserts the separation at import.

**How the answer checks were validated (2026-11-21, `research/audit/`).**
- **Blind labels.** Two graders per video labelled 127 saved videos from the spec and
  the key alone, without seeing the checkers. The checkers were then compared with
  their consensus, and every false pass and false fail was read.
- **Mutation test.** Wrong answers were planted into real specs, and the test counts
  how many still pass.
- **The fix rule.** Any checker fix must:
  - leave every unmutated verdict that matches the graders unchanged;
  - keep the mutation false-pass counts at zero;
  - land before the step that uses it: the teacher pick, then the SFT filter.
- **Freeze.** The checkers freeze before any held-out output exists, so no change
  can be tuned to the held-out topics.

**What is not measured and why.** Teaching quality, as in "is this a good
explanation", is not scored by a model. A model judge is a weak, gameable signal
(`evaluation-2026-11.md`). The hard checks bound the failures that matter most: wrong
answer, unreadable or unspeakable text, broken layout, no lesson shape. The rest is a
watch list of videos for Lokavya and Shivam (FINDINGS.md).

**Run hygiene.** Every run appends one JSON record per video to
`research/results/<exp>.jsonl`: model, quant, method, seed, commit, prompt hash, all
metrics, timings. Its spec, raw completions, scores and server log are kept under
`.lab/runs/` (git-ignored). Each model gets a probe request before any experiment, so a
model whose chat template still "thinks" is skipped and reported, not scored as bad.
Results never overwrite: a fixed bug means a new experiment id, and the old file stays
as the record of what was wrong.

**Settings that changed mid-study** (each record carries `score_workers` and
`cache_ram_mib` from E2 on):
- E1, E1b and E1c ran with 4 scoring workers and llama-server's default 8 GiB prompt
  cache.
- From E2 the queue runs 2 scoring workers (`SCORE_WORKERS=2`) and no prompt cache
  (`--cache-ram 0`). Four workers beside gpt-oss-20b would overcommit a 32 GB laptop.
- E2's first record (Gemma 4 12B, `dev-quadratic-roots`, 2026-11-21 11:40) ran with a
  1 GiB cache and 4 workers. It is kept. Its wall time is not like-for-like with the
  rest of E2, and `pick_teacher` averages it in.
- Validator wall times are therefore comparable within E2 and later experiments,
  not with E1.
- The answer checkers changed twice on 2026-11-21: `20eb828` (mutation test) and
  `1a0b4f0` (blind audit).
  - `analyze` re-scores every run with the current checkers, so all reported numbers
    use `1a0b4f0`.
  - The teacher pick also ran on `1a0b4f0`: a queue gate held it until that commit.

## 4. Inference-time methods tested

| Method string | What it does |
|---|---|
| `oneshot` | whole video in one call, union schema |
| `oneshot:free=1` | same, no grammar (JSON shape only in the prompt) |
| `staged` | plan, then one fill per beat, n=1 |
| `staged:free=1` | same, no grammar |
| `staged:n=4,repair=1` | best of 4 per beat, one validator-feedback round |
| `...,calc=1` | the plan also writes `answer_calc`; Python evaluates it; beats are told the number |
| `...,plans=5` | 5 plans sampled at once; majority vote on the answer value (clusters within 1%) |
| `...,planfrom=EXP/MODEL/METHOD` | fill a plan another model made (split pipeline) |

All are run by `python -m research.run --exp ... --models ... --methods ...`.

## 5. Bold proposals, and how each is tested

1. **Take the arithmetic away from the model (calc).** Small models plan a correct
   method and then get the number wrong. The plan writes the answer as one expression;
   Python evaluates it; the fills are given the number. Test: answer shown/spoken with
   and without `calc=1`, same model and seeds (E1b). Kill condition: no gain in answer
   correctness.
2. **Split pipeline: big model plans, small model fills.** On a laptop, one 12B plan
   call costs about as much as a few 2B fill calls, and the plan is where the reasoning
   is. Test: `planfrom=` with tier-A plans filled by tier-C models, against the same
   small model planning for itself (E4). Kill condition: no gain over the small model
   alone, or no gain over the big model doing everything at equal wall-clock.
3. **Spend compute on samples, not size.** Best-of-n with the validator selecting,
   against a bigger model single-shot, at equal wall-clock (E3).
4. **Teach the validator's rules by rejection sampling, then RL on the validator.**
   An open-weight teacher you are allowed to distil from generates on the training
   pool. Only clean beats from videos with a correct answer become SFT targets
   (`build_sft.py`). A LoRA on the student learns the format and the house style. Test:
   tuned against untuned on HELDOUT, paired by topic and seed. The primary metric is
   the first-sample clean rate (`n=1, repair=0`), the thing SFT can move. Then GRPO
   with reward = validator gate x content floor (section 6).
5. **Move the validator's blind spots into the validator.** Narration LaTeX, prose
   LaTeX and stray `$` become refusal kinds, so every path (UI, CLI, training filter,
   RL reward) refuses them. A recommendation; the harness flags show the size of the
   problem.

## 6. Training

- **Teacher licence first.** Only open-weight models with licences that put no
  restriction on outputs are used as teachers: Apache-2.0 or MIT
  (`models-2026-11.md`). No closed-model outputs are used as training targets. Claude
  wrote the topic templates and prompts; those are inputs, not targets, and are
  disclosed in the module headers.
- **Teacher choice by measured yield**: kept examples per GPU-minute on DEV-like
  work, not parameter count.
- **Data:** `build_sft.py` keeps a beat sample only if it is clean, needed no escape
  repair, and comes from a video whose answer is right on screen and (for numeric
  keys) in speech; at most 2 distinct samples per beat. The validation split holds out
  whole families.
- **One edit to teacher outputs.** Nearly every TitleCard sample has a leaked kicker
  (FINDINGS item 10). Dropping those samples would remove the opening beat from the
  data, so the kicker is trimmed to its first line without the stray tail instead.
  - The edit replaces only the JSON-escaped kicker in the raw completion, then
    re-parses it to check that nothing else changed. A sample that cannot be
    trimmed safely is dropped.
  - Edited rows carry `"edited": "kicker_trim"`, and `summary.json` counts them.
  - These rows are therefore edited teacher outputs, not verbatim validator-passed
    ones. Every other row is verbatim.
- **Student:** a text-only dense model first (Granite 4.2 3B, `GraniteForCausalLM`,
  Apache-2.0). Qwen3.5 small models are vision-bearing and mostly linear-attention
  (18 of 24 layers in the 2B), which makes LoRA targets, kernels on Windows and GGUF
  export open risks; second attempt only.
- **Recipe:** LoRA on all linear layers, rank 16, rsLoRA, loss on completions only,
  cosine schedule, 2 epochs, QLoRA when bf16 does not fit next to the desktop
  (`research/train_sft.py`, recipe from `techniques-2026-11.md`). Over-length rows are
  dropped, not truncated.
- **Fair comparison:** the tuned adapter is merged into the bf16 base. Tuned and
  untuned are both converted with the same llama.cpp converter (b12466) and quantised
  the same way, so the converter and imatrix are not confounded with the tuning.
- **RL (if within reach):** GRPO on the fill job only, on the SFT model.
  - **Reward** = 1 if the beat is clean, else 0.5 for valid-with-flags, or 0.05 per
    rung passed for invalid. It is multiplied by 0.5 below a content floor, so the
    model cannot win by saying little:
    - narration of at least 20 words, the structure check's per-beat minimum;
    - shown text of at least 12 characters, the floor best-of-N selection already
      uses, with TitleCard exempt.
  - The validator runs in the harness venv as a local service.
  - **Prompts** come from the validation families first (families SFT never trained
    on), then from training rows. On rows SFT had just fitted, every sample in a
    group scored the same, so there was nothing to learn.
    - The set is capped at one prompt per step. TRL's sampler shuffles the set, so a
      larger set would undo the order.
  - **Settings:** LoRA weights are kept in fp32 under the 4-bit base; 120 steps of 4
    samples; learning rate 2e-5.
  - A run whose groups mostly have equal rewards (mean `frac_reward_zero_std` above
    0.8) exits non-zero and is not evaluated, because a null result there means no
    signal, not no effect.
  - These settings were fixed before T3 ran, after a training-chain review
    (2026-11-21).
- **Guards:**
  - `build_sft` stops the queue below 100 training rows.
  - The SFT eval batch is 1, and the adapter is saved before the final eval.
  - A queue gate before each held-out evaluation checks that `PROMPT_HASH` still
    equals the SFT set's, so the tuned model is never evaluated on prompts it was
    not trained on.

## 7. Experiments

| Id | Question | Design |
|---|---|---|
| E0 | Does the loop work end to end? | 1 topic, 1 model, rendered |
| E1 / E1b | Which inference method? | 7 methods x 10 DEV x 1 model (E1 superseded: prompt flaw) |
| E2 | Which model per tier? | ~12 models x DEV x {staged n=1, staged n=4+repair+calc} |
| E3 | Samples or size at equal time? | small + best-of-n vs big single-shot |
| E4 | Split pipeline | tier-A plans + tier-C fills |
| E5 | Student-laptop proxy | CPU-only (`--ngl 0`) speed, labelled a proxy |
| T1 | Teacher data | best tier-A Apache model on the training pool |
| T2 | SFT | tuned vs untuned on HELDOUT x 5 seeds, paired |
| T3 | GRPO | if reachable |
