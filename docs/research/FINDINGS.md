# Findings: making small local models good at ChalkDust specs

**Status: 2026-11-21, 23:00. Everything has run except GRPO (T3, running) and the
optional Qwen3.6-35B-A3B row of E2.**

- **Done:**
  - the research notes and the method;
  - E1b and E1c, E2 (teachers and small models), E4 (split pipeline), E5 (CPU only);
  - T1 (70 teacher videos), SFT of Granite 4.2 3B, and T2 (tuned against untuned
    on HELDOUT, 6 topics × 5 seeds × 2 methods);
  - two independent reviews with fixes, and blind audits of the answer checks, the
    teacher data and the T2 answers (items 9 and 11).
- **The training result in one line:** SFT fixed the form and broke some of the
  maths (item 11).

Every number below comes from `research/results/*.jsonl` through
`python -m research.analyze` or `python -m research.report`. Both re-score every run
with the current checkers. The checkers froze before any T2 output existed. n is
small: 10 DEV topics with one seed for E1 to E5, and 6 HELDOUT topics with 5 seeds for
T2. Read the E numbers as directions, not effect sizes.

## 1. Outcome

1. **Method beats raw size on validity. Best-of-4 plus one repair round is the
   biggest single lever.** On Qwen3-4B-Instruct-2507:
   - Clean beats rise from 0.65 to 0.77, and videos where every beat is clean rise
     from 0.10 to 0.30.
   - Adding the calculator expression and worked steps (item 2) takes them to 0.82
     and 0.60.
   - It costs about 45 s per video against 28 s on the RTX 5070 Ti laptop GPU.

   Clean means the beat passes validator rungs 1 to 3 and has no quality flag.
2. **With grammar-constrained decoding, the schema's field order is the order the
   model thinks in.** The staged planner wrote `final_answer` before any working, so
   it guessed. Staged answers were 0.60 correct against 0.90 for oneshot. Adding a
   `working` field before `final_answer` brings staged to 1.00 shown and 0.86 spoken
   (E1c; re-scored after the audit in item 9).
   - A calculator expression (`calc=1`) without the working did nothing, because it
     was written from the guess.
   - A 5-plan vote did nothing and cost 6 s.
3. **Grammar-constrained JSON is worth 18 to 26 points of validity** at the same
   model and method: oneshot 0.69 → 0.90, staged 0.61 → 0.87.
4. **The validator cannot tell a right answer from a wrong one, so answer checks
   must sit beside it.** One E1b video passed every rung while showing 2.0 for an
   answer of 3.27; the 2 was the block's mass from the question
   (`docs/research/observations.md`).
5. **Thinking is on by default and silently breaks JSON.** Qwen3.5, Granite 4.2 and
   Gemma 4 all reason first under llama-server and run out of tokens with empty
   content unless `enable_thinking=false` is sent.
   - Granite 4.2 and Qwen3.5 default to thinking in their own templates.
   - Gemma 4's template defaults it off, but llama-server turns it on.
   - The harness now sends the flag, probes every model before an experiment, and
     trains with the same template arguments. A parity check found the trained
     prompt identical to the served prompt on 6 of 6 rows.
6. **The whole training path runs natively on Windows on this laptop (no WSL).**
   The path: Granite 4.2 3B, QLoRA SFT, merge into the bf16 weights, convert to GGUF
   with llama.cpp b12466, quantise to Q4_K_M, serve, evaluate. The numbers are from
   3-step smoke runs, so they test the plumbing, not quality:

   | Run | Peak VRAM | Speed |
   |---|---|---|
   | QLoRA SFT | 4.8 GiB | about 13.8 s per 8-row step |
   | bf16 LoRA | 9.4 GiB | about 12.6 s per step |
   | GRPO, bf16 base | 12.4 GiB | about 45 s per step with 4 generations |
   | Untuned Granite 3B served | 4.8 GB model VRAM | about 100 tok/s |

   - bf16 LoRA works, but it leaves no headroom beside the desktop's 2.4 GB, so the
     real run uses QLoRA.
   - GRPO with the validator as reward runs. The bf16 run spilled into shared memory,
     so the real run loads the base in 4 bits.
   - The untuned Granite 3B passed 1 of 9 and 3 of 7 beats on two dev topics. That is
     a smoke result, not a measurement, but it leaves room for SFT to show an effect.
7. **Memory on a 32 GB Windows laptop is the binding limit, and the first diagnosis
   was incomplete.**
   - llama-server's default prompt cache is up to 8 GiB of system RAM
     (`--cache-ram 8192`). A 1 GiB cap did not stop the problem: under Gemma 4 12B one
     saved slot state is 0.9–1.2 GB, so the cache held one entry and copied about
     1 GB off the GPU on nearly every request.
   - On the 2026-11-21 relaunch, the server's private bytes went from 10.1 GB after
     load to 13.4 GB within two minutes. System commit reached 51.3 of 53.4 GB, and the
     queue was stopped.
   - The harness now turns the cache off (`--cache-ram 0`) and runs 2 scoring workers,
     not 4. Each worker is about 1.7 GB.
   - The 10.1 GB after load is mostly not host buffers: the load log shows 0.6 GB on
     the host. We infer Windows charges VRAM to system commit.
   - ChalkDust's own bundling of llama.cpp should disable the prompt cache and size
     validator parallelism to the machine.
8. **An independent review found five metric and data defects. All were fixed before
   any model comparison ran.**
   - The answer checks passed on any number in the line or narration.
   - FreeBodyDiagram beats could never count as clean.
   - A held-out near-twin was in the training pool.
   - Two dev answers were in the prompt examples.
   - The SFT trainer would have crashed.

   The E1 numbers here are re-scored with the fixed checks. That is why their clean
   rates are lower than the first report: narration containing `^` now counts as
   unclean, because the voice reads it as "caret".
9. **A blind audit of the answer checks found they missed right answers and passed a
   wrong one on only 1 of 127 real videos (an AnswerBox showing `n^2`; see "What is
   left"). The misses are fixed (2026-11-21).**
   - Two graders per video labelled 127 saved videos from the spec and the key,
     without seeing the checkers. The videos came from E1 to E1c, the smoke runs and
     E2's first Gemma videos.
   - The graders agreed on every final-answer label. Their notes and step-level flags
     differ, so they did not copy each other; a final answer against a known key is
     an easy call.
   - A mutation test planted wrong answers into real specs. It found two false-pass
     routes: a shown set with an extra root, and narration that states the roots
     and then closes on another value. Both are fixed (`20eb828`).
   - Against the graders' consensus:

     | Checker | shown: false pass / false fail | spoken: false pass / false fail |
     |---|---|---|
     | Before the audit (used for the first E1 report) | 1 / 5 of 127 | 0 / 19 of 89 |
     | After `1a0b4f0` | 1 / 0 of 127 | 0 / 4 of 89 |

   - **Main cause:** a question that asks for several quantities was scored on the
     last value stated. "Find the current, and the voltage across each resistor" was
     scored on the last voltage. The fix uses the unit the key already names: a value
     in another unit is set aside.
   - **Two more causes:** scientific notation was read as two numbers, and
     "one-sixth" as no number.
   - **What is left:**
     - The one shown false pass: an AnswerBox showing `n^2` has no number, so the
       working line's 16 decides.
     - Two videos never restate the answer in their last two narrations. The
       graders looked further back.
     - Two narrations end on a justification ("as there are 6 favorable
       outcomes"), whose cued 6 is read as the answer.
   - **Wrong steps get past every hard check.**
     - About 1 in 8 videos with a right final answer still shows a false
       intermediate line: 11 of the 85 that pass the SFT filter by either grader,
       7 by both.
     - The step-consistency check (`derivation_consistency`) applies only to roots
       topics (14 of 127 videos). It caught none of these, and it misread a correct
       `\implies` line as wrong.
     - So SFT data filtered on the final answer can still teach a false line. Its
       share in the T1 teacher data is measured by blind graders, not filtered by
       them: hard checks decide what is kept.
     - **The T1 teacher data (wave 1, 2026-11-21)**, using the same blind procedure
       on the first 29 Gemma 4 12B videos that pass the SFT filter
       (`research/audit/t1-wave1`, `t1_summary.py`):
       - The graders agreed on every shown and spoken label.
       - The current checkers matched them on every video they could score: no false
         pass and no false fail (26 shown, 23 spoken).
       - 4 of 29 show a line at least one grader called false (3 by both graders).
         An adjudicator re-checked each line with sympy:
         - 1 teaches an error: a unit-circle diagram labels the sine leg
           "Horizontal" and the radius "y-axis".
         - 3 are harmless slips, such as swapped leg letters with the right
           lengths.
       - So about 1 teacher video in 30 carries an error-teaching line into SFT,
         against about 1 in 8 for Qwen3-4B on DEV.
   - The audit covers 10 topics and is mostly one model, so read these as failure
     modes, not a precision figure. Labels, the agreement script and the mutation
     script are in `research/audit/`.
   - **The training topics needed a second pass** (`6067b2c`, before any SFT data
     was built). A false-pass sweep wrote right and wrong answers for every training
     family and ran them through the checks that gate the SFT data. It found wrong
     answers that passed:
     - `v = 338.5 m/s` did not parse. The narration decided instead, and its
       misconception sentence ("…which would give 572 metres per second") equalled
       the key.
     - The 2% tolerance passed 58 for 57, 122 for 5! = 120, and simple interest
       (2480) for compound interest (2519.42).
     - It also found correct answers that failed: `KE = 60,000 J` read as 0,
       "192 joules per second" set aside as an energy, and a Unicode minus dropped.

     All of these are fixed and pinned in `research/tests/test_train_checks.py`.
     No verdict changed on the 148 saved videos, and the audit numbers above
     stand.
     - A later code review fixed three more before T2 (`a12154b`, `d4f605b`):
       - a LaTeX `\;` read as a separator;
       - the x/y swap on simultaneous equations (`x = 1, y = 3` for x = 3; the
         asked variable now decides);
       - a statement in the key's unit losing to a later given (`v = 572 m/s,
         f = 440 Hz`).
     - Across 182 saved videos, only one verdict changed, the intended fix.
     - **Known limits, found by the final review and not fixed.** The checks froze
       before T2, so fixing these would re-score held-out output after the fact.
       They are queued for after the study (section 6 item 7):
       - The asked-variable rule splits only on commas, `;`, `\quad`, `\\` and
         `\text{ and }`, and needs a bare `x =`. So `x = 2 and y = 3` still passes
         for x = 3, and `So x = 3, y = 2` fails.
       - `check_spoken` does not use the asked variable: "x is two and y is three"
         passes for x = 3.
       - A statement in the key's unit now beats a later untagged one, even when
         both name the same variable. `F = 5 N, so F = 10` fails for 10 N; before
         `d4f605b` it passed.
       - `_part_unit` reads an exponent digit as the value, so `m/s^2` statements
         are never tagged.
       - A mean/median line is read by its last statement.
       - None of these touches a HELDOUT topic's question form except the
         same-variable correction, and the T2 answer audit (item 11) does not use
         the checker.
   - **T1 teacher data, all of it** (wave 2, 2026-11-21 evening). 64 of 70 teacher
     videos pass the SFT filter, and two blind graders read all 64. They agree with
     the checks on every answer.
     - 2 of 64 carry an error-teaching line:
       - wave 1: a trig-identity construction that labels the vertical leg
         "Horizontal";
       - wave 2: a Fibonacci StepTrace whose "sum" column repeats `b` (1, 1 → 1).
     - Both stay in the SFT set. The filter cannot see them (item 4), and the
       results are not edited after the fact.

10. **The same schema-order trap hit the title card, and the validator cannot see
    it** (found 2026-11-21 while checking the teacher's first videos).
    - **The cause.** TitleCard's params are generated in class order: title, then
      an optional `subtitle`, then an optional `kicker`. The prompt (`catalog.py`)
      describes them as title, kicker, subtitle, and unconstrained models write
      that order (6 of 6 E1 "free" samples).
    - **What happens.** Under the grammar, models skip `subtitle` and write the
      kicker. They then put the subtitle, a `Subtitle:` label or JSON punctuation
      inside the kicker, and it is drawn on screen: `JEE MAIN - CALCULUS",`. Some
      samples loop inside the kicker until the token limit.
    - **How often**, as leaky clean TitleCard samples:

      | model | leaky |
      |---|---|
      | Qwen3-4B | 40 of 40 |
      | Gemma 4 12B | 24 of 25 |
      | gpt-oss-20b | 5 of 12 |

      No other component leaked in about 800 samples. Per video (`leak` column):
      - 0.00 for oneshot and staged:free;
      - 0.50 to 0.80 for Qwen's constrained staged methods;
      - 1.00 for Gemma;
      - 0.40 for gpt-oss.
    - **What changed and what did not.** Generation is frozen until the study ends,
      so nothing changed there.
      - The rate is reported as `leak`, defined before any T2 output.
      - SFT targets get the kicker trimmed (METHODOLOGY section 6).
      - The fix belongs in `generate.beat_schema`: put `kicker` before `subtitle`,
        or follow the prompt's order. It is item 6 in section 6.

11. **SFT fixed the form and broke some of the maths** (T2, section 2c). The student
    is Granite 4.2 3B after QLoRA SFT.
    - Its training data: 932 rows (57 plans and 875 beat and beat-repair rows) from
      57 Gemma 4 12B videos. Of 70 generated videos, 64 passed the filter; the
      other 7 that passed went into the 114-row validation set.
    - It is compared with the same model untuned on 6 HELDOUT topics, 5 seeds and
      2 methods, paired.
    - **Form, the primary metric, improves a lot.**
      - First-sample clean beats rise from 0.38 to 0.90 (+0.52, 95% CI +0.42 to
        +0.62).
      - 29 of the 30 (topic, seed) pairs favour the tuned model. The exception is
        arithmetic-sum, seed 3, at −0.07. Resampling topics instead of pairs gives
        +0.33 to +0.70, and all 6 topics favour it.
      - Field leaks fall from 0.23 to 0.00.
    - **Answers get worse, and the audit says the checker understated by how much.**
      - By the checker, shown answers fall from 0.80 to 0.68 (95% CI −0.28 to
        +0.05; topic bootstrap −0.33 to +0.05) and spoken answers from 0.78 to 0.60.
        The spoken figures cover 25 pairs on 5 topics; the stack topic has no
        spoken check.
      - Blind graders then read every T2 video the checker did not pass, plus 24
        controls. For the tuned model they confirm every shown verdict.
      - They do not confirm the spoken ones. For the tuned model, 3 spoken
        "wrong" are right and 4 unchecked (`n/a`) are wrong; for the untuned, 2
        "wrong" are right. The spoken drop is checker-only and not corrected by
        the audit.
      - For the untuned model they find 8 right answers among the 12 the checker
        failed. Its answers come in looser formats, such as an unevaluated product
        or an answer only in the narration.
      - Audited, shown answers are 0.87 to 0.93 untuned against 0.68 tuned. The
        range comes from scaling the untuned model's checker passes by its control
        false-pass rate (1 of 12, a control with no answer on screen). That rate is
        estimated from 12 videos.
      - Among checker-passed videos, both graders flag a false working line in 5 of
        12 tuned and 0 of 12 untuned.
    - **The likely mechanism: the student copied the teacher's style, not its
      accuracy.** No ablation tested it, but the traces are consistent with it.
      - Gemma 4 12B writes `working` as long hand arithmetic, and the student does
        the same (`T = 2π × 0.3266 = 0.69`), with wrong numbers. The untuned model
        writes one line (`≈ 2.006 s`).
      - SFT made the student write a calculator expression (`answer_calc`) far more
        often: 53 of 60 videos against 31 untuned. It got the value right 45 times
        (0.85 of those it wrote), against 30 of 31 untuned.
      - The beat prompt gives the plan's final answer with "(calculator check: X)"
        beside it. In 10 of the 53 tuned videos the calculator was right and the
        shown answer wrong, and the audit confirms every one of those shown
        verdicts.
      - In the teacher data the two always agreed, because only correct videos
        were kept. So the student never saw a case where the calculator should
        win.
      - Counterfactual, computed after the run and not a result: if the calculator
        value decided whenever there was one, untuned would score 0.88 and tuned
        0.82. The untuned figure leans on the checker's false fails, because the
        untuned model rarely wrote an expression.
      - Script: `research/audit/scripts/t2_extra_stats.py`. It also computes the
        topic bootstrap.
    - **What it suggests:** at 3B, SFT on teacher outputs bought validity and
      polish here, and cost correctness. Taking the arithmetic out of the model's
      hands may close the gap. That is a prediction, and section 6 item 1 tests it.
    - The losses concentrate on one topic: pendulum period, −0.6 shown. Parabola
      vertex and ideal-gas volume each lose about 0.1 to 0.2. So "−0.2" is a few
      topics, not a uniform shift.

12. **Without any training, Qwen3.5-4B with best-of-4 and repair is the best small
    model** (E2, section 2d). On DEV it has every beat clean in 10 of 10 videos,
    0.90 shown and 0.86 spoken, at 65 s per video and 4.6 GiB of VRAM. Tuned Granite
    3B scores 0.63 shown on HELDOUT with the same method. Those are different topic
    sets, so it is not a paired comparison, but it is the obvious next student (section
    6 item 3). It is a tier C/B model.

## 2. E1: how to ask a small model (Qwen3-4B-Instruct-2507, Q4_K_M)

The table covers 10 DEV topics with seed 0, re-scored. Columns:
- `clean`: share of beats that are valid and unflagged.
- `allclean`: share of videos with every beat clean.
- `ans`: shown answer correct.
- `spk`: the narration's final stated value is correct.
- `wall_s`: seconds per video (generation plus validation).

| method | valid | clean | allclean | ans | spk | struct | wall_s |
|---|---|---|---|---|---|---|---|
| oneshot | 0.90 | 0.42 | 0.00 | 0.90 | 1.00 | 0.95 | 23 |
| oneshot, free decoding | 0.69 | 0.34 | 0.00 | 1.00 | 1.00 | 0.89 | 20 |
| staged | 0.87 | 0.65 | 0.10 | 0.60 | 0.57 | 0.89 | 28 |
| staged, free decoding | 0.61 | 0.46 | 0.00 | 0.60 | 0.57 | 0.89 | 28 |
| staged + work | 0.93 | 0.62 | 0.10 | 1.00 | 0.86 | 0.89 | 33 |
| staged, best-of-4 + repair | 0.97 | 0.77 | 0.30 | 0.60 | 0.57 | 0.93 | 45 |
| + calc | 1.00 | 0.83 | 0.40 | 0.70 | 0.57 | 0.90 | 46 |
| + calc + 5-plan vote | 1.00 | 0.76 | 0.30 | 0.60 | 0.57 | 0.95 | 52 |
| **+ calc + work** | **1.00** | **0.82** | **0.60** | **1.00** | **0.86** | 0.90 | 45 |

`ans` and `spk` were re-scored on 2026-11-21 with the audited checkers (item 9). Every
change is upward, because the earlier checkers missed right answers and passed no
wrong one. `spk` is out of the 7 topics whose key is not in the question.

Speed: 69 to 121 tok/s per request. E1c's server used 7.4 GB of VRAM for the model
with a 32k context over 4 slots, and peaked at 11.1 GB of host RAM before the cap.

## 2b. E2 so far: the teacher candidates (2026-11-21)

The table covers 10 DEV topics with seed 0 and the recommended method
(`staged:n=4,repair=1,calc=1,work=1`), from `python -m research.report --exp E2-models`.

| model | licence | valid | clean | allclean | ans | spk | struct | wall_s | tok/s | VRAM MB |
|---|---|---|---|---|---|---|---|---|---|---|
| Gemma 4 12B (Q4) | Apache-2.0 | 0.99 | 0.97 | 0.80 | **1.00** | **0.86** | 1.00 | 154 | 29 | 9751 |
| gpt-oss-20b (MXFP4, experts in host RAM) | Apache-2.0 | 1.00 | 1.00 | **1.00** | 0.80 | 0.57 | 0.94 | 132 | 27 | 10130 |

- **The cleanest model is not the most correct.** gpt-oss-20b produced every beat
  clean but showed 2 of 10 answers wrong. Gemma 4 12B showed all 10 right.
  Validity and correctness are separate axes, which is why the answer checks sit
  beside the validator (item 4).
- **Teacher pick (`research/pick_teacher.py`):** most usable videos per hour, where
  usable means the right answer shown and spoken, among models with at least half
  usable.
  - At pick time: Gemma 9 of 10 usable, 21.1 per hour; gpt-oss 6 of 10, 16.3 per
    hour.
  - Gemma 4 12B became the T1 teacher. At 154 s per video, the 3-hour budget covers
    70 training topics.
- **Correction: the pick is a near-tie, not a clear win.** A later code review found
  two checker false fails on gpt-oss videos:
  - a `\;` LaTeX space was read as a separator (fixed; the table above is
    re-scored);
  - "one is two, the other is three" heard "one" as a number (a known limit, not
    fixed).

  Re-scored, gpt-oss is 7 of 10 usable (19.1 per hour). Counting the second case as
  well, it would be 8 of 10 (21.8 per hour) against Gemma's 21.1. On 10 topics that
  is noise. Gemma stays the teacher for three reasons:
  - T1 was already half done;
  - its shown answers are 10 of 10 against 8 of 10;
  - it does not need about 12 GB of experts in host RAM.

  The rule's outcome alone does not justify the choice.
- Gemma's first record ran under the old memory settings, so its `wall_s` is "MIXED"
  (METHODOLOGY section 3).

## 2c. T2: tuned against untuned on HELDOUT (2026-11-21)

`python -m research.report --exp T2-heldout`. Models: Granite 4.2 3B, untuned
(`granite-4.2-3b-self`) and after SFT (`granite-4.2-3b-sft-v1`). Topics: 6 HELDOUT
topics × 5 seeds. Methods: `staged:calc=1,work=1` (one sample, no repair) and
`staged:n=4,repair=1,calc=1,work=1`. The teacher's ceiling uses seeds 0 and 1.

| model | method | n | valid | clean | allclean | ans | spk | leak | wall_s |
|---|---|---|---|---|---|---|---|---|---|
| untuned | n=1 | 30 | 0.54 | 0.37 | 0.03 | 0.73 | 0.72 | 0.30 | 24 |
| untuned | best-of-4 + repair | 30 | 0.82 | 0.72 | 0.33 | 0.87 | 0.84 | 0.17 | 69 |
| **SFT** | n=1 | 30 | **0.93** | **0.91** | 0.40 | 0.73 | 0.64 | **0.00** | 34 |
| **SFT** | best-of-4 + repair | 30 | **0.96** | **0.96** | **0.70** | 0.63 | 0.56 | **0.00** | 49 |
| Gemma 4 12B (teacher) | n=1 | 12 | 0.90 | 0.86 | 0.33 | 1.00 | 0.90 | 0.83 | 61 |
| Gemma 4 12B (teacher) | best-of-4 + repair | 12 | 0.98 | 0.96 | 0.67 | 1.00 | 1.00 | 0.75 | 110 |

Paired, SFT minus untuned. Both methods are averaged into one pair per (topic, seed):
30 units, bootstrap 95% CI, exact sign test. Topic CI resamples the 6 topics
(`research/audit/scripts/t2_extra_stats.py`; `report.py` gives the rest).

| metric | SFT | untuned | diff | 95% CI | sign p | topic CI |
|---|---|---|---|---|---|---|
| first-sample clean, per beat (primary) | 0.897 | 0.377 | +0.521 | [+0.421, +0.615] | <0.0001 | [+0.327, +0.695] |
| first-sample all-clean, per video | 0.317 | 0.033 | +0.283 | [+0.150, +0.417] | 0.0007 | [+0.083, +0.483] |
| allclean (final video) | 0.550 | 0.183 | +0.367 | [+0.167, +0.567] | 0.0066 | [−0.067, +0.733] |
| ans (checker) | 0.683 | 0.800 | −0.117 | [−0.283, +0.050] | 0.24 | [−0.333, +0.050] |
| spk (checker) | 0.600 | 0.780 | −0.180 | [−0.340, −0.020] | 0.12 | [−0.400, 0.000] |
| field leak | 0.000 | 0.233 | −0.233 | [−0.333, −0.133] | 0.0002 | [−0.317, −0.150] |

- **Audited shown answers** (`research/audit/t2-heldout/`): 0.87 to 0.93 untuned
  against 0.68 tuned (item 11).
- **The teacher's own leak is 0.75 to 0.83**, and the student's is 0.00. The SFT
  trim (item 10) removed it from the student's targets, and the student never learned
  it.
- **Speed:** at n=1 the tuned model is slower (34 s against 24 s), and it writes
  longer working. Under best-of-4 it is faster (49 s against 69 s). Repair rounds
  were not counted per arm, so the cause of that difference is not measured.
- **SFT run:** QLoRA r16, 2 epochs over 932 rows (57 plans and 875 beat and
  beat-repair rows, from 57 videos), 49 minutes,
  5.2 GiB peak VRAM. Train loss went from 0.77 to 0.13; eval loss on the 4 held-out
  training families is 0.48.

## 2d. E2 small models, E4 split pipeline, E5 CPU only (DEV, seed 0)

E2, 10 DEV topics:

| model | VRAM GiB | method | allclean | ans | spk | wall_s |
|---|---|---|---|---|---|---|
| Qwen3.5-4B | 4.6 | best-of-4 + repair | **1.00** | **0.90** | **0.86** | 65 |
| Qwen3.5-4B | 4.6 | staged, n=1 | 0.10 | **1.00** | **1.00** | 29 |
| Qwen3.5-2B | 2.5 | best-of-4 + repair | **1.00** | 0.60 | 0.71 | 52 |
| Gemma 4 E2B | 2.8 | best-of-4 + repair | 0.60 | 0.50 | 0.29 | 50 |
| Granite 4.2 3B | 5.3 | best-of-4 + repair | 0.20 | 0.80 | 0.57 | 62 |
| Gemma 4 12B | 9.4 | best-of-4 + repair | 0.80 | 1.00 | 0.86 | 154 |
| gpt-oss-20b | 9.8 | best-of-4 + repair | 1.00 | 0.80 | 0.57 | 132 |

- **Best-of-4 plus repair is what makes small models clean.** Every model gains, and
  Qwen3.5 at 2B and 4B reaches 10 of 10 all-clean videos. Correctness is a property of
  the model: Qwen3.5-2B is clean and 0.60 right.
- **E4, split pipeline** (Gemma 4 12B's saved plans, filled by a small model). The
  borrowed plan mostly lifts what the small model says, not what it shows:
  - Granite 3B: spoken 1.00 against 0.43 at n=1, and 0.86 against 0.57 under
    best-of-4. Shown answers and allclean are unchanged.
  - Qwen3.5-2B: shown 0.70 against 0.60 at n=1. Under best-of-4, allclean is 0.90
    against 1.00.
  - Qwen3.5-4B: no gain.
  - On 10 topics none of this is clear of noise. The split also costs a 12B model's
    planning time, so it is not worth it on one machine.
- **E5, CPU only** (`--ngl 0`, 4 threads, 3 topics): Qwen3.5-2B takes 65 to 95 s per
  video at about 22 tok/s, and Granite 3B takes 166 to 287 s at 10 to 13 tok/s. A
  laptop without a GPU can draft a video in a few minutes. Quality at n=3 is too noisy
  to rank.

## 3. Recommended method (provisional: E2 picks the model per machine)

Plan first, then fill each beat:
1. **Plan:** grammar-constrained. Fields in order: title, objective, key insight,
   misconception, **working** (numbered steps with the numbers), **answer_calc** (a
   calculator expression the app evaluates), final answer, beats.
2. **Fill:** per beat, grammar-constrained to that component's schema. Draw 4
   samples, and the validator picks the best.
3. **Repair:** one round, feeding the typed refusal back.
4. **Check:**
   - the shown answer against the sympy-computed value;
   - the spoken answer's final stated value;
   - text flags for LaTeX or `^` read aloud.

   These checks are an extra rung beside the validator, not a replacement for it.
5. **Model settings:** thinking off. Turn llama-server's prompt cache off
   (`--cache-ram 0`).

The method is laid out in full in `docs/research/METHODOLOGY.md`, and the background
research in the four `docs/research/*-2026-11.md` notes. In short:
- Machine A: Qwen3.6-35B-A3B or Gemma 4 12B.
- Machine B: Qwen3.5-9B.
- Machine C: Qwen3.5-4B or Granite 4.2 3B, all Apache-2.0.

These are vendor numbers; E2 is what measures them here.

## 4. Licences and risks found

- **Teacher outputs:** only Apache-2.0 and MIT open models are used as teachers, and
  no closed-model output is a training target. Avoid as teachers:
  - Llama 4 (naming clause);
  - Gemma 1 to 3n terms (Gemma 4 is plain Apache-2.0);
  - LFM (revenue cap);
  - the Qwen Community License model (model-as-a-service clause).
- **Rendering, for a paid app:** Manim 0.21 hard-codes `libx264`, and PyAV bundles
  GPL x264 into the process. Ship an LGPL FFmpeg with the OS encoder, and get counsel
  on H.264 patents (`rendering-2026-11.md`).
- **3b1b material:** the 3b1b video code is CC BY-NC-SA and cannot go into a paid
  product (`evaluation-2026-11.md`).
- **Voices (from a 2026-11-21 primary-source pass; the four research notes did not
  cover TTS):**
  - **Kokoro-82M** is Apache-2.0 (model card), and its phonemiser `misaki` is
    Apache-2.0.
  - A default `kokoro` install puts **two GPL-3 components in-process**:
    `phonemizer-fork` 3.3.2 (GPL-3) and the espeak-ng shared library
    (GPL-3.0-or-later). The library is loaded by ctypes through `espeakng-loader`,
    whose own code is MIT.
  - `kokoro` depends on `misaki[en]`, and that extra pulls both in.
  - For English, espeak-ng is only the last-resort fallback for words outside
    misaki's lexicon. For most other languages it is the primary G2P.
    (Source read of kokoro and misaki 0.9.4, 2026-11-21. Not run.)
  - **Way out, untested:** install without those two packages so misaki falls back
    to its own neural G2P (`FallbackNetwork`, a BART model), then check pronunciation
    of out-of-lexicon words. No documented non-GPL mode exists.
  - Some Kokoro voices carry their own dataset terms. `af_heart` shows none.
  - The Windows SAPI and macOS voices come with no redistribution grant we could find.
    Keep them as development voices, as `docs/VOICE.md` already does.
- **Fonts:** Archivo, Inter and JetBrains Mono are SIL OFL 1.1 with no Reserved Font
  Name, so they may be bundled with paid software. Each needs its copyright line and
  the licence text in the app's third-party notices, as each bundled model needs its
  Apache-2.0 text.

## 5. What failed, was skipped, or is unverified

- **Stopped twice for memory:**
  - 2026-11-17, about 21:05: Claude Code stopped the GPU queue, the downloads and the
    keep-awake lock.
  - 2026-11-21: the queue, relaunched in its own window, was stopped by hand after
    2 minutes at 51.3 of 53.4 GB commit (item 7).
  - It was relaunched at 11:50 with the cache off, 2 scoring workers and a memory guard.
- **Not run:** the Qwen3.6-35B-A3B row of E2 (queued last). The T2 held-out set is
  6 topics. Its intervals are honest about that, but a second held-out set would make
  item 11's effect sizes firmer.
- **Checker errors are not symmetric across models.** On T2 they fell on the
  untuned model's looser answer formats (item 11). A checker-only comparison
  understated the tuned model's loss, which is why the T2 answers were hand-audited.
- **Where each model's weights went (GPU or CPU) is not recorded per run:** b12466
  does not log `--fit`'s layer split at the default log level. One-off loads at `-lv 4`
  on 2026-11-21, with 32k context over 4 slots:

  | model | placement | KV cache | after load |
  |---|---|---|---|
  | Gemma 4 12B | 49/49 layers on the GPU: 6.8 GB weights, 0.5 GB embeddings mapped on the host | 2.4 GB on the GPU | private 10.1 GB |
  | gpt-oss-20b | 25 layers' dense weights on the GPU; most expert weights in host RAM | n/a | commit 29.5 → 39.4 GB, 6.3 GB physical free |
- **E1, E1b and E1c ran with llama-server's default 8 GiB host cache and 4 scoring
  workers.** E2 onwards uses no cache and 2 workers, so speed comparisons across that
  change are not like-for-like (METHODOLOGY.md, "Settings that changed mid-study").
- **Videos:** one complete local video exists so far: E0, Qwen3-4B, projectile range,
  draft, 170.6 s, at `.lab/videos/E0/`. The watch list (`research/watchlist.py`) is
  queued behind the teacher sweep.
- **Skills:** the brief asked for the Karpathy guidelines and software-design-
  philosophy skills. Neither is installed here, so neither was used.
- **Disclosure:** Claude wrote the topic set, the training-topic templates and all
  prompts. These are model inputs, never training targets.

## 6. Ranked next experiments

1. **Take the arithmetic out of the model's hands** (item 11). When the plan's
   `answer_calc` evaluates, the harness writes that value as the final answer. The
   beat prompts then carry the calculator's number, not the model's.
   - Also add SFT rows where the plan's hand arithmetic disagrees with the calculator
     and the target follows the calculator.
   - This changes generation and the prompt hash, so it means a new SFT version.
   - Confirm on DEV and on a new held-out set. HELDOUT has now been audited by hand,
     and the counterfactual in item 11 was computed on it.
2. **Reward correctness in RL, not just validity.** After SFT the validator reward is
   saturated: in most GRPO groups all 4 samples score 1 (section 2c, T3). A reward of
   validator × answer check on keyed TRAIN topics, applied to plan prompts as well as
   beat prompts, gives the signal where the student is actually weak.
3. **Qwen3.5-4B as the student** (item 12). Untuned, it is the best small model on
   DEV. Unsloth lists bf16 LoRA at 10 GB. It has vision and linear-attention layers,
   so expect a slower path on Windows.
4. **Filter teacher data on working, not only on the answer.** 2 of 64 teacher videos
   teach an error the filter cannot see (item 9). `research/audit/stepcheck.py` checks
   whether each derivation line follows from the one before. It could gate SFT rows
   if it agrees with the graders.
5. **Fix the TitleCard field order** (item 10).
   - Generate `kicker` before `subtitle` in `generate.beat_schema`.
   - Audit every component's optional fields for the same trap.
   - This changes the prompt hash, so it starts a new SFT data version. The `leak`
     column measures the effect.
6. **Qwen3.6-35B-A3B** (E2): queued last, behind a 22 GiB RAM gate.
7. **After the study: fix the checker limits in item 9** (the asked variable in
   sentences and narration, same-variable corrections, exponent units). Pin each case
   in `research/tests/test_train_checks.py`, re-run `diff_checkers` over every saved
   video, and report re-scored tables under a new label. Don't overwrite the frozen
   T2 numbers.

## 7. How to resume

Run the queue in its own PowerShell window. Claude Code stops its own background
jobs when memory runs low, but it does not stop a window you opened:

    cd C:\dev\chalkdust\ChalkDust
    .\.lab\venv-harness\Scripts\python.exe -m research.gpu_queue

The queue:
- skips finished jobs (`.lab/queue/done.txt`);
- keeps the laptop awake while it runs;
- waits before each job until no other program holds more than 4.5 GB of the GPU
  and at least 8 GiB of RAM is free;
- stops at a job that exits non-zero, or that its memory guard kills (available RAM
  under 1.5 GiB or commit headroom under 2 GiB for 60 s). It leaves that job out of
  `done.txt`, so starting the queue again retries it, and `--resume` keeps the
  records already saved;
- logs every start, finish and failure to `.lab/queue/queue.log`;
- prints `ALL DONE` at the end.

At the stop, WSL Ubuntu held 15.4 GB of RAM, 12.7 GB of it file cache, and a voice
job running in Ubuntu held about 5 GB of the GPU. Once Ubuntu's work is finished,
`wsl --shutdown` frees that memory.
