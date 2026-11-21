# Answer-check audit (2026-11-21)

The answer checks (`research/content.py`: `check_answer` for the answer shown on screen,
`check_spoken` for the answer the narration commits to) decide every accuracy number
in `docs/research/FINDINGS.md`. `check_answer` also filters which teacher videos
become fine-tuning data. This folder is the evidence that they measure what they
claim. The outcome is FINDINGS item 9.

## What was done

1. **Blind labels.**
   - 127 saved videos were split into three slices (`slice{1,2,3}.json`). The specs
     come from E1 to E1c, the smoke runs and E2's first Gemma videos.
   - Each slice went to two independent graders, A and B. They had only the spec, the
     topic and its key (`topics.json`), and the rules in `GRADING.md`.
   - They never saw the checkers or each other's labels. The labels are in
     `labels_slice{N}_{A,B}.jsonl`.
2. **Agreement.**
   - `agreement.py` reports inter-grader agreement (raw and Cohen's kappa) and the
     consensus.
   - It also runs the current checkers against that consensus and lists every false
     pass and false fail.
   - `agreement-1a0b4f0.md` is its output at commit `1a0b4f0`.
3. **Mutation test.**
   - `mutate.py` plants wrong answers into real E1b and E1c specs: a changed number,
     an extra or dropped root, a replaced spoken value, a sign or coefficient change.
   - It counts how many the checkers still pass.
   - The m1 and m2 mutations change only the shown answer, so their `spoken` rows are
     expected to stay correct.
4. **Checker versions.** `versions.py` scores the audited videos with the checker
   before the audit, after `20eb828` and at `1a0b4f0`. Its header says how to run it.
5. **Wrong steps.** `stepcheck.py` compares `derivation_consistency` with the
   graders' `wrong_steps`, and counts SFT-eligible videos (`build_sft.video_ok`) that
   still show a false line.
   - Result: 11 of 85 by either grader, 7 by both. The check flags none of them.
   - It applies only to set keys (14 videos), and it misreads `\implies`.

## Running

From the repo root, with `.lab/venv-harness/Scripts/python.exe`:

```
python research/audit/agreement.py
python research/audit/mutate.py
```

The slice files point at spec traces under `.lab/runs/`, which is git-ignored and
exists only on the lab machine.

## Limits

- 10 DEV topics, mostly one model (Qwen3-4B-Instruct-2507). Read the results as
  failure modes found and fixed, not as a precision figure.
- The two graders agreed on every final-answer label. Their free-text notes and
  step-level flags (`wrong_steps`, `unspeakable`) differ, which shows they worked
  independently. Final-answer correctness against a known key is simply an easy call.
- The `wrong_steps` and `unspeakable` labels are not used by any checker yet.

## T1 teacher data (2026-11-21)

`t1-wave1/` applies the same blind procedure to the first 29 T1 teacher videos (Gemma
4 12B, training topics) that pass `build_sft.video_ok`:

- `slice{1..6}.json` and `topics.json` are built by the scratch slice builder from
  `T1-teacher.jsonl`;
- `labels_slice{N}_{A,B}.jsonl` hold the graders' labels;
- `adjudicated.jsonl` holds an adjudicator's sympy check of every flagged line.

Run `python research/audit/t1_summary.py research/audit/t1-wave1` for the summary.
The result is FINDINGS item 9. The wrong-step count measures the SFT data and does
not filter it.
