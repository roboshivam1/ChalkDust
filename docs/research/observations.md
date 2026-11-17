# Observations from reading generated specs

Things a person sees in a spec that the numbers do not show. Each entry names the run
directory under `.lab/runs/` (git-ignored) so it can be re-checked. Written while the
experiments ran; FINDINGS.md cites the ones that matter.

## E1b, qwen3-4b-2507, `staged:n=4,repair=1`, dev-incline-friction (every beat validator-clean)

Run: `.lab/runs/20261117-194059-e1b/qwen3-4b-2507/staged-n-4-repair-1/dev-incline-friction_s0/spec.json`

- **Valid but wrong answer.** The AnswerBox says 2.0 m/s^2; the key is 3.27. The plan
  wrote `final_answer: "2.0 m/s²"` before any working (the schema order put it first), and
  the working beats (EquationDerivation, SolutionStep) stop at the symbolic
  `a = g(sin θ - μ cos θ)` and never substitute numbers. Nothing in the video computes
  the number it announces.
- **Valid but meaningless visual.** b04 GraphPlot plots `x*sin(30)` (30 radians) over
  [0, 1] while the narration describes resolving gravity into components. The validator
  checks that the graph renders, not that it shows the idea.
- **Garbage in a short field.** b01 TitleCard kicker is `JEE Main - Mechanics”, ` (a
  stray closing quote and comma) and passes.
- **Symbols in narration.** Narration says "m g sin(θ)" and "μ times m g cos(θ)". These
  are not LaTeX, so the narration_latex flag does not fire, but a TTS voice may read
  "θ" and "μ" badly or not at all. Unchecked: how Windows SAPI reads them.
- Good: the misconception beat (friction = μmg, forgetting cos θ) is a real, common
  student error and is explained correctly.

## E1b, staged plans: answers guessed before working

Staged runs with a wrong answer (qwen3-4b-2507, seed 0), plan `final_answer` vs key:

| Topic | Plan said | Key |
|---|---|---|
| dev-projectile-range | 30 m | 34.64 m |
| dev-incline-friction | 2.0 m/s² | 3.27 m/s² |
| dev-series-resistors | 1 A, 4 V, 8 V | 1.2 A |
| dev-binary-search | index 4 | 5 |

One-shot got all four right: there the model writes the working beats before the
AnswerBox, so the answer follows the working. The fix under test is `work=1`.
