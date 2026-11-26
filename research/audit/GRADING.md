# Grading instructions: is the video's answer right?

You are grading ChalkDust explainer-video specs that small local LLMs generated. Each
spec is a JSON video: a list of beats, each with a `component` (TitleCard, WorkedStep,
AnswerBox, EquationDerivation, ...), its `params` (what is shown on screen) and a
`narration` (what the voice says). Your labels are the ground truth used to measure
how often automatic answer checkers are wrong. Be careful and literal; grade what is
in the spec, not what the model probably meant.

## Inputs
- Your slice file lists the videos: `id`, the absolute `spec` path and the `topic` id.
- `topics.json`, in this folder, gives each topic's `request` (the question), `key_kind`
  (number, set, expr or antiderivative), `key_value` (the correct answer, computed by
  sympy) and `rel_tol`.

## Labels, one JSON object per line
```json
{"id": "...", "shown": "correct|wrong|absent|ambiguous", "shown_value": "verbatim",
 "spoken": "correct|wrong|absent|ambiguous", "spoken_value": "verbatim snippet",
 "wrong_steps": ["verbatim on-screen line that is mathematically false", "..."],
 "unspeakable": ["narration fragment a text-to-speech voice would read badly", "..."],
 "confidence": "high|medium|low", "note": "at most 20 words"}
```

## shown: the final answer displayed on screen
- If the spec has an AnswerBox (or a similar final-answer component), use its value.
- Otherwise, use the final value of the last working or derivation line that states
  the result.
- **number:** correct if within `rel_tol` (default 2%) of the key.
- **set** (for example the roots of an equation): correct only if the shown set equals
  the key set, with nothing missing and nothing extra.
- **expr:** correct if algebraically equivalent to the key.
- **antiderivative:** correct if equivalent up to a constant.
- **absent:** nothing on screen states a final result.
- **ambiguous:** several competing final results are shown and none is clearly final.
  Explain in `note`.
- A formula with no value, when the key is a number, is `absent`, not `wrong`.

## spoken: the final answer the narration commits to
- Grade it the way a listener hears the end of the video, usually the last one or two
  narrations.
- Numbers merely restated from the question are not an answer.
- **correct:** the committed final value matches the key, however it is written
  (digits, words, fractions).
- **wrong:** the narration commits to a different value.
- **absent:** it never states the result.
- **ambiguous:** it states several candidate values without committing.

## wrong_steps
Any on-screen maths line that is false, even if the final answer is right. Examples:
an algebra slip, a wrong substitution, an equation that doesn't hold.

## unspeakable
Narration text that would be read aloud badly:
- LaTeX (`\frac`, `\sqrt`, `^`, `_`, braces);
- code syntax;
- raw math symbols a voice reads oddly (`≤`, `→`, `√`);
- `x^2` written instead of "x squared".

## Rules
- Read only the `spec.json` files in your slice and `topics.json`.
- Do not open `trace.json`, anything under `research/results/`, other files under
  `.lab/runs/`, or `research/content.py` / `research/analyze.py`. They hold the
  automatic checker's verdicts and logic, and you must stay blind to them.
- You may run small Python or sympy arithmetic with
  `C:\dev\chalkdust\ChalkDust\.lab\venv-harness\Scripts\python.exe`.
- Run nothing that uses the GPU, Manim, LaTeX or the chalkdust CLI. A long GPU job is
  running on this machine.
- Do not edit anything in the repo. Write only your own labels file in this folder.
