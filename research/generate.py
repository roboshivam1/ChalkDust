"""Ways to turn a topic into a ChalkDust spec with a (small) local model.

Methods (each recorded by name in the results):

  oneshot   one call writes the whole video (title + beats with component and
            params), constrained to a schema that is a union of the 17
            component beat schemas.
  staged    step 1: plan (title, objective, key insight, misconception, final
            answer, beats with purpose/idea/component); step 2: one call per
            beat writes narration + params against that component's schema.
            Options:
              n        best-of-N samples per beat, picked by the validator
              repair   rounds of validator-feedback regeneration for a beat
                       whose best sample still fails (SCENE_SPEC.md §9 step 2)
              calc     the plan also gives the final answer as an arithmetic
                       expression; sympy evaluates it and the beats are told
                       the computed value (a tool step for arithmetic)

Everything returned is plain data: prompts, raw outputs, scores and timings,
so a bad result can be explained later.

Authorship note: the prompt text below was written by Claude. Prompts are
model inputs; for training data only model OUTPUTS are kept as targets.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import sympy as sp

from research.catalog import (ALLOWED, beat_schema, card_text, catalog_text,
                              oneshot_schema, plan_schema)
from research.llm import Completion, chat, chat_many, parse_json
from research.score import BeatScore, score_beat, score_many, selection_key
from research.topics import Topic

RULES_JSON = (
    "Output one JSON object and nothing else. Inside JSON strings every backslash "
    "must be doubled: write \\\\frac{1}{2}, \\\\theta, \\\\sqrt{3}, \\\\mathrm{m}. "
    "A single backslash before t, f, b, r or n turns into an invisible control "
    "character and the beat is refused.")

PLAN_SYSTEM = f"""You are the lesson planner for ChalkDust, which makes short 3Blue1Brown-style explainer videos that help a student understand one exam topic.
You do not draw anything. You choose, for each beat, one visual component from a fixed library and say what the beat must teach. Another step writes the narration and fills the component.

A good lesson:
- opens with the question or a hook that makes the student want the answer;
- builds intuition before formal working;
- shows the working one step per beat, each step correct;
- names the most tempting wrong idea and shows why it fails;
- ends with the answer clearly stated (for a problem) and what to remember.

Rules:
- 5 to 9 beats. One idea per beat (8-25 seconds of speech).
- Pick the component that SHOWS the idea best. Use at least 3 different components. Do not use one component for more than half the beats.
- For a numerical problem: a ProblemStatement near the start, SolutionStep or EquationDerivation for the working, and an AnswerBox at the end.
- final_answer: the answer in a few words with units (or "n/a" for a pure explanation). Work it out carefully.
- {RULES_JSON}

The component library:
{catalog_text()}"""

PLAN_CALC_ADDENDUM = """
- answer_calc: the final numerical answer written as ONE arithmetic expression that a calculator can evaluate, using numbers, + - * / ** ( ), sqrt(), sin() cos() tan() (radians; write 30 degrees as 30*deg), pi, E. Example: "20**2*sin(2*30*deg)/10". Use "" if the answer is not a single number."""

BEAT_SYSTEM = f"""You write one beat of a ChalkDust explainer video: the narration the student hears, and the params of the one visual component on screen.

Narration rules:
- 30 to 60 words, spoken English, warm and precise, like a good teacher at a whiteboard.
- Say maths the way you would read it aloud ("x squared minus five x"). Never put LaTeX, backslashes or symbols like ^ in the narration.
- Talk about what is on screen at this moment. Continue naturally from the previous beat; do not greet, do not say "in this video".
- Every number you say must agree with the numbers on screen and with the lesson's final answer.

Visual rules:
- Fill only the fields the component has; keep text short so it fits on screen.
- Maths fields are math-mode LaTeX WITHOUT $ signs unless the card says $...$.
- {RULES_JSON}"""


@dataclass
class Trace:
    """Everything one generation did, for the run log."""
    calls: list[dict] = field(default_factory=list)

    def add(self, step: str, messages: list[dict], comps: list[Completion], extra: dict | None = None):
        self.calls.append({"step": step, "messages": messages,
                           "completions": [c.to_dict() for c in comps], **(extra or {})})

    @property
    def llm_seconds(self) -> float:
        # concurrent samples overlap: count the slowest of each call group
        return sum(max((c["wall_s"] for c in call["completions"]), default=0) for call in self.calls)

    @property
    def tokens(self) -> tuple[int, int]:
        p = sum(c["prompt_tokens"] for call in self.calls for c in call["completions"])
        g = sum(c["completion_tokens"] for call in self.calls for c in call["completions"])
        return p, g


def _topic_msg(topic: Topic) -> str:
    return f"Course / exam: {topic.level}\nStudent's request: {topic.request}"


def safe_calc(expr: str) -> float | None:
    """Evaluate a model-written arithmetic expression (no eval: safe_math)."""
    from research.safe_math import safe_sympify
    if not expr:
        return None
    try:
        v = safe_sympify(expr)
        if v.free_symbols:
            return None
        f = complex(sp.N(v))
        return float(f.real) if abs(f.imag) < 1e-9 else None
    except Exception:
        return None


# --- staged ------------------------------------------------------------------------

def _plan_value(plan: dict) -> float | None:
    """The plan's numerical answer: the calculator value if there is one,
    else the first number in final_answer."""
    v = plan.get("answer_calc_value")
    if isinstance(v, (int, float)) and abs(v) < 1e12:
        return float(v)
    import re as _re
    m = _re.search(r"-?\d+(?:\.\d+)?", str(plan.get("final_answer", "")).replace(",", ""))
    return float(m.group()) if m else None


def vote(plans: list[dict]) -> tuple[dict, dict]:
    """Self-consistency: cluster the plans' answers (within 1%), take the
    biggest cluster, and within it the plan with the most beats in range.
    Returns (chosen plan, vote record)."""
    vals = [_plan_value(p) for p in plans]
    clusters: list[list[int]] = []
    for i, v in enumerate(vals):
        if v is None:
            continue
        for c in clusters:
            ref = vals[c[0]]
            if abs(v - ref) <= 0.01 * max(abs(ref), 1e-9) + 1e-9:
                c.append(i)
                break
        else:
            clusters.append([i])
    if clusters:
        best = max(clusters, key=len)
        pick = max(best, key=lambda i: (5 <= len(plans[i]["beats"]) <= 9, -abs(len(plans[i]["beats"]) - 7)))
    else:
        best, pick = [], 0
    return plans[pick], {"values": vals, "cluster_sizes": sorted((len(c) for c in clusters), reverse=True),
                         "chosen": pick, "agreement": (len(best) / len(plans)) if plans else 0}


def make_plan(url: str, topic: Topic, trace: Trace, *, seed: int, temperature: float,
              constrained: bool = True, calc: bool = False, k: int = 1,
              max_tokens: int = 1500) -> dict | None:
    """One plan, or k plans sampled at once and chosen by answer vote."""
    system = PLAN_SYSTEM + (PLAN_CALC_ADDENDUM if calc else "")
    schema = plan_schema()
    if calc:
        schema = json.loads(json.dumps(schema))
        schema["properties"]["answer_calc"] = {"type": "string", "maxLength": 200}
        schema["required"].append("answer_calc")
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": _topic_msg(topic)}]
    comps = chat_many(url, msgs, k, seed0=seed, schema=schema if constrained else None,
                      temperature=temperature, max_tokens=max_tokens)
    plans, errs = [], []
    for c in comps:
        plan, err = parse_json(c.text) if c.ok else (None, c.error)
        errs.append(err)
        if not isinstance(plan, dict) or not isinstance(plan.get("beats"), list):
            continue
        plan["beats"] = [b for b in plan["beats"] if isinstance(b, dict) and b.get("component") in ALLOWED]
        if not plan["beats"]:
            continue
        if calc:
            plan["answer_calc_value"] = safe_calc(str(plan.get("answer_calc", "")))
        plans.append(plan)
    extra = {"parse_errors": errs, "n_plans": len(plans)}
    chosen = None
    if plans:
        chosen, rec = vote(plans) if len(plans) > 1 else (plans[0], {})
        extra["vote"] = rec
        extra["answer_calc_value"] = chosen.get("answer_calc_value")
        if rec:
            chosen["vote"] = rec
    trace.add("plan", msgs, comps, extra)
    return chosen


def _beat_user_msg(topic: Topic, plan: dict, i: int, prev_narration: str | None,
                   feedback: str | None) -> str:
    beats = plan["beats"]
    outline = "\n".join(f"{'>>' if j == i else '  '} {j + 1}. [{b['purpose']}] {b['component']}: {b['idea']}"
                        for j, b in enumerate(beats))
    answer = plan.get("final_answer", "")
    if plan.get("answer_calc_value") is not None:
        answer = f"{answer} (calculator check: {plan['answer_calc_value']:.6g})"
    b = beats[i]
    parts = [
        _topic_msg(topic),
        f"Lesson: {plan.get('title', '')}\nObjective: {plan.get('objective', '')}\n"
        f"Key insight: {plan.get('key_insight', '')}\nMisconception to address: {plan.get('misconception', '')}\n"
        f"Final answer: {answer}",
        f"Beat outline (you write the beat marked >>):\n{outline}",
        f"Previous beat's narration: {prev_narration}" if prev_narration else "This is the first beat.",
        f"Write beat {i + 1}: purpose '{b['purpose']}', idea: {b['idea']}",
        "The component for this beat:\n" + card_text(b["component"]),
    ]
    if feedback:
        parts.append("Your previous attempt at this beat was refused by the validator:\n" + feedback +
                     "\nWrite the beat again so it passes. Keep the same idea; shorten or simplify if needed.")
    return "\n\n".join(parts)


FLAG_TEXT = {
    "narration_latex": "the narration contains LaTeX or maths symbols; the voice would read them aloud. Write the maths in words.",
    "prose_latex": "a text field contains LaTeX commands, which are drawn literally there. Use plain words, or put maths only in maths fields",
    "stray_dollar": "a text field contains a $ sign, which is drawn literally there",
}


def _feedback(sc: BeatScore) -> str:
    lines = [f"- {e}" for e in sc.rung1_errors[:6]]
    lines += [f"- [{f['kind']}] {f['message']}" for f in sc.findings[:6]]
    lines += [f"- {FLAG_TEXT.get(fl.split(':')[0], fl)}{(' (field ' + fl.split(':')[1] + ')') if ':' in fl else ''}"
              for fl in sc.flags]
    return "\n".join(lines) or "- (no detail)"


def fill_beat(url: str, topic: Topic, plan: dict, i: int, prev: str | None, trace: Trace,
              work_dir: Path, *, seed: int, temperature: float, n: int, repair: int,
              constrained: bool = True, max_tokens: int = 900) -> tuple[dict, BeatScore, dict]:
    """Best-of-n samples for beat i, then up to `repair` feedback rounds."""
    comp = plan["beats"][i]["component"]
    schema = beat_schema(comp) if constrained else None
    feedback = None
    best = None
    stats = {"samples": 0, "repair_rounds": 0, "validator_seconds": 0.0}
    for round_ in range(repair + 1):
        msgs = [{"role": "system", "content": BEAT_SYSTEM},
                {"role": "user", "content": _beat_user_msg(topic, plan, i, prev, feedback)}]
        comps = chat_many(url, msgs, n, seed0=seed + 1000 * round_, schema=schema,
                          temperature=temperature, max_tokens=max_tokens)
        cands, errs = [], []
        for k, c in enumerate(comps):
            v, err = parse_json(c.text) if c.ok else (None, c.error)
            beat = {"component": comp, "narration": "", "params": {}}
            if isinstance(v, dict):
                beat["narration"] = v.get("narration", "")
                beat["params"] = v.get("params", {}) if isinstance(v.get("params", {}), dict) else {}
            cands.append(beat)
            errs.append(err)
        bid = f"b{i + 1:02d}"
        tv = time.perf_counter()
        scs = score_many([(b, bid, work_dir / f"{bid}r{round_}s{k}") for k, b in enumerate(cands)])
        scored = []
        for k, (beat, sc, err) in enumerate(zip(cands, scs, errs)):
            if sc.harness_fault:   # retry a machine/library fault once before blaming the model
                sc2 = score_beat(beat, bid, work_dir / f"{bid}r{round_}s{k}-retry")
                sc = sc2 if not sc2.harness_fault else sc
            if err:
                sc.rung1_errors.insert(0, err)
            scored.append((beat, sc))
        stats["validator_seconds"] += time.perf_counter() - tv
        stats["samples"] += len(comps)
        trace.add(f"beat{i + 1}.r{round_}", msgs, comps,
                  {"scores": [s.to_dict() for _, s in scored]})
        cand = max(scored, key=lambda t: selection_key(t[1]))
        if best is None or selection_key(cand[1]) > selection_key(best[1]):
            best = cand
        if best[1].clean:
            break
        if round_ < repair:
            stats["repair_rounds"] += 1
            feedback = _feedback(best[1])
    return best[0], best[1], stats


def generate_staged(url: str, topic: Topic, work_dir: Path, *, seed: int = 0,
                    temperature: float = 0.7, n: int = 1, repair: int = 0,
                    constrained: bool = True, calc: bool = False, plans: int = 1) -> dict:
    trace = Trace()
    t0 = time.perf_counter()
    plan = make_plan(url, topic, trace, seed=seed, temperature=temperature,
                     constrained=constrained, calc=calc, k=plans)
    beats, scores, per_beat = [], [], []
    if plan:
        prev = None
        for i in range(len(plan["beats"])):
            beat, sc, st = fill_beat(url, topic, plan, i, prev, trace, work_dir, seed=seed + 17 * (i + 1),
                                     temperature=temperature, n=n, repair=repair, constrained=constrained)
            beat["purpose"] = plan["beats"][i].get("purpose")
            beats.append(beat)
            scores.append(sc)
            per_beat.append(st)
            prev = beat.get("narration")
    return {"plan": plan, "beats": beats, "scores": [s.to_dict() for s in scores],
            "per_beat": per_beat, "trace": trace, "wall_s": time.perf_counter() - t0}


# --- oneshot -----------------------------------------------------------------------

ONESHOT_SYSTEM = PLAN_SYSTEM.replace(
    "You do not draw anything. You choose, for each beat, one visual component from a fixed library and say what the beat must teach. Another step writes the narration and fills the component.",
    "You write the whole video at once: for each beat, the narration and one visual component with its params."
) + "\n\nNarration rules:" + BEAT_SYSTEM.split("Narration rules:", 1)[1] \
  + "\n\nComponent details:\n" + "\n".join(card_text(c) for c in ALLOWED)


def generate_oneshot(url: str, topic: Topic, work_dir: Path, *, seed: int = 0,
                     temperature: float = 0.7, constrained: bool = True,
                     max_tokens: int = 5000) -> dict:
    trace = Trace()
    t0 = time.perf_counter()
    msgs = [{"role": "system", "content": ONESHOT_SYSTEM}, {"role": "user", "content": _topic_msg(topic)}]
    c = chat(url, msgs, schema=oneshot_schema() if constrained else None,
             temperature=temperature, seed=seed, max_tokens=max_tokens)
    v, err = parse_json(c.text) if c.ok else (None, c.error)
    trace.add("oneshot", msgs, [c], {"parse_error": err})
    beats, scores = [], []
    if isinstance(v, dict) and isinstance(v.get("beats"), list):
        for i, b in enumerate(v["beats"]):
            if not isinstance(b, dict):
                continue
            beat = {"component": b.get("component"), "narration": b.get("narration", ""),
                    "params": b.get("params", {}) if isinstance(b.get("params"), dict) else {}}
            sc = score_beat(beat, f"b{i + 1:02d}", work_dir / f"b{i + 1:02d}")
            beats.append(beat)
            scores.append(sc)
    plan = {"title": v.get("title", "") if isinstance(v, dict) else "", "beats": []}
    return {"plan": plan, "beats": beats, "scores": [s.to_dict() for s in scores],
            "per_beat": [], "trace": trace, "wall_s": time.perf_counter() - t0}
