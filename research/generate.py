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
              anim     run 3: the animation components (research/animate.py).
                       An animation is one plan item, expanded by code into one
                       beat per phase; the setup is filled once and carried;
                       later phases are told the numbers on screen. With calc,
                       the calculator also takes derivative/limit/integral and
                       its value decides the answer shown and spoken. Plans of
                       8-11 items, narration of 45-62 words: 3-5 minute videos.

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

from research import animate
from research.catalog import (ALLOWED, beat_schema, card_text, catalog_text,
                              oneshot_schema, plan_schema)
from research.llm import Completion, chat, chat_many, parse_json
from research.score import BeatScore, score_beat, score_many, selection_key
from research.topics import Topic

RULES_JSON = (
    "Output one JSON object and nothing else. Inside JSON strings every backslash "
    "must be doubled: write \\\\frac{1}{2}, \\\\theta, \\\\sqrt{2}, \\\\mathrm{m}. "
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

Output shape (exactly these keys):
{{"title": "...", "objective": "...", "key_insight": "...", "misconception": "...", "final_answer": "...",
 "beats": [{{"purpose": "hook|problem|intuition|definition|worked_step|misconception|check|answer|recap", "idea": "what this beat must teach", "component": "ComponentName"}}, ...]}}

The component library:
{catalog_text()}"""

PLAN_CALC_ADDENDUM = """
- answer_calc: the final numerical answer written as ONE arithmetic expression that a calculator can evaluate, using numbers, + - * / ** ( ), sqrt(), sin() cos() tan() (radians; write 30 degrees as 30*deg), pi, E. Example: "15*cos(40*deg)". Use "" if the answer is not a single number."""

PLAN_WORK_ADDENDUM = """
- working: BEFORE you write final_answer, solve the problem in a few short numbered steps with the actual numbers (formula, substitution, arithmetic). final_answer must be the result of this working. Use "n/a" for a pure explanation."""

# --- anim=1 (run 3) --------------------------------------------------------------------
PLAN_SYSTEM_ANIM = PLAN_SYSTEM.replace(
    "- 5 to 9 beats. One idea per beat",
    "- 8 to 11 beats in your plan (an animation counts as one). One idea per beat") + "\n" + animate.catalog_text()

PLAN_ANIM_RULES = """
- Animations: the components marked ANIMATION are the best visuals in the library. Each is ONE beat in your plan but plays as 3-4 video beats, one per phase, and every number it shows is computed by code. If an animation shows the topic's central idea, use it where the student should SEE that idea. At most 2 animations."""

PLAN_CALC_ANIM_ADDENDUM = """
- answer_calc may also be ONE calculus call, with f written in the one variable x: derivative(f, a) for f'(a), limit(f, a) for the limit of f as x approaches a, integral(f, a, b) for the definite integral of f from a to b. Example: "integral(x^2/4 + 1, 0, 3)"."""

ANSWER_DECIDED = "(computed by the calculator: say and show exactly this)"

BEAT_SYSTEM = f"""You write one beat of a ChalkDust explainer video: the narration the student hears, and the params of the one visual component on screen.

Narration rules:
- 30 to 60 words, spoken English, warm and precise, like a good teacher at a whiteboard.
- Say maths the way you would read it aloud ("x squared minus five x"). Never put LaTeX, backslashes or symbols like ^ in the narration.
- Talk about what is on screen at this moment. Continue naturally from the previous beat; do not greet, do not say "in this video".
- Every number you say must agree with the numbers on screen and with the lesson's final answer.

Visual rules:
- Fill only the fields the component has; keep text short so it fits on screen.
- Maths fields are math-mode LaTeX WITHOUT $ signs unless the card says $...$.
- {RULES_JSON}

Output shape (exactly these two keys; params holds the component's fields from its card):
{{"narration": "...", "params": {{...}}}}"""

BEAT_ANIM_RULES = """

Run-3 rules (they override the word count above):
- 45 to 62 words of narration.
- If FACTS are given, they are the numbers on screen in this beat: say those numbers (in words), never different ones.
- When you say the lesson's final answer, say exactly the value given as the final answer, reading its digits aloud. Never say a value for it that you worked out yourself.
- cues (animations only): for each moment, copy 2 to 5 consecutive words from YOUR narration where that moment should start. Leave a moment out if unsure."""


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


def _plan_schema(calc: bool, work: bool, anim: bool = False) -> dict:
    """The plan schema with optional fields in GENERATION order: with grammar-
    constrained decoding the property order is the order the model writes, so
    'working' (and 'answer_calc') must come before 'final_answer' for the
    answer to be reasoned rather than guessed (E1b: staged plans guessed)."""
    base = plan_schema(8, 11) if anim else plan_schema()
    if anim:
        base["properties"]["beats"]["items"]["properties"]["component"]["enum"] += animate.offered()
    props, req = {}, list(base["required"])
    for k, v in base["properties"].items():
        if k == "final_answer":
            if work:
                props["working"] = {"type": "string", "minLength": 10, "maxLength": 700}
                req.insert(req.index("final_answer"), "working")
            if calc:
                props["answer_calc"] = {"type": "string", "maxLength": 200}
                req.insert(req.index("final_answer"), "answer_calc")
        props[k] = v
    return {**base, "properties": props, "required": req}


def make_plan(url: str, topic: Topic, trace: Trace, *, seed: int, temperature: float,
              constrained: bool = True, calc: bool = False, k: int = 1, work: bool = False,
              anim: bool = False, max_tokens: int = 1800) -> dict | None:
    """One plan, or k plans sampled at once and chosen by answer vote."""
    system = ((PLAN_SYSTEM_ANIM + PLAN_ANIM_RULES) if anim else PLAN_SYSTEM) \
        + ((PLAN_CALC_ADDENDUM + (PLAN_CALC_ANIM_ADDENDUM if anim else "")) if calc else "") \
        + (PLAN_WORK_ADDENDUM if work else "")
    schema = _plan_schema(calc, work, anim)
    allowed = ALLOWED + tuple(animate.offered()) if anim else ALLOWED
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": _topic_msg(topic)}]
    comps = chat_many(url, msgs, k, seed0=seed, schema=schema if constrained else None,
                      temperature=temperature, max_tokens=max_tokens)
    plans, errs = [], []
    for c in comps:
        plan, err = parse_json(c.text) if c.ok else (None, c.error)
        errs.append(err)
        if not isinstance(plan, dict) or not isinstance(plan.get("beats"), list):
            continue
        plan["beats"] = [b for b in plan["beats"] if isinstance(b, dict) and b.get("component") in allowed]
        if not plan["beats"]:
            continue
        if calc:
            plan["answer_calc_value"] = (animate.calc_value if anim else safe_calc)(str(plan.get("answer_calc", "")))
        plans.append(plan)
    extra = {"parse_errors": errs, "n_plans": len(plans)}
    chosen = None
    if plans:
        chosen, rec = vote(plans) if len(plans) > 1 else (plans[0], {})
        extra["vote"] = rec
        extra["answer_calc_value"] = chosen.get("answer_calc_value")
        if rec:
            chosen["vote"] = rec
        if anim:
            chosen = animate.expand_plan(chosen)
            v = chosen.get("answer_calc_value")
            if calc and v is not None:   # the calculator decides the answer
                chosen["final_answer_model"] = chosen.get("final_answer", "")
                chosen["final_answer"] = animate.answer_text(v)[1]
    trace.add("plan", msgs, comps, extra)
    return chosen


def _beat_user_msg(topic: Topic, plan: dict, i: int, prev_narration: str | None,
                   feedback: str | None, extra: str | None = None) -> str:
    beats = plan["beats"]
    outline = "\n".join(f"{'>>' if j == i else '  '} {j + 1}. [{b['purpose']}] {b['component']}: {b['idea']}"
                        for j, b in enumerate(beats))
    answer = plan.get("final_answer", "")
    if plan.get("final_answer_model") is not None:   # anim + calc: decided by the calculator
        answer = f"{answer} {ANSWER_DECIDED}"
    elif plan.get("answer_calc_value") is not None:
        answer = f"{answer} (calculator check: {plan['answer_calc_value']:.6g})"
    b = beats[i]
    parts = [
        _topic_msg(topic),
        f"Lesson: {plan.get('title', '')}\nObjective: {plan.get('objective', '')}\n"
        f"Key insight: {plan.get('key_insight', '')}\nMisconception to address: {plan.get('misconception', '')}\n"
        f"Final answer: {answer}" + (f"\nWorking: {plan['working']}" if plan.get("working") else ""),
        f"Beat outline (you write the beat marked >>):\n{outline}",
        f"Previous beat's narration: {prev_narration}" if prev_narration else "This is the first beat.",
        f"Write beat {i + 1}: purpose '{b['purpose']}', idea: {b['idea']}",
        "The component for this beat:\n" + (animate.card_text(b["component"]) if b.get("anim_phase")
                                              else card_text(b["component"])),
    ]
    if extra:
        parts.append(extra)
    if feedback:
        parts.append("Your previous attempt at this beat was refused by the validator:\n" + feedback +
                     "\nWrite the beat again so it passes. Keep the same idea; shorten or simplify if needed.")
    return "\n\n".join(parts)


FLAG_TEXT = {
    "narration_latex": "the narration contains LaTeX or maths symbols; the voice would read them aloud. Write the maths in words.",
    "prose_latex": "a text field contains LaTeX commands, which are drawn literally there. Use plain words, or put maths only in maths fields",
    "stray_dollar": "a text field contains a $ sign, which is drawn literally there",
    "answer_not_spoken": "the narration does not say the lesson's final answer; say it exactly as given, reading its digits",
    "setup_disagrees": "the animation computes a different value from the lesson's final answer: use the function and the point (or interval) from the student's question",
}


def _feedback(sc: BeatScore) -> str:
    lines = [f"- {e}" for e in sc.rung1_errors[:6]]
    lines += [f"- [{f['kind']}] {f['message']}" for f in sc.findings[:6]]
    lines += [f"- {FLAG_TEXT.get(fl.split(':')[0], fl)}{(' (field ' + fl.split(':')[1] + ')') if ':' in fl else ''}"
              for fl in sc.flags]
    return "\n".join(lines) or "- (no detail)"


def _anim_flags(beat: dict, comp: str, ctx: dict | None, decided: float | None,
                kind: str | None, answer_beat: bool) -> list[str]:
    """anim=1 quality flags: the answer beat must say the decided answer; an
    animation's own headline number must agree with it when it computes the
    same kind of quantity (research/animate.py)."""
    out = []
    if decided is None:
        return out
    if answer_beat and not animate.says(str(beat.get("narration", "")), decided):
        out.append("answer_not_spoken")
    if ctx and ctx["setup"] is None and animate.HEADLINE_CALC.get(comp) == kind:
        try:
            _, head = animate.facts(comp, animate.setup_of(beat["params"]))
            if abs(head - decided) > 1e-6 * max(1.0, abs(decided)):
                out.append("setup_disagrees")
        except Exception:   # a setup the component refuses: the validator says why
            pass
    return out


def fill_beat(url: str, topic: Topic, plan: dict, i: int, prev: str | None, trace: Trace,
              work_dir: Path, *, seed: int, temperature: float, n: int, repair: int,
              constrained: bool = True, max_tokens: int = 900, anim: bool = False,
              anim_ctx: dict | None = None, extra: str | None = None) -> tuple[dict, BeatScore, dict]:
    """Best-of-n samples for beat i, then up to `repair` feedback rounds.
    anim: run-3 mode; anim_ctx: {phase, setup} for an animation beat (setup
    None for its first phase, which fills it)."""
    comp = plan["beats"][i]["component"]
    if anim_ctx:
        schema = animate.beat_schema(comp, anim_ctx["phase"], anim_ctx["setup"] is None)
    elif anim:
        schema = {**beat_schema(comp), "properties": {**beat_schema(comp)["properties"],
                                                      "narration": animate.NARRATION_ANIM}}
    else:
        schema = beat_schema(comp)
    schema = schema if constrained else None
    decided = plan.get("answer_calc_value") if (anim and plan.get("final_answer_model") is not None) else None
    kind = animate.calc_kind(str(plan.get("answer_calc", ""))) if decided is not None else None
    answer_beat = plan["beats"][i].get("purpose") == "answer" or comp == "AnswerBox"
    system = BEAT_SYSTEM + BEAT_ANIM_RULES if anim else BEAT_SYSTEM
    feedback = None
    best = None
    stats = {"samples": 0, "repair_rounds": 0, "validator_seconds": 0.0}
    for round_ in range(repair + 1):
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": _beat_user_msg(topic, plan, i, prev, feedback, extra)}]
        comps = chat_many(url, msgs, n, seed0=seed + 1000 * round_, schema=schema,
                          temperature=temperature, max_tokens=max_tokens)
        cands, errs = [], []
        for k, c in enumerate(comps):
            v, err = parse_json(c.text) if c.ok else (None, c.error)
            beat = {"component": comp, "narration": "", "params": {}}
            if isinstance(v, dict):
                beat["narration"] = v.get("narration", "")
                beat["params"] = v.get("params", {}) if isinstance(v.get("params", {}), dict) else {}
            if anim_ctx:
                beat["params"] = animate.merge_params(comp, anim_ctx["phase"], beat["params"], anim_ctx["setup"])
            elif decided is not None and comp == "AnswerBox":
                beat["params"] = {**beat["params"],
                                  "value": animate.answerbox_value(beat["params"].get("value"), decided)}
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
            if anim:
                sc.flags += _anim_flags(beat, comp, anim_ctx, decided, kind, answer_beat)
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
    b0, s0 = best
    if (anim_ctx and s0.level < 3 and not s0.rung1_errors and s0.findings
            and all(f["kind"] == "cue" for f in s0.findings) and b0["params"].get("cues")):
        # Mechanical repair: cues the narration does not support are dropped;
        # the moments then split by weight (chalkdust/core/cues.py).
        b1 = {**b0, "params": {**b0["params"], "cues": {}}}
        s1 = score_beat(b1, f"b{i + 1:02d}", work_dir / f"b{i + 1:02d}-nocues")
        s1.flags += _anim_flags(b1, comp, anim_ctx, decided, kind, answer_beat)
        stats["cues_dropped"] = 1
        if selection_key(s1) > selection_key(s0):
            best = (b1, s1)
    return best[0], best[1], stats


def generate_staged(url: str, topic: Topic, work_dir: Path, *, seed: int = 0,
                    temperature: float = 0.7, n: int = 1, repair: int = 0,
                    constrained: bool = True, calc: bool = False, plans: int = 1,
                    plan_override: dict | None = None, work: bool = False, anim: bool = False) -> dict:
    """plan_override: a plan made earlier (by any model) to fill instead of
    planning here -- the split pipeline, big model plans / small model fills."""
    trace = Trace()
    t0 = time.perf_counter()
    if plan_override is not None:
        plan = json.loads(json.dumps(plan_override))
        trace.add("plan", [], [], {"override": True})
    else:
        plan = make_plan(url, topic, trace, seed=seed, temperature=temperature,
                         constrained=constrained, calc=calc, k=plans, work=work, anim=anim)
    beats, scores, per_beat = [], [], []
    if plan:
        prev = None
        setups: dict[int, dict] = {}   # anim_id -> the setup its first phase filled
        for i in range(len(plan["beats"])):
            ctx, extra = _anim_context(plan["beats"][i], setups) if anim else (None, None)
            beat, sc, st = fill_beat(url, topic, plan, i, prev, trace, work_dir, seed=seed + 17 * (i + 1),
                                     temperature=temperature, n=n, repair=repair, constrained=constrained,
                                     anim=anim, anim_ctx=ctx, extra=extra)
            if ctx and ctx["setup"] is None:
                setups[plan["beats"][i]["anim_id"]] = animate.setup_of(beat["params"])
                if anim and calc:
                    _decide_from_animation(plan, plan["beats"][i]["component"],
                                           setups[plan["beats"][i]["anim_id"]], trace)
            beat["purpose"] = plan["beats"][i].get("purpose")
            beats.append(beat)
            scores.append(sc)
            per_beat.append(st)
            prev = beat.get("narration")
    return {"plan": plan, "beats": beats, "scores": [s.to_dict() for s in scores],
            "per_beat": per_beat, "trace": trace, "wall_s": time.perf_counter() - t0}


def _decide_from_animation(plan: dict, comp: str, setup: dict, trace: Trace | None = None) -> None:
    """anim + calc. A calculation that is not a derivative/limit/integral call
    decides nothing: R3-anim-b's three plans all wrote their own number there
    ("1/2", "2", "3.75"; the last one wrong), so the calculator only echoed
    the model. When the lesson has exactly one animation that computes such a
    quantity, the calculator computes it from that animation's setup (the
    value its last phase shows on screen), and that is the answer said and
    shown from here on."""
    if animate.calc_kind(str(plan.get("answer_calc", ""))) is not None:
        return
    heads = {b.get("anim_id") for b in plan["beats"]
             if b.get("anim_phase") and b["component"] in animate.HEADLINE_CALC}
    call = animate.headline_call(comp, setup) if len(heads) == 1 else None
    v = animate.calc_value(call) if call else None
    if v is None:
        return
    plan.setdefault("final_answer_model", plan.get("final_answer", ""))
    plan["answer_calc_model"] = plan.get("answer_calc")
    plan["answer_calc"], plan["answer_calc_value"] = call, v
    plan["final_answer"] = animate.answer_text(v)[1]
    plan["answer_source"] = "animation"
    if trace is not None:
        trace.add("decide", [], [], {"call": call, "value": v, "model_calc": plan["answer_calc_model"]})


def _anim_context(b: dict, setups: dict[int, dict]) -> tuple[dict | None, str | None]:
    """For an animation beat: its fill context and the prompt text naming the
    phase, the carried setup and the FACTS on screen."""
    if not b.get("anim_phase"):
        return None, None
    comp, phase = b["component"], b["anim_phase"]
    names = animate.phases(comp)
    k = names.index(phase)
    moments = ", ".join(m.name for m in animate.get_component(comp).PHASES[phase])
    head = (f"This beat is phase {k + 1} of {len(names)} ({phase}) of the {comp} animation. "
            f"Its moments, in order: {moments}.")
    if b.get("anim_first"):
        return {"phase": phase, "setup": None}, head + (
            " You choose the setup (params); it is used unchanged for every phase of this animation, "
            "so pick the function, point and range the lesson is about. Write only this phase's narration.")
    setup = setups.get(b["anim_id"], {})
    try:
        facts = animate.facts(comp, setup)[0][phase]
    except Exception:
        facts = []
    text = head + f" The setup is fixed by the earlier beat: {json.dumps(setup)}. In params, fill only cues."
    if facts:
        text += "\nFACTS on screen in this beat:\n- " + "\n- ".join(facts)
    return {"phase": phase, "setup": setup}, text


# --- oneshot -----------------------------------------------------------------------

_SHAPE = "Output shape ("
ONESHOT_SYSTEM = PLAN_SYSTEM.split(_SHAPE)[0].rstrip().replace(
    "You do not draw anything. You choose, for each beat, one visual component from a fixed library and say what the beat must teach. Another step writes the narration and fills the component.",
    "You write the whole video at once: for each beat, the narration and one visual component with its params."
) + "\n\nThe component library:" + PLAN_SYSTEM.split("The component library:", 1)[1] \
  + "\n\nNarration rules:" + BEAT_SYSTEM.split("Narration rules:", 1)[1].split(_SHAPE)[0].rstrip() \
  + "\n\nComponent details:\n" + "\n".join(card_text(c) for c in ALLOWED) \
  + ("\n\nOutput shape (exactly these keys; params holds the component's fields from its card):\n"
     '{"title": "...", "beats": [{"component": "ComponentName", "narration": "...", "params": {...}}, ...]}')


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


# Records carry this hash: an SFT set is only valid for the prompts it was made with.
import hashlib as _hashlib
PROMPT_HASH = _hashlib.sha256("\x1e".join([PLAN_SYSTEM, PLAN_CALC_ADDENDUM, PLAN_WORK_ADDENDUM, BEAT_SYSTEM, ONESHOT_SYSTEM])
                              .encode("utf-8")).hexdigest()[:12]
# anim=1 runs carry their own hash; the one above is unchanged for every other method.
PROMPT_HASH_ANIM = _hashlib.sha256("\x1e".join([
    PLAN_SYSTEM_ANIM, PLAN_ANIM_RULES, PLAN_CALC_ADDENDUM, PLAN_CALC_ANIM_ADDENDUM, PLAN_WORK_ADDENDUM,
    BEAT_SYSTEM, BEAT_ANIM_RULES, ANSWER_DECIDED, json.dumps({c: animate.CARDS[c] for c in animate.offered()}, sort_keys=True),
    json.dumps(animate.NARRATION_ANIM, sort_keys=True)]).encode("utf-8")).hexdigest()[:12]
