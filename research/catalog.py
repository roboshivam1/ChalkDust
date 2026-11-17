"""What a model is shown about the component library, and the JSON schemas
its outputs are constrained to.

The cards are short on purpose: a small model reads one card per beat (the
component its plan chose), not the whole 21k-character schema dump.

v1 scope: Callout and ZoomHighlight are left out because they act on a
carried artifact (carry_in), which makes a beat depend on an earlier one;
RawScene is left out because the model must never write Manim (D-001).
That leaves 17 components.

Authorship note: card wording was written by Claude from the component
module docstrings and Params schemas. Cards are model INPUTS (prompt text),
never training targets.
"""

from __future__ import annotations

import copy
import json
from functools import lru_cache

import chalkdust.scenes.components  # noqa: F401  (registers every component)
from chalkdust.scenes.components.base import get_component

# name -> (what it shows, when to use it, how to fill it)
CARDS: dict[str, tuple[str, str, str]] = {
    "TitleCard": (
        "A big title with an optional small kicker line above and subtitle below.",
        "The opening beat, or a section break. Never more than two per video.",
        "title: 2-7 words. kicker: the course or exam, e.g. 'JEE Main - Mechanics'. "
        "subtitle: one short line saying what the viewer will learn."),
    "BulletReveal": (
        "A heading and 1-6 short points revealed one by one.",
        "A plan of attack, a list of conditions, a recap.",
        "items: 2-5 points, each under 10 words, plain text (no LaTeX). heading: optional, 1-5 words."),
    "SplitCompare": (
        "Two cards side by side (left, right), each a title plus a line of text and/or maths, "
        "with an optional verdict line underneath.",
        "Contrasting two cases, a right vs a wrong method, before vs after, a misconception vs the truth.",
        "left/right: {title, body?, math?}; title 1-4 words; body one short sentence; math is "
        "math-mode LaTeX without $ signs. verdict: one short line. emphasis: left|right|verdict."),
    "EquationDerivation": (
        "Algebra steps stacked top to bottom, each line grown out of the previous one, "
        "with optional margin annotations.",
        "Any multi-step algebra or calculus manipulation.",
        "steps: 2-6 lines, each ONE line of math-mode LaTeX without $ signs (e.g. "
        "'x^2 - 5x + 6 = 0'). Every line must follow correctly from the line before. "
        "annotations: [{step, text}] where step is the 0-based index of the line it explains; "
        "text is 2-6 plain words."),
    "GraphPlot": (
        "Axes with up to 3 plotted functions of x and up to 4 labelled marker points on a curve.",
        "Showing roots, intercepts, maxima/minima, how a function behaves.",
        "functions: [{expr, name?}] where expr is in the ONE variable x only (no other letters: put numbers in for "
        "every other quantity), using + - * / ^, sin cos tan exp log sqrt abs, pi, e (e.g. 'x^2 - 5*x + 6'); "
        "always write * for multiplication (5*x, never 5x). x_range: [min, max] "
        "covering the interesting part. markers: [{x, label?}] at points the narration names."),
    "NumberLineWalk": (
        "A number line with a walker that is placed, jumps along arcs, or highlights intervals.",
        "Counting steps, signed changes, inequalities and intervals.",
        "range: [min, max] or [min, max, step]. steps: first {at, label?}, then {to, label?} "
        "jumps and/or {interval: [lo, hi], closed?: [bool, bool], label?}. Labels are short plain text."),
    "VectorField": (
        "Arrows showing a 2D vector field F(x, y).",
        "Fields, flows, forces that vary with position.",
        "field_fn: {x: expr, y: expr} in x and y, e.g. {'x': '-y', 'y': 'x'}."),
    "GeometryConstruct": (
        "Named points, segments and circles, built step by step like a ruler-and-compass construction.",
        "Geometry problems: triangles, circles, constructions, angles.",
        "shapes: start with points {kind:'point', name, at:[x, y]} in your own units, then "
        "segments/polygons referring to point names. construction: ordered steps referring to "
        "names, each with an optional short note."),
    "UnitBreakdown": (
        "A quantity's unit shown equal to a product of simpler units, each labelled.",
        "Dimensional analysis, checking units of an answer, SI base units.",
        "quantity: {unit, label}; decomposition: [{unit, label}]. unit is math-mode LaTeX "
        "such as '\\\\mathrm{kg}' or '\\\\mathrm{s}^{-2}'; label is plain words."),
    "CodeWalk": (
        "A short code listing with lines highlighted in turn.",
        "Showing an algorithm's code and walking through it.",
        "language: e.g. 'python'. source: at most ~12 short lines, real newlines. highlights: "
        "[{start, end?}] 1-based line numbers, in the order the narration discusses them."),
    "DataStructureViz": (
        "An array, stack, binary tree or graph, then operations on it (highlight, swap, set, push, pop, insert).",
        "Tracing an algorithm on a data structure: searching, sorting, stacks, trees, graphs.",
        "kind: array|stack|tree|graph. initial: the starting values. operations: in narration order; "
        "for an array use {op:'highlight', at:[indices]} to show the elements being looked at."),
    "StepTrace": (
        "A table of variable values, one row per step of an algorithm, revealed row by row.",
        "Tracing loops and algorithms: the values of lo, hi, mid, i, sum ... at each step.",
        "variables: 1-6 short names. frames: 1-8 rows, each a list with one value per variable."),
    "BoxFlow": (
        "Labelled boxes joined by arrows, laid out left to right.",
        "Processes, pipelines, cause and effect, a sequence of stages.",
        "nodes: [{id, label}] with 2-7 boxes, labels 1-4 words. edges: [{source, target}] using node ids."),
    "ProblemStatement": (
        "The question, formatted: a paragraph, then GIVEN values and what to FIND.",
        "Stating a numerical or exam problem before solving it.",
        "text: 1-3 sentences of prose with inline maths written as $...$. given: 1-6 items, each "
        "usually inline maths like '$m = 2\\\\,\\\\mathrm{kg}$'. find: a short phrase."),
    "FreeBodyDiagram": (
        "A body as a labelled box with force arrows drawn outward, appearing one by one.",
        "Mechanics: identifying every force on a body before writing Newton's laws.",
        "body: short name. forces: [{label, angle, magnitude}] where label is math-mode LaTeX "
        "(e.g. 'mg', 'N', 'f_k'), angle in degrees (0 = right, 90 = up, 270 = down), "
        "magnitude a positive number proportional to the force."),
    "SolutionStep": (
        "One numbered step of a worked solution: the claim at the top, the work lines in the "
        "middle, the justification at the bottom.",
        "Each step of solving a numerical problem.",
        "n: the step number. claim: what this step establishes, one short sentence. work: 1-4 "
        "lines; by default each is math-mode LaTeX without $ signs. justification: why the work "
        "is allowed, one short sentence."),
    "AnswerBox": (
        "The final answer written large inside a box.",
        "The last step of a solved problem.",
        "value: the answer as math-mode LaTeX without $ signs, e.g. 'R = 20\\\\sqrt{3} \\\\approx 34.6'. "
        "units: math-mode LaTeX like '\\\\mathrm{m}' or null. option: A-D only for multiple choice, else null."),
}

ALLOWED: tuple[str, ...] = tuple(CARDS)

PURPOSES: tuple[str, ...] = (
    "hook",               # why this matters / the question that makes it interesting
    "problem",            # state the problem or the setup
    "intuition",          # the picture or idea before the formal step
    "definition",         # a rule, law or definition
    "worked_step",        # one step of working
    "misconception",      # a tempting wrong idea, and why it is wrong
    "check",              # a quick check for understanding / sanity check
    "answer",             # the final answer
    "recap",              # what to remember
)


def _hoist(params_schema: dict) -> tuple[dict, dict]:
    """Split a pydantic schema into (body, $defs) so it can be nested under
    another object while its '#/$defs/...' refs still resolve at the root."""
    s = copy.deepcopy(params_schema)
    defs = s.pop("$defs", {})
    s.pop("title", None)
    return s, defs


@lru_cache(maxsize=None)
def params_schema(component: str) -> dict:
    return get_component(component).Params.model_json_schema()


NARRATION = {"type": "string", "minLength": 60, "maxLength": 420}


@lru_cache(maxsize=None)
def beat_schema(component: str) -> dict:
    """What one beat-fill call must output: the narration, then the params."""
    body, defs = _hoist(params_schema(component))
    s = {"type": "object", "additionalProperties": False,
         "properties": {"narration": NARRATION, "params": body},
         "required": ["narration", "params"]}
    if defs:
        s["$defs"] = defs
    return s


def plan_schema(min_beats: int = 5, max_beats: int = 9) -> dict:
    """What the planning call must output (the lesson plan)."""
    return {
        "type": "object", "additionalProperties": False,
        "properties": {
            "title": {"type": "string", "minLength": 3, "maxLength": 60},
            "objective": {"type": "string", "minLength": 10, "maxLength": 200},
            "key_insight": {"type": "string", "minLength": 10, "maxLength": 240},
            "misconception": {"type": "string", "minLength": 10, "maxLength": 240},
            "final_answer": {"type": "string", "maxLength": 120},
            "beats": {
                "type": "array", "minItems": min_beats, "maxItems": max_beats,
                "items": {
                    "type": "object", "additionalProperties": False,
                    "properties": {
                        "purpose": {"enum": list(PURPOSES)},
                        "idea": {"type": "string", "minLength": 10, "maxLength": 200},
                        "component": {"enum": list(ALLOWED)},
                    },
                    "required": ["purpose", "idea", "component"],
                },
            },
        },
        "required": ["title", "objective", "key_insight", "misconception", "final_answer", "beats"],
    }


def oneshot_schema(min_beats: int = 5, max_beats: int = 9) -> dict:
    """A whole video in one call: beats is an array whose items are one of the
    17 component-specific beat schemas (discriminated by a const component)."""
    defs: dict = {}
    variants = []
    for name in ALLOWED:
        body, d = _hoist(params_schema(name))
        for k, v in d.items():
            # $defs names collide across components (e.g. 'Side'); prefix them.
            defs[f"{name}__{k}"] = v
        body_txt = json.dumps(body)
        sub_defs_txt = {f"{name}__{k}": json.dumps(v) for k, v in d.items()}
        for k in d:
            body_txt = body_txt.replace(f'"#/$defs/{k}"', f'"#/$defs/{name}__{k}"')
        for nk, txt in sub_defs_txt.items():
            for k in d:
                txt = txt.replace(f'"#/$defs/{k}"', f'"#/$defs/{name}__{k}"')
            defs[nk] = json.loads(txt)
        variants.append({"type": "object", "additionalProperties": False,
                         "properties": {"narration": NARRATION, "component": {"const": name},
                                        "params": json.loads(body_txt)},
                         "required": ["narration", "component", "params"]})
    s = {"type": "object", "additionalProperties": False,
         "properties": {"title": {"type": "string", "minLength": 3, "maxLength": 60},
                        "beats": {"type": "array", "minItems": min_beats, "maxItems": max_beats,
                                  "items": {"anyOf": variants}}},
         "required": ["title", "beats"]}
    if defs:
        s["$defs"] = defs
    return s


def card_text(component: str) -> str:
    shows, when, fill = CARDS[component]
    return f"{component}: {shows}\n  Use for: {when}\n  How to fill: {fill}"


def catalog_text() -> str:
    return "\n".join(f"- {n}: {CARDS[n][0]} Use for: {CARDS[n][1]}" for n in ALLOWED)
