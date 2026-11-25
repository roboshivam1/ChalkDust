"""The fixed topic sets for every ChalkDust small-model experiment.

FROZEN on 2026-11-17, before any teacher data was generated. Do not edit a
topic in place: results in research/results/*.jsonl refer to these ids. Add
new topics with new ids instead.

* DEV      -- the comparison set for untuned models, methods and decoding.
* HELDOUT  -- never shown to the teacher or the student during training; the
              tuned-vs-untuned comparison is reported on these only.
* FAMILIES -- the problem family of every DEV and HELDOUT topic. The training
              topic pool (research/train_topics.py) must not contain any of
              these families, so a tuned model cannot have seen a near-twin
              of an evaluation problem.

Answer keys are computed with sympy here, never typed as decimals, so a wrong
key cannot be a typo. A topic without a single checkable answer (an
explanation topic) has key=None and is scored on validity and structure only.

Authorship note: the topic wording below was written by Claude (the research
session's lead model). Topics are model INPUTS, not training targets.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import sympy as sp

x = sp.Symbol("x")


@dataclass(frozen=True)
class Key:
    """A checkable final answer.

    kind:
      number -- one real value (units given separately; compared within rel_tol)
      set    -- a finite set of real values (roots), order-free
      expr   -- an expression in x (derivative); compared numerically at
                sample points
      antiderivative -- an expression in x up to an additive constant
                (indefinite integral); compared by differentiating
    """

    kind: str
    value: object
    units: str | None = None
    rel_tol: float = 0.02  # 2%: allows g=9.8 vs rounding in a stated answer


@dataclass(frozen=True)
class Topic:
    id: str
    subject: str        # maths | physics | chemistry | cs
    level: str          # the exam/course framing a student would give
    request: str        # what the student types into the app
    family: str         # problem family (for train/eval separation)
    key: Key | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)


DEV: tuple[Topic, ...] = (
    Topic("dev-quadratic-roots", "maths", "Class 10 CBSE",
          "Solve x^2 - 5x + 6 = 0 by factoring, and show why the roots are where "
          "the graph of y = x^2 - 5x + 6 crosses the x-axis.",
          "quadratic-solve-factor",
          Key("set", sorted(sp.solve(x**2 - 5*x + 6, x)))),
    Topic("dev-product-rule", "maths", "Class 12 CBSE / JEE Main",
          "Differentiate f(x) = x^2 sin x using the product rule, and explain why "
          "the product rule has two terms.",
          "derivative-product-rule",
          Key("expr", sp.diff(x**2 * sp.sin(x), x))),
    Topic("dev-integration-by-parts", "maths", "Class 12 CBSE / JEE Main",
          "Evaluate the integral of x e^x dx using integration by parts.",
          "integral-by-parts",
          Key("antiderivative", sp.integrate(x * sp.exp(x), x))),
    Topic("dev-projectile-range", "physics", "JEE Main",
          "A ball is thrown at 20 m/s at 30 degrees above the horizontal. Taking "
          "g = 10 m/s^2 and ignoring air resistance, find its horizontal range.",
          "projectile-range",
          Key("number", float(sp.N(20**2 * sp.sin(2 * sp.pi / 6) / 10)), "m")),
    Topic("dev-incline-friction", "physics", "JEE Main",
          "A 2 kg block slides down a rough 30 degree incline. The coefficient of "
          "kinetic friction is 0.2 and g = 10 m/s^2. Find its acceleration.",
          "incline-friction-acceleration",
          Key("number", float(sp.N(10 * (sp.sin(sp.pi / 6)
                                         - sp.Rational(1, 5) * sp.cos(sp.pi / 6)))),
              "m/s^2")),
    Topic("dev-series-resistors", "physics", "Class 10 CBSE",
          "Resistors of 4 ohms and 6 ohms are connected in series across a 12 V "
          "battery. Find the current, and the voltage across each resistor.",
          "series-circuit",
          Key("number", float(sp.Rational(12, 10)), "A")),
    Topic("dev-moles-water", "chemistry", "Class 11 CBSE",
          "How many moles of water, and how many molecules, are in 36 g of water? "
          "(H = 1, O = 16, Avogadro's number = 6.022 x 10^23)",
          "mole-concept-mass-to-moles",
          Key("number", float(sp.Rational(36, 18)), "mol")),
    Topic("dev-binary-search", "cs", "First-year CS",
          "Explain binary search, and trace it finding 23 in the sorted list "
          "[2, 5, 8, 12, 16, 23, 38, 56, 72, 91]. At which index (counting from 0) "
          "is 23 found?",
          "binary-search-trace",
          Key("number", 5.0)),
    Topic("dev-nested-loop-big-o", "cs", "First-year CS",
          "Why does a loop nested inside another loop over the same n items take "
          "O(n^2) steps? Count the inner-loop steps for n = 4.",
          "big-o-nested-loops",
          Key("number", 16.0)),
    Topic("dev-two-dice-seven", "maths", "Class 11 CBSE",
          "Two fair dice are rolled. What is the probability that the sum is 7?",
          "probability-two-dice-sum",
          Key("number", float(sp.Rational(6, 36)))),
)

HELDOUT: tuple[Topic, ...] = (
    Topic("ho-parabola-vertex", "maths", "Class 11 CBSE",
          "Find the vertex of the parabola y = 2x^2 - 8x + 3 by completing the "
          "square. What is the minimum value of y?",
          "parabola-vertex-complete-square",
          Key("number", float(sp.minimum(2*x**2 - 8*x + 3, x)))),
    Topic("ho-pendulum-period", "physics", "Class 11 CBSE",
          "Find the period of a simple pendulum of length 1 m, taking g = 9.8 m/s^2, "
          "and explain why the mass of the bob does not matter.",
          "simple-pendulum-period",
          Key("number", float(sp.N(2 * sp.pi * sp.sqrt(1 / sp.Float(9.8)))), "s")),
    Topic("ho-arithmetic-sum", "maths", "Class 10 CBSE",
          "Find the sum of the first 50 positive integers, and explain Gauss's "
          "pairing trick.",
          "arithmetic-series-sum",
          Key("number", float(sp.summation(sp.Symbol("k"), (sp.Symbol("k"), 1, 50))))),
    Topic("ho-ideal-gas-volume", "chemistry", "Class 11 CBSE",
          "What volume do 2 moles of an ideal gas occupy at 300 K and 1 atm? "
          "Use R = 0.0821 L atm / (mol K).",
          "ideal-gas-volume",
          Key("number", float(2 * sp.Float("0.0821") * 300 / 1), "L")),
    Topic("ho-stack-trace", "cs", "First-year CS",
          "Start with an empty stack. Push 3, push 5, pop, push 7, push 2, pop. "
          "What value is on top of the stack, and why is a stack 'last in, first out'?",
          "stack-operations-trace",
          Key("number", 7.0)),
    Topic("ho-free-fall-time", "physics", "Class 9 CBSE / JEE foundation",
          "A stone is dropped from rest from a height of 45 m. Taking g = 10 m/s^2, "
          "how long does it take to reach the ground?",
          "free-fall-time",
          Key("number", float(sp.sqrt(sp.Rational(2 * 45, 10))), "s")),
)

FAMILIES: frozenset[str] = frozenset(t.family for t in DEV + HELDOUT)
BY_ID: dict[str, Topic] = {t.id: t for t in DEV + HELDOUT}

# Run 3 (2026-11-26): the three Calculus 1 lessons that the hand-written
# reference videos (examples/calculus/ref_*.json) also teach, for the
# side-by-side with the untuned model under `anim=1`. Not evaluation topics
# and not in FAMILIES: definite-integral-area and limit-removable are near
# twins of the TRAIN families definite-integral-polynomial and
# limit-sinx-over-x, so never score a tuned model on these.
RUN3: tuple[Topic, ...] = (
    Topic("r3-derivative", "maths", "Calculus 1 (first-year university)",
          "Explain what the derivative of f(x) = x^2/4 + 1 at x = 1 means, as the slope of the "
          "tangent line, and find its value.",
          "derivative-at-point",
          Key("number", 0.5)),
    Topic("r3-limit", "maths", "Calculus 1 (first-year university)",
          "Explain what the limit of (x^2 - 1)/(x - 1) as x approaches 1 means, even though the "
          "function is undefined at x = 1, and find the limit.",
          "limit-removable",
          Key("number", 2.0)),
    Topic("r3-integral", "maths", "Calculus 1 (first-year university)",
          "Explain the definite integral of f(x) = x^2/4 + 1 from x = 0 to x = 3 as the area under "
          "the curve, using rectangle sums, and find its value.",
          "definite-integral-area",
          Key("number", 5.25)),
)
_RUN3_BY_ID: dict[str, Topic] = {t.id: t for t in RUN3}


def topic_set(name: str) -> tuple[Topic, ...]:
    if name == "dev":
        return DEV
    if name == "heldout":
        return HELDOUT
    if name == "run3":
        return RUN3
    if name in _RUN3_BY_ID:
        return (_RUN3_BY_ID[name],)
    if name == "train" or name.startswith("train:"):
        from research.train_topics import TRAIN   # inputs for the teacher only
        if name == "train":
            return TRAIN
        a, _, b = name[6:].partition(":")
        return TRAIN[int(a or 0):int(b) if b else None]
    if name in BY_ID:
        return (BY_ID[name],)
    ids = [s for s in name.split(",") if s]
    if ids and all(i in BY_ID for i in ids):
        return tuple(BY_ID[i] for i in ids)
    raise KeyError(f"unknown topic set {name!r}")


if __name__ == "__main__":
    for t in DEV + HELDOUT:
        print(f"{t.id:28s} {t.family:34s} key={t.key.kind if t.key else None}:"
              f"{t.key.value if t.key else ''}")
