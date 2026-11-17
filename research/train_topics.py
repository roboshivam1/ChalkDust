"""The training topic pool: inputs for the teacher model, never evaluation.

Built from problem-family templates with varied numbers. Every family here is
outside research/topics.FAMILIES, and near-twins of an evaluation family are
left out on purpose (no projectiles, inclines, pendulums, free fall, gas
laws, series/parallel resistor networks, mass->moles, binary search, stacks,
arithmetic-series sums, quadratic solving, product or quotient rule). The
module asserts the separation at import.

Keys are computed here (python/sympy), so teacher outputs can be filtered on
answer correctness as well as validity.

Authorship note: the family templates were written by Claude; they are model
INPUTS. Training targets are teacher-model outputs only.
"""

from __future__ import annotations

import math
import random

import sympy as sp

from research.topics import FAMILIES, Key, Topic

x = sp.Symbol("x")
EXCLUDED_NEAR_TWINS = {"projectile", "incline", "pendulum", "free-fall", "gas", "series", "parallel",
                       "mass-to-moles", "binary-search", "stack", "arithmetic-series", "quadratic",
                       "product-rule", "quotient-rule"}


def _fams(rng: random.Random):
    """Yield (family, subject, level, request, key) for one instance of each family."""
    r = rng.randint
    # ---------------- maths ----------------
    a, b, c = r(2, 9), r(1, 20), r(21, 60)
    yield ("linear-equation-solve", "maths", "Class 8 CBSE",
           f"Solve {a}x + {b} = {c} and explain each step as keeping a balance level.",
           Key("number", (c - b) / a))
    p, q = r(1, 5), r(1, 5)
    s1, s2 = r(1, 6), r(1, 6)
    yield ("simultaneous-linear-2x2", "maths", "Class 10 CBSE",
           f"Solve the simultaneous equations x + y = {p + q} and x - y = {p - q}. What is x?",
           Key("number", float(p)))
    k = r(2, 5)
    yield ("derivative-chain-rule", "maths", "Class 12 CBSE / JEE Main",
           f"Differentiate sin({k}x^2) using the chain rule, and explain the 'outside then inside' idea.",
           Key("expr", sp.diff(sp.sin(k * x**2), x)))
    lo, hi, m = 0, r(1, 4), r(1, 6)
    yield ("definite-integral-polynomial", "maths", "Class 12 CBSE",
           f"Evaluate the definite integral of {m}x^2 from x = {lo} to x = {hi}, and explain it as an area.",
           Key("number", float(sp.integrate(m * x**2, (x, lo, hi)))))
    k2 = r(2, 9)
    yield ("limit-sinx-over-x", "maths", "Class 11 CBSE",
           f"Find the limit of sin({k2}x)/x as x approaches 0.",
           Key("number", float(k2)))
    base, n = rng.choice([2, 3, 5, 10]), r(2, 4)
    yield ("logarithm-solve", "maths", "Class 11 CBSE",
           f"Solve log base {base} of x = {n}. What does a logarithm ask?",
           Key("number", float(base**n)))
    P, rate, yrs = rng.choice([1000, 2000, 5000]), rng.choice([5, 8, 10]), r(2, 4)
    yield ("compound-interest", "maths", "Class 8 CBSE",
           f"Rs {P} is invested at {rate}% per year compound interest for {yrs} years. What is the amount?",
           Key("number", P * (1 + rate / 100) ** yrs, "Rs"))
    # (geometric-series sum left out: a near-twin of the held-out arithmetic-series topic)
    l1, l2 = rng.choice([(3, 4), (5, 12), (8, 15), (6, 8), (9, 12)])
    yield ("pythagoras-hypotenuse", "maths", "Class 7 CBSE",
           f"A right triangle has legs {l1} cm and {l2} cm. Find the hypotenuse and explain why Pythagoras' theorem holds.",
           Key("number", math.hypot(l1, l2), "cm"))
    sa, sb, ang = r(4, 10), r(4, 10), rng.choice([30, 90, 150])
    yield ("triangle-area-sine", "maths", "Class 11 CBSE",
           f"Find the area of a triangle with sides {sa} cm and {sb} cm and the angle {ang} degrees between them.",
           Key("number", 0.5 * sa * sb * math.sin(math.radians(ang)), "cm^2"))
    red, blue = r(2, 5), r(2, 5)
    yield ("probability-without-replacement", "maths", "Class 12 CBSE",
           f"A bag has {red} red and {blue} blue balls. Two are drawn without replacement. What is the probability both are red?",
           Key("number", (red / (red + blue)) * ((red - 1) / (red + blue - 1))))
    nn, kk = r(5, 9), r(2, 3)
    yield ("combinations-ncr", "maths", "Class 11 CBSE",
           f"In how many ways can a team of {kk} be chosen from {nn} students? Explain why order does not matter.",
           Key("number", float(math.comb(nn, kk))))
    m11, m12, m21, m22 = r(1, 6), r(1, 6), r(1, 6), r(1, 6)
    yield ("matrix-determinant-2x2", "maths", "Class 12 CBSE",
           f"Find the determinant of the matrix [[{m11}, {m12}], [{m21}, {m22}]] and explain what it says about area.",
           Key("number", float(m11 * m22 - m12 * m21)))
    x1, y1, x2, y2 = r(-3, 3), r(-3, 3), r(4, 9), r(4, 9)
    yield ("distance-two-points", "maths", "Class 10 CBSE",
           f"Find the distance between the points ({x1}, {y1}) and ({x2}, {y2}).",
           Key("number", math.hypot(x2 - x1, y2 - y1)))
    yield ("slope-two-points", "maths", "Class 9 CBSE",
           f"Find the slope of the line through ({x1}, {y1}) and ({x2}, {y2}), and explain slope as rise over run.",
           Key("number", (y2 - y1) / (x2 - x1)))
    data = sorted(r(1, 20) for _ in range(5))
    yield ("mean-median", "maths", "Class 9 CBSE",
           f"Find the mean of the data {data}, and explain how it differs from the median.",
           Key("number", sum(data) / 5))
    yield ("trig-identity-explain", "maths", "Class 10 CBSE",
           "Explain why sin^2 x + cos^2 x = 1 using the unit circle.", None)
    # ---------------- physics ----------------
    V, R = rng.choice([6, 12, 24]), rng.choice([2, 3, 4, 6])
    yield ("electric-power", "physics", "Class 10 CBSE",
           f"A {R} ohm heater is connected to a {V} V supply. What power does it use?",
           Key("number", V**2 / R, "W"))
    mass, v = rng.choice([800, 1000, 1200]), rng.choice([10, 20, 25])
    yield ("kinetic-energy", "physics", "Class 9 CBSE",
           f"Find the kinetic energy of a {mass} kg car moving at {v} m/s. Why does doubling speed quadruple it?",
           Key("number", 0.5 * mass * v**2, "J"))
    m1, u1, m2 = r(1, 5), r(2, 8), r(1, 5)
    yield ("momentum-inelastic-collision", "physics", "Class 11 CBSE",
           f"A {m1} kg trolley moving at {u1} m/s sticks to a stationary {m2} kg trolley. What is their common speed?",
           Key("number", m1 * u1 / (m1 + m2), "m/s"))
    kspring, F = rng.choice([100, 200, 400]), rng.choice([10, 20, 40])
    yield ("hookes-law", "physics", "Class 11 CBSE",
           f"A spring of stiffness {kspring} N/m is pulled with {F} N. How far does it stretch?",
           Key("number", F / kspring, "m"))
    Fp, area = rng.choice([100, 500, 1000]), rng.choice([0.5, 2, 4])
    yield ("pressure-force-area", "physics", "Class 8 CBSE",
           f"A force of {Fp} N acts on an area of {area} m^2. What is the pressure? Why do knives cut better when sharp?",
           Key("number", Fp / area, "Pa"))
    mc, vc, rc = r(1, 4), r(2, 6), rng.choice([0.5, 1, 2])
    yield ("centripetal-force", "physics", "Class 11 CBSE",
           f"A {mc} kg ball on a string moves in a circle of radius {rc} m at {vc} m/s. What tension provides the centripetal force?",
           Key("number", mc * vc**2 / rc, "N"))
    fq, lam = rng.choice([256, 440, 512]), rng.choice([0.5, 0.75, 1.3])
    yield ("wave-speed", "physics", "Class 9 CBSE",
           f"A sound wave has frequency {fq} Hz and wavelength {lam} m. What is its speed?",
           Key("number", fq * lam, "m/s"))
    n2, inc = rng.choice([1.33, 1.5]), rng.choice([30, 45, 60])
    yield ("snells-law", "physics", "Class 10 CBSE",
           f"Light enters glass of refractive index {n2} from air at {inc} degrees to the normal. Find the angle of refraction.",
           Key("number", math.degrees(math.asin(math.sin(math.radians(inc)) / n2)), "degrees"))
    mh, cw, dT = rng.choice([0.5, 1, 2]), 4200, r(10, 40)
    yield ("specific-heat", "physics", "Class 11 CBSE",
           f"How much heat raises {mh} kg of water by {dT} degrees C? (c = {cw} J/kg K)",
           Key("number", mh * cw * dT, "J"))
    q1, q2, dist = r(1, 5), r(1, 5), rng.choice([0.1, 0.2, 0.5])
    yield ("coulombs-law", "physics", "Class 12 CBSE",
           f"Two charges of {q1} microcoulomb and {q2} microcoulomb are {dist} m apart. Find the force between them (k = 9 x 10^9).",
           Key("number", 9e9 * q1 * 1e-6 * q2 * 1e-6 / dist**2, "N"))
    u0, acc, tt = r(0, 10), r(1, 4), r(3, 8)
    yield ("suvat-final-velocity", "physics", "Class 9 CBSE",
           f"A car starts at {u0} m/s and accelerates at {acc} m/s^2 for {tt} s on a straight road. What is its final speed?",
           Key("number", u0 + acc * tt, "m/s"))
    mw, wt = rng.choice([50, 60, 70]), rng.choice([10, 20, 30])
    yield ("power-work-time", "physics", "Class 9 CBSE",
           f"A {mw} kg student climbs 3 m of stairs in {wt} s. What power do they develop? (g = 10)",
           Key("number", mw * 10 * 3 / wt, "W"))
    yield ("newtons-third-law-explain", "physics", "Class 9 CBSE",
           "Explain Newton's third law with a person pushing a wall, and why the forces do not cancel.", None)
    # ---------------- chemistry ----------------
    M1, V1, V2 = rng.choice([1, 2, 6]), rng.choice([50, 100]), rng.choice([250, 500])
    yield ("dilution-m1v1", "chemistry", "Class 11 CBSE",
           f"{V1} mL of {M1} M acid is diluted to {V2} mL. What is the new concentration?",
           Key("number", M1 * V1 / V2, "M"))
    expo = r(1, 4)
    yield ("ph-strong-acid", "chemistry", "Class 11 CBSE",
           f"What is the pH of a 10^-{expo} M solution of hydrochloric acid? Why is the scale logarithmic?",
           Key("number", float(expo)))
    yield ("percent-composition", "chemistry", "Class 11 CBSE",
           "What percentage by mass of water (H2O) is hydrogen? (H = 1, O = 16)",
           Key("number", 2 / 18 * 100, "%"))
    hl, tot = rng.choice([5, 8, 10]), rng.choice([2, 3, 4])
    yield ("half-life", "chemistry", "Class 12 CBSE",
           f"A sample has a half-life of {hl} days. What fraction remains after {hl * tot} days?",
           Key("number", 0.5**tot))
    yield ("covalent-vs-ionic-explain", "chemistry", "Class 10 CBSE",
           "Explain the difference between ionic and covalent bonding with sodium chloride and water.", None)
    # ---------------- cs ----------------
    arr = rng.sample(range(1, 40), 6)
    tgt = rng.choice(arr)
    yield ("linear-search-comparisons", "cs", "First-year CS",
           f"Trace linear search for {tgt} in the list {arr}. How many comparisons does it make?",
           Key("number", float(arr.index(tgt) + 1)))
    q_ops = [r(1, 9) for _ in range(3)]
    yield ("queue-trace", "cs", "First-year CS",
           f"Start with an empty queue. Enqueue {q_ops[0]}, enqueue {q_ops[1]}, dequeue, enqueue {q_ops[2]}. What is at the front, and why is a queue first in, first out?",
           Key("number", float(q_ops[1])))
    bits = "".join(rng.choice("01") for _ in range(5))
    bits = "1" + bits
    yield ("binary-to-decimal", "cs", "First-year CS",
           f"Convert the binary number {bits} to decimal, explaining place values.",
           Key("number", float(int(bits, 2))))
    nf = r(4, 6)
    yield ("recursion-factorial", "cs", "First-year CS",
           f"Trace the recursive factorial function for n = {nf}. What does it return?",
           Key("number", float(math.factorial(nf))))
    nfib = r(6, 9)
    yield ("fibonacci-iterative", "cs", "First-year CS",
           f"Trace an iterative Fibonacci function (F(0) = 0, F(1) = 1) up to F({nfib}). What is F({nfib})?",
           Key("number", float([0, 1, 1, 2, 3, 5, 8, 13, 21, 34][nfib])))
    key_, msz = r(20, 99), rng.choice([7, 10, 11])
    yield ("hash-mod-index", "cs", "First-year CS",
           f"A hash table of size {msz} uses h(k) = k mod {msz}. Which bucket does key {key_} go to?",
           Key("number", float(key_ % msz)))
    nl = rng.choice([10, 16, 20])
    yield ("single-loop-step-count", "cs", "First-year CS",
           f"How many times does 'for i in range(0, {nl}, 2)' run its body? Explain why this is still O(n).",
           Key("number", float(len(range(0, nl, 2)))))
    yield ("bubble-sort-explain", "cs", "First-year CS",
           f"Explain one pass of bubble sort on {rng.sample(range(1, 20), 5)}: why does the largest value end up last?",
           None)
    yield ("bfs-vs-dfs-explain", "cs", "First-year CS",
           "Explain the difference between breadth-first and depth-first search on a small tree.", None)


def pool(n_per_family: int = 3, seed: int = 20261117) -> tuple[Topic, ...]:
    rng = random.Random(seed)
    out = []
    for i in range(n_per_family):
        for fam, subj, level, req, key in _fams(rng):
            out.append(Topic(f"tr-{fam}-{i}", subj, level, req, fam, key))
    return tuple(out)


TRAIN = pool()
_fam_set = {t.family for t in TRAIN}
assert not (_fam_set & FAMILIES), f"training pool overlaps eval families: {_fam_set & FAMILIES}"
assert not any(w in f for f in _fam_set for w in EXCLUDED_NEAR_TWINS), "near-twin family in training pool"

if __name__ == "__main__":
    print(len(TRAIN), "topics,", len(_fam_set), "families,",
          sum(1 for t in TRAIN if t.key), "with keys")
    for t in TRAIN[:45]:
        print(f"{t.id:42s} {t.key.value if t.key and t.key.kind == 'number' else (t.key.kind if t.key else None)}")
