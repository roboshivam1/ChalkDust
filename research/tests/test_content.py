"""Pins the answer checker and the safe maths parser: the findings' accuracy
numbers rest on these. Run: .lab/venv-harness/Scripts/python -m pytest research/tests -q
"""

import pytest

from research.content import check_answer, derivation_consistency
from research.generate import safe_calc
from research.score import quality_flags, repair_escapes
from research.topics import BY_ID

CASES = [
    ("dev-product-rule", "2*x*sin(x) + x**2*cos(x)", "correct"),
    ("dev-product-rule", r"f'(x) = 2x\sin x + x^2\cos x", "correct"),
    ("dev-product-rule", r"f'(x) = 2x\cos x", "wrong"),
    ("dev-two-dice-seven", "1/6", "correct"),
    ("dev-two-dice-seven", r"\frac{1}{6}", "correct"),
    ("dev-two-dice-seven", r"\frac{1}{12}", "wrong"),
    ("dev-projectile-range", "34.6 m", "correct"),
    ("dev-projectile-range", r"R = 20\sqrt{3}", "correct"),
    ("dev-projectile-range", "30 m", "wrong"),
    ("dev-quadratic-roots", "x = 2 or x = 3", "correct"),
    ("dev-quadratic-roots", r"x = 2 \text{ or } x = 3", "correct"),
    ("dev-quadratic-roots", "x = 2", "wrong"),
    ("dev-integration-by-parts", "e^x(x-1) + C", "correct"),
    ("dev-integration-by-parts", "x*exp(x) + exp(x)", "wrong"),
    ("dev-moles-water", "2 mol", "correct"),
    ("ho-free-fall-time", "t = 3 s", "correct"),
    ("ho-ideal-gas-volume", r"V \approx 49.3\,\mathrm{L}", "correct"),
    ("dev-binary-search", "index 5", "correct"),
    ("dev-binary-search", "nothing to see", "unparsed"),
]


@pytest.mark.parametrize("tid,text,want", CASES)
def test_check_answer(tid, text, want):
    assert check_answer(BY_ID[tid].key, [text])["status"] == want


def test_step_consistency_counts_invented_lines():
    beats = [{"component": "EquationDerivation",
              "params": {"steps": ["{{ x^2 - 5x }} + 6 = 0", "(x-2)(x-3) = 0", "x = 2", "x = 7"]}}]
    r = derivation_consistency(beats, BY_ID["dev-quadratic-roots"].key)
    assert (r["consistent"], r["inconsistent"]) == (3, 1)


@pytest.mark.parametrize("expr", ["__import__('os').system('echo hi')", "(1).__class__",
                                  "9**9**9", "open('x')", "lambda: 1", "[1][0]"])
def test_safe_calc_refuses_code(expr):
    assert safe_calc(expr) is None


def test_safe_calc_degrees():
    assert abs(safe_calc("20**2*sin(2*30*deg)/10") - 34.641016) < 1e-5


def test_quality_flags():
    assert quality_flags({"component": "ProblemStatement", "narration": "Given $v = 20$",
                          "params": {"given": [r"v = 20\,\mathrm{m/s}"], "text": "t", "find": "f"}}) \
        == ["narration_latex", "prose_latex:given"]
    assert quality_flags({"component": "ProblemStatement", "narration": "plain words here",
                          "params": {"given": [r"$v = 20\,\mathrm{m/s}$"], "text": "t", "find": "f"}}) == []
    assert quality_flags({"component": "BulletReveal", "narration": "ok",
                          "params": {"items": ["use $x^2$"]}}) == ["stray_dollar:items"]


def test_escape_repair_restores_latex_commands():
    fixed, n = repair_escapes({"steps": ["\theta = 30", "\frac{1}{2}"], "source": "a\tb"})
    assert fixed["steps"] == ["\\theta = 30", "\\frac{1}{2}"] and n == 2
    assert fixed["source"] == "a\tb"   # code keeps its real tab
