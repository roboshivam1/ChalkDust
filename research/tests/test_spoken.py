"""The spoken-answer matcher: finds the key in narration however it is written."""

import pytest

from research.content import check_spoken
from research.topics import BY_ID

CASES = [
    ("dev-projectile-range", r"So, the horizontal range of the ball is $20\sqrt{3}$ meters.", "correct"),
    ("dev-projectile-range", "The range is about thirty four point six metres.", "correct"),
    ("dev-projectile-range", "The range is about 34.6 m.", "correct"),
    ("dev-projectile-range", "The range is 30 metres.", "not_found"),
    ("dev-two-dice-seven", r"So the probability is \frac{1}{6}.", "correct"),
    ("dev-two-dice-seven", "The probability is one sixth, or one in six.", "correct"),
    ("dev-two-dice-seven", "favorable = 6, total = 36, so 6/36", "correct"),
    ("dev-quadratic-roots", "the roots are where it crosses the x-axis, which happens when $y = 0$.", "not_found"),
    ("dev-quadratic-roots", "So x equals two or x equals three.", "correct"),
    ("dev-nested-loop-big-o", "For n equals four, the inner loop runs sixteen times.", "correct"),
    ("dev-moles-water", "36 grams of water is 2 moles.", "correct"),
    ("ho-arithmetic-sum", "The sum is one thousand two hundred and seventy five.", "correct"),
    ("ho-arithmetic-sum", "The sum is 1,275.", "correct"),
    ("ho-parabola-vertex", "The minimum value of y is minus five.", "correct"),
    ("dev-product-rule", "So the derivative is 2x sin(x) + x^2 cos(x).", "n/a"),
]


@pytest.mark.parametrize("tid,text,want", CASES)
def test_spoken(tid, text, want):
    assert check_spoken(BY_ID[tid].key, text)["status"] == want
