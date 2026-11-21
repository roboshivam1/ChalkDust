"""The spoken-answer check: the narration's final stated value, not any number
it happens to contain. Cases marked (review) are the false positives a fresh
review found in the presence-check version (2026-11-17)."""

import pytest

from research.content import check_spoken, spoken_numbers
from research.topics import BY_ID

CASES = [
    ("dev-projectile-range", r"So, the horizontal range of the ball is $20\sqrt{3}$ meters.", "correct"),
    ("dev-projectile-range", "The range is about thirty four point six metres.", "correct"),
    ("dev-projectile-range", "The range is about 34.6 m.", "correct"),
    ("dev-projectile-range", "The range is twenty root three metres.", "correct"),
    ("dev-projectile-range", "The range is 30 metres.", "wrong"),
    ("dev-projectile-range", "Thrown at 20 metres per second and 30 degrees, it goes far.", "not_found"),
    ("dev-projectile-range", "The range is 34.6 m. Next, try 45 degrees and compare.", "correct"),
    ("dev-two-dice-seven", r"So the probability is \frac{1}{6}.", "correct"),
    ("dev-two-dice-seven", "The probability is one sixth, or one in six.", "correct"),
    ("dev-two-dice-seven", "favorable = 6, total = 36, so 6/36", "correct"),
    ("dev-quadratic-roots", "the roots are where it crosses the x-axis, which happens when $y = 0$.", "not_found"),
    ("dev-quadratic-roots", "So x equals two or x equals three.", "correct"),
    # (mutation test, 2026-11-21) the closing statement must be exactly the roots
    ("dev-quadratic-roots", "The roots are 2 and 3. So the answer is 7.", "wrong"),
    ("dev-quadratic-roots", "So the roots are 2, 3 and 7.", "wrong"),
    ("dev-quadratic-roots", "The roots are 2 and 3. Both satisfy the equation.", "correct"),
    ("dev-quadratic-roots", "So x equals three.", "not_found"),
    ("dev-nested-loop-big-o", "For n equals four, the inner loop runs sixteen times.", "correct"),
    ("dev-moles-water", "36 grams of water is 2 moles.", "correct"),
    ("ho-arithmetic-sum", "The sum is one thousand two hundred and seventy five.", "correct"),
    ("ho-arithmetic-sum", "The sum is 1,275.", "correct"),
    ("ho-parabola-vertex", "The minimum value of y is minus five.", "correct"),
    ("dev-product-rule", "So the derivative is 2x sin(x) + x^2 cos(x).", "n/a"),
    # (review) a rejected value or a formula's constant is not the stated answer
    ("dev-incline-friction", "The acceleration is 3.27 m/s squared. That is too high; it should be 2.0.", "wrong"),
    ("ho-pendulum-period", "T equals two pi root L over g, about two point eight seconds.", "wrong"),
    ("ho-pendulum-period", "T equals two pi root L over g, which comes to about 2.0 seconds.", "correct"),
    # (review) the key is in the request: speech cannot show it was worked out
    ("dev-binary-search", "So 23 is found at index 4 of the list 2, 5, 8.", "n/a"),
    ("ho-stack-trace", "We pushed 7 and then 2, so the top is 2.", "n/a"),
]


@pytest.mark.parametrize("tid,text,want", CASES)
def test_spoken(tid, text, want):
    t = BY_ID[tid]
    assert check_spoken(t.key, text, t.request)["status"] == want


def test_and_joins_only_after_hundred():
    assert spoken_numbers("five and two") == [5.0, 2.0]
    assert spoken_numbers("one thousand two hundred and seventy five") == [1275.0]


def test_fraction_read_once_and_exponents_ignored():
    assert spoken_numbers("so 6/36 of the outcomes") == [6 / 36]
    assert spoken_numbers("g is 9.8 m/s^2") == [9.8]
