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
    ("dev-quadratic-roots", "So x equals two. And x equals three.", "correct"),
    # (audit, 2026-11-21) a verification sentence before the answer is not part of it
    ("dev-quadratic-roots", "At x equals two and x equals three, the height of the graph is zero, confirming "
                            "these are our roots. So, we have found that the roots are x equals two and x equals "
                            "three. Whenever y equals zero, you have found your solution.", "correct"),
    # (audit, 2026-11-21) a hyphenated fraction word states a value
    ("dev-two-dice-seven", "The probability of rolling a sum of 7 is exactly one-sixth.", "correct"),
    ("dev-two-dice-seven", "The probability is two-thirds.", "wrong"),
    # (audit, 2026-11-21) several asked quantities: the key's unit picks the answer
    ("dev-series-resistors", "The current is 1.2 amperes. The voltage across the 4 ohm resistor is 4.8 volts, "
                             "and across the 6 ohm resistor is 7.2 volts.", "correct"),
    ("dev-series-resistors", "The current is 1.5 amperes. The voltage across the 6 ohm resistor is 9 volts.", "wrong"),
    ("dev-moles-water", "There are exactly 2 moles of water molecules, which equals 1.2044 × 10²⁴ molecules.",
     "correct"),
    ("dev-moles-water", "So 36 g of water is 3 moles, or 1.8 × 10^{24} molecules.", "wrong"),
    ("dev-moles-water", "molecules = 2 × 6.022 × 10^{23} = 1.2044 × 10^{24}. So, 36 g of water contains 2 moles "
                        "and 1.2044 × 10^{24} molecules.", "correct"),
    ("dev-moles-water", r"So there are 2 moles, which is $1.2044 \times 10^{24}$ molecules.", "correct"),
    ("dev-moles-water", "Thirty-six grams of water contains exactly two moles. These two moles consist of one point "
                        "two zero four four times ten to the twenty-fourth molecules.", "correct"),
    ("dev-incline-friction", "The normal force is only 17.32 N. The block accelerates down the incline at "
                             "3.27 m/s².", "correct"),
    ("dev-incline-friction", "The block accelerates at 3.27 m/s². Wait, it should be 2.0.", "wrong"),
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


def test_scientific_notation_is_one_value():
    assert spoken_numbers("which is 1.2044 × 10²⁴ molecules") == pytest.approx([1.2044e24])
    assert spoken_numbers(r"$6.02 \times 10^{23}$ per mole") == pytest.approx([6.02e23])
