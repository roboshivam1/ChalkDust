"""Answer checks on training-topic shapes (2026-11-21 false-pass sweep over TRAIN).

These keys feed build_sft's gate, so a false pass puts a wrong video into the
fine-tuning set. Each case is one the sweep found, run through the real checkers."""

import pytest

from research.content import check_answer, check_spoken, final_value
from research.topics import Key

WAVE = Key("number", 572.0, "m/s")
MISCONCEPTION_WAVE = ("So the speed of the wave is 338.5 metres per second. A common mistake is to "
                      "multiply instead, which would give 572 metres per second.")


@pytest.mark.parametrize("line,value", [
    ("v = 338.5 m/s", 338.5),                       # compound unit: did not parse
    ("a = 9.8 m/s^2", 9.8),
    ("v = 20 m s^-1", 20.0),
    (r"A = 9\,\text{cm}^2", 9.0),                   # read as 81
    ("M_2 = 0.4 M", 0.4),                           # molar
    (r"\(P = 192\,\mathrm{W}\)", 192.0),            # inline-maths delimiters
    ("A = Rs 2519.42", 2519.42),
])
def test_final_value_reads_units(line, value):
    assert final_value(line) == pytest.approx(value)


def test_unparsed_shown_unit_no_longer_hands_the_answer_to_the_misconception():
    assert check_answer(WAVE, ["v = 338.5 m/s"], MISCONCEPTION_WAVE)["status"] == "wrong"


def test_spoken_misconception_value_is_not_the_result():
    assert check_spoken(WAVE, MISCONCEPTION_WAVE)["status"] == "wrong"
    ke = Key("number", 60000.0, "J")
    t = ("So the kinetic energy is 60,000 joules. A common mistake is to forget the half, "
         "which would give 120,000 joules.")
    assert check_spoken(ke, t)["status"] == "correct"


@pytest.mark.parametrize("line,key", [
    (r"v = 20\,\mathrm{m/s},\ KE = 60,000\,\mathrm{J}", Key("number", 60000.0, "J")),
    (r"\Delta T = 20\,\mathrm{K},\ Q = 50,400\,\mathrm{J}", Key("number", 50400.0, "J")),
])
def test_thousands_comma_is_not_a_separator(line, key):
    assert check_answer(key, [line])["status"] == "correct"


def test_thousands_comma_value_is_not_a_smaller_key():
    assert check_answer(Key("number", 200.0, "W"), [r"v = 2\,\mathrm{m/s},\ P = 1,200\,\mathrm{W}"])["status"] == "wrong"


@pytest.mark.parametrize("key,shown,status", [
    (Key("number", 57.0), "n = 58", "wrong"),                   # a count is exact
    (Key("number", 120.0), "122", "wrong"),
    (Key("number", 120.0), "120", "correct"),
    (Key("number", 2519.42, "Rs"), "A = Rs 2480", "wrong"),     # simple, not compound, interest
    (Key("number", 2519.42, "Rs"), "A = Rs 2519", "correct"),
    (Key("number", 34.64, "m"), "R = 34.6 m", "correct"),       # rounding still allowed
])
def test_tolerance_by_key_shape(key, shown, status):
    assert check_answer(key, [shown])["status"] == status


def test_joules_per_second_is_a_power():
    t = "The work is 576 joules, and dividing by 3 seconds gives 192 joules per second."
    assert check_spoken(Key("number", 192.0, "W"), t)["status"] == "correct"


def test_unicode_minus_is_a_sign():
    assert check_spoken(Key("number", 10.0), "So the determinant is −10.")["status"] == "wrong"
    assert check_spoken(Key("number", -21.0), "So the determinant is −21.")["status"] == "correct"


@pytest.mark.parametrize("text,status", [
    ("So the roots are 2, 3 and 6.", "wrong"),                  # 6 is from the request, claimed as a root
    ("So the roots are two, three and six.", "wrong"),
    ("So, for x squared minus five x plus six, the roots are 2 and 3.", "correct"),
    ("The roots are 2 and 3.", "correct"),
])
def test_request_number_listed_with_the_roots_is_a_root(text, status):
    key = Key("set", [2.0, 3.0])
    assert check_spoken(key, text, "Find the roots of x^2 - 5x + 6 = 0.")["status"] == status


def test_large_integer_valued_key_keeps_its_tolerance():
    t = "So, 36 g of water contains 2 moles and 1.2044 × 10^{24} molecules."
    assert check_spoken(Key("number", 1.2044e24, "molecules"), t)["status"] == "correct"
