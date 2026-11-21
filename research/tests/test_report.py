"""The paired statistics behind the tuned-against-untuned claim (research/report.py)."""
from research.report import boot_ci, sign_p


def test_sign_test_is_exact_two_sided_and_drops_ties():
    assert sign_p([1, 1, 1, 1, 1]) == 2 / 32          # 5 of 5 positive
    assert sign_p([0.5, -0.5]) == 1.0
    assert sign_p([0.2, 0, 0, 0.1]) == 2 / 4           # ties dropped: 2 of 2 positive
    assert sign_p([0, 0]) is None                      # all ties: no test


def test_bootstrap_ci_is_seeded_and_brackets_the_mean():
    d = [0.1, 0.3, -0.1, 0.2, 0.0, 0.4]
    lo, hi = boot_ci(d)
    assert (lo, hi) == boot_ci(d)                      # fixed RNG seed: reproducible
    assert lo <= sum(d) / len(d) <= hi
    assert boot_ci([0.25] * 5) == (0.25, 0.25)
