import numpy as np
import pytest

from scripts import crossover as cx
from scripts.summary_stats import bootstrap_mean_ci, mean_ci


def test_direction():
    assert cx.direction(10, 3) == "nb_better"
    assert cx.direction(3, 10) == "decider_better"
    assert cx.direction(5, 5) == "tie"


@pytest.mark.parametrize(
    "p, d, expected",
    [
        (0.5, "decider_better", True),  # not significant
        (0.01, "decider_better", False),  # significantly worse
        (0.01, "nb_better", True),  # significantly better
        (0.05, "decider_better", True),  # p == alpha is not significant
        (0.049, "decider_better", False),
        (1.0, "tie", True),
    ],
)
def test_seed_qualifies(p, d, expected):
    assert cx.seed_qualifies(p, d) is expected


def test_n_qualifies_needs_mean_and_strict_majority():
    assert cx.n_qualifies(0.75, 0.72, 3, 5)
    assert not cx.n_qualifies(0.75, 0.72, 2, 5)  # no majority
    assert not cx.n_qualifies(0.70, 0.72, 5, 5)  # mean below Decider
    assert cx.n_qualifies(0.72, 0.72, 3, 5)  # equal mean passes
    assert not cx.n_qualifies(0.8, 0.7, 2, 4)  # 2 of 4 is not a majority


def test_find_crossover_is_smallest_qualifying_n_not_first_in_order():
    rows = [
        {"n": 100, "qualifies": True},
        {"n": 5, "qualifies": False},
        {"n": 50, "qualifies": True},
        {"n": 1000, "qualifies": True},
    ]
    assert cx.find_crossover(rows) == 50
    assert cx.find_crossover([{"n": 1, "qualifies": False}]) is None


def test_compare_counts_discordant_items_and_flags_direction():
    nb = np.array([True] * 30 + [False] * 5 + [True] * 10 + [False] * 5)
    dec = np.array([True] * 30 + [True] * 5 + [False] * 10 + [False] * 5)
    r = cx.compare(nb, dec)
    assert (r["nb_only_correct"], r["decider_only_correct"]) == (10, 5)
    assert r["direction"] == "nb_better"
    assert r["p_value"] == pytest.approx(0.3018, abs=1e-3)
    assert not r["significant"] and r["seed_qualifies"]


def test_correctness_asserts_identical_item_sets():
    import pandas as pd

    a = pd.DataFrame({"label": [0, 1], "pred": [0, 0]}, index=pd.Index([1, 2], name="item_id"))
    b = pd.DataFrame({"label": [0, 1], "pred": [0, 1]}, index=pd.Index([1, 3], name="item_id"))
    with pytest.raises(AssertionError):
        cx.correctness(a, b)
    ok, dec = cx.correctness(a, a.rename(index={2: 2}))
    assert ok.tolist() == dec.tolist() == [True, False]


def test_mean_ci_and_bootstrap_ci_bracket_the_mean():
    m, lo, hi = mean_ci([0.70, 0.72, 0.74, 0.71, 0.73])
    assert lo < m < hi and m == pytest.approx(0.72)
    assert mean_ci([0.5]) == (0.5, 0.5, 0.5)
    vals = np.r_[np.ones(60), np.zeros(40)]
    lo, hi = bootstrap_mean_ci(vals, seed=1, n_boot=1000)
    assert lo < 0.6 < hi
