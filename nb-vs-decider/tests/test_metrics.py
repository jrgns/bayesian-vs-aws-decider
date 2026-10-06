import numpy as np
import pytest
from sklearn.metrics import f1_score

from scripts import metrics


def test_accuracy_and_macro_f1():
    y = [0, 0, 1, 1, 2]
    p = [0, 1, 1, 1, 0]
    assert metrics.accuracy(y, p) == pytest.approx(0.6)
    # class 0: tp1 fp1 fn1 -> 0.5; class 1: tp2 fp1 fn0 -> 0.8; class 2: tp0 -> 0
    assert metrics.macro_f1(y, p, 3) == pytest.approx((0.5 + 0.8 + 0.0) / 3)


def test_confusion_and_recall():
    y, p = [0, 0, 1, 1, 2], [0, 1, 1, 1, 0]
    cm = metrics.confusion_matrix(y, p, 3)
    assert cm.tolist() == [[1, 1, 0], [0, 2, 0], [1, 0, 0]]
    assert metrics.per_class_recall(y, p, 3).tolist() == [0.5, 1.0, 0.0]


def test_brier_hand_computed():
    probs = [[0.7, 0.3], [0.4, 0.6]]
    # item 1 (true 0): 0.09 + 0.09 = 0.18; item 2 (true 0): 0.36 + 0.36 = 0.72
    assert metrics.brier(probs, [0, 0]) == pytest.approx((0.18 + 0.72) / 2)
    assert metrics.brier([[1.0, 0.0]], [0]) == 0.0


def test_ece_hand_computed():
    # conf 0.9 (correct), conf 0.9 (wrong) -> one bin: |0.5 - 0.9| = 0.4
    probs = [[0.9, 0.1], [0.9, 0.1]]
    assert metrics.ece(probs, [0, 1]) == pytest.approx(0.4)
    # perfectly confident and right: 0
    assert metrics.ece([[1.0, 0.0], [0.0, 1.0]], [0, 1]) == 0.0
    # two bins: conf 0.55 right (bin 9), conf 0.95 wrong (bin 14): (|1-.55| + |0-.95|) / 2
    probs = [[0.55, 0.45], [0.05, 0.95]]
    assert metrics.ece(probs, [0, 0]) == pytest.approx((0.45 + 0.95) / 2)


def test_top2_margin():
    m = metrics.top2_margin([[0.1, 0.6, 0.3], [0.5, 0.5, 0.0]])
    assert m == pytest.approx([0.3, 0.0])


def test_mcnemar_exact():
    a = [True] * 8 + [False] * 2 + [True] * 2
    b = [False] * 8 + [True] * 2 + [True] * 2
    out = metrics.mcnemar_exact(a, b)
    assert (out["a_only_correct"], out["b_only_correct"]) == (8, 2)
    # two-sided exact: 2 * P(X <= 2 | n=10, p=.5) = 2 * 56/1024
    assert out["p_value"] == pytest.approx(2 * 56 / 1024)
    assert metrics.mcnemar_exact([True], [True])["p_value"] == 1.0


def test_vectorised_f1_matches_sklearn():
    rng = np.random.default_rng(0)
    yt, yp = rng.integers(0, 7, (5, 300)), rng.integers(0, 7, (5, 300))
    fast = metrics._macro_f1_batch(yt, yp, 7)
    ref = [
        f1_score(t, p, labels=range(7), average="macro", zero_division=0)
        for t, p in zip(yt, yp, strict=True)
    ]
    assert fast == pytest.approx(ref)


def test_bootstrap_ci_seeded_and_brackets_estimate():
    rng = np.random.default_rng(1)
    yt = rng.integers(0, 3, 400)
    yp = np.where(rng.random(400) < 0.8, yt, rng.integers(0, 3, 400))
    a = metrics.bootstrap_ci(yt, yp, 3, seed=7, n_boot=1000)
    assert a == metrics.bootstrap_ci(yt, yp, 3, seed=7, n_boot=1000)
    assert a != metrics.bootstrap_ci(yt, yp, 3, seed=8, n_boot=1000)
    lo, hi = a["accuracy"]
    assert lo < metrics.accuracy(yt, yp) < hi
    lo, hi = a["macro_f1"]
    assert lo < metrics.macro_f1(yt, yp, 3) < hi


def test_bootstrap_all_correct_is_degenerate():
    y = np.array([0, 1, 2] * 10)
    assert metrics.bootstrap_ci(y, y, 3, seed=0, n_boot=100)["accuracy"] == (1.0, 1.0)
