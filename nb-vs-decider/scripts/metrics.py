"""Classification metrics shared by the Decider and Naive Bayes runs.

`probs` is an (n, k) array of class probabilities, `y_true`/`y_pred` are integer labels 0..k-1.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import f1_score

N_BOOT = 10_000
ECE_BINS = 15
_CHUNK = 250  # bootstrap resamples per vectorised block (bounds memory)


def accuracy(y_true, y_pred) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def macro_f1(y_true, y_pred, n_classes: int) -> float:
    """sklearn macro-F1 over all `n_classes` labels (a class with no support scores 0)."""
    return float(
        f1_score(y_true, y_pred, labels=list(range(n_classes)), average="macro", zero_division=0)
    )


def confusion_matrix(y_true, y_pred, n_classes: int) -> np.ndarray:
    """Rows = true class, columns = predicted class."""
    idx = np.asarray(y_true) * n_classes + np.asarray(y_pred)
    return np.bincount(idx, minlength=n_classes**2).reshape(n_classes, n_classes)


def per_class_recall(y_true, y_pred, n_classes: int) -> np.ndarray:
    cm = confusion_matrix(y_true, y_pred, n_classes)
    support = cm.sum(axis=1)
    return np.divide(np.diag(cm), support, out=np.zeros(n_classes), where=support > 0)


def _macro_f1_batch(yt: np.ndarray, yp: np.ndarray, n_classes: int) -> np.ndarray:
    """Macro-F1 per row of (r, n) label matrices; same convention as `macro_f1`."""
    f1 = np.zeros(yt.shape[0])
    for c in range(n_classes):
        t, p = yt == c, yp == c
        tp = (t & p).sum(axis=1)
        denom = t.sum(axis=1) + p.sum(axis=1)
        f1 += np.divide(2 * tp, denom, out=np.zeros(len(tp)), where=denom > 0)
    return f1 / n_classes


def bootstrap_ci(
    y_true, y_pred, n_classes: int, *, seed: int, n_boot: int = N_BOOT, level: float = 0.95
) -> dict[str, tuple[float, float]]:
    """Percentile bootstrap CI for accuracy and macro-F1. Both use the same resamples."""
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    acc, f1 = [], []
    for start in range(0, n_boot, _CHUNK):
        idx = rng.integers(0, len(yt), size=(min(_CHUNK, n_boot - start), len(yt)))
        bt, bp = yt[idx], yp[idx]
        acc.append((bt == bp).mean(axis=1))
        f1.append(_macro_f1_batch(bt, bp, n_classes))
    lo, hi = 100 * (1 - level) / 2, 100 * (1 + level) / 2
    return {
        "accuracy": tuple(float(x) for x in np.percentile(np.concatenate(acc), [lo, hi])),
        "macro_f1": tuple(float(x) for x in np.percentile(np.concatenate(f1), [lo, hi])),
    }


def brier(probs, y_true) -> float:
    """Multiclass Brier: mean over items of sum over classes of (p - onehot)^2."""
    p = np.asarray(probs, dtype=np.float64)
    onehot = np.eye(p.shape[1])[np.asarray(y_true)]
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))


def ece(probs, y_true, n_bins: int = ECE_BINS) -> float:
    """Expected calibration error: equal-width bins on top-label confidence, bins (lo, hi]."""
    p = np.asarray(probs, dtype=np.float64)
    conf, pred = p.max(axis=1), p.argmax(axis=1)
    correct = pred == np.asarray(y_true)
    bins = np.clip(np.ceil(conf * n_bins).astype(int) - 1, 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            total += mask.sum() * abs(correct[mask].mean() - conf[mask].mean())
    return float(total / len(conf))


def top2_margin(probs) -> np.ndarray:
    """Top-1 minus top-2 probability for each item."""
    top2 = np.sort(np.asarray(probs), axis=1)[:, -2:]
    return top2[:, 1] - top2[:, 0]


def mcnemar_exact(correct_a, correct_b) -> dict:
    """Exact McNemar test on paired per-item correctness (two-sided binomial on discordants)."""
    a, b = np.asarray(correct_a, dtype=bool), np.asarray(correct_b, dtype=bool)
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    n = only_a + only_b
    p = 1.0 if n == 0 else float(binomtest(min(only_a, only_b), n, 0.5).pvalue)
    return {"a_only_correct": only_a, "b_only_correct": only_b, "p_value": p}


def summarise(y_true, y_pred, probs, n_classes: int, *, seed: int) -> dict:
    """All headline metrics for one prediction set."""
    ci = bootstrap_ci(y_true, y_pred, n_classes, seed=seed)
    return {
        "n": len(y_true),
        "accuracy": accuracy(y_true, y_pred),
        "accuracy_ci_lo": ci["accuracy"][0],
        "accuracy_ci_hi": ci["accuracy"][1],
        "macro_f1": macro_f1(y_true, y_pred, n_classes),
        "macro_f1_ci_lo": ci["macro_f1"][0],
        "macro_f1_ci_hi": ci["macro_f1"][1],
        "brier": brier(probs, y_true),
        "ece": ece(probs, y_true),
    }
