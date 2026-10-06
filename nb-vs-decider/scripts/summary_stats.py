"""Small statistics helpers shared by the analysis stages (crossover, energy summary, plots)."""

from __future__ import annotations

import numpy as np
from scipy import stats


def mean_ci(values, level: float = 0.95) -> tuple[float, float, float]:
    """(mean, lo, hi): Student-t interval of the mean. Needs >= 2 values, else lo = hi = mean."""
    v = np.asarray(values, dtype=float)
    mean = float(v.mean())
    if len(v) < 2:
        return mean, mean, mean
    half = float(stats.t.ppf(0.5 + level / 2, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))
    return mean, mean - half, mean + half


def bootstrap_mean_ci(
    values, *, seed: int, n_boot: int = 10_000, level: float = 0.95, chunk: int = 250
) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean of per-item values (one resample of items per draw)."""
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = []
    for start in range(0, n_boot, chunk):
        idx = rng.integers(0, len(v), size=(min(chunk, n_boot - start), len(v)))
        means.append(v[idx].mean(axis=1))
    lo, hi = 100 * (1 - level) / 2, 100 * (1 + level) / 2
    return tuple(float(x) for x in np.percentile(np.concatenate(means), [lo, hi]))
