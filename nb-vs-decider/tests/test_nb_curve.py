import warnings
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from scripts import nb_curve as nb

WORDS = ["alpha", "beta", "gamma", "delta", "eps", "zeta", "eta"]


def toy(per_class: int, classes: int = 7):
    """Texts whose words depend on the class, so every variant can learn them."""
    texts, labels = [], []
    for c in range(classes):
        for i in range(per_class):
            texts.append(f"{WORDS[c]} {WORDS[c]} common word{i % 3} {WORDS[(c + 1) % 7]}")
            labels.append(c)
    return texts, np.array(labels)


def test_sample_subset_is_deterministic_stratified_and_seeded():
    labels = np.repeat(np.arange(7), 20)
    a = nb.sample_subset(labels, 5, seed=1)
    assert np.array_equal(a, nb.sample_subset(labels, 5, seed=1))
    assert not np.array_equal(a, nb.sample_subset(labels, 5, seed=2))
    assert np.bincount(labels[a]).tolist() == [5] * 7
    assert list(a) == sorted(a)
    assert len(set(a)) == len(a)


def test_sample_subset_too_large_raises():
    with pytest.raises(ValueError):
        nb.sample_subset(np.repeat(np.arange(7), 3), 4, seed=0)


def test_n1_uses_defaults_and_picks_first_variant():
    texts, y = toy(1)
    fits, n_splits = nb.tune_all(texts, y, seed=0)
    assert n_splits == 0
    assert [f.variant for f in fits] == list(nb.VARIANTS)
    assert all(np.isnan(f.cv_score) for f in fits)
    assert all((f.alpha, f.min_df) == (1.0, 1) for f in fits)
    assert nb.select_best(fits).variant == "mnb_count"


def grid_search_scores(variant, texts, y, n_splits, seed):
    """Reference: the GridSearchCV the manual loop must reproduce exactly."""
    gs = GridSearchCV(
        nb.make_pipeline(variant),
        {"nb__alpha": list(nb.ALPHAS), "vec__min_df": list(nb.MIN_DFS)},
        scoring="accuracy",
        cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed),
        n_jobs=1,
        error_score=np.nan,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gs.fit(texts, y)
    return gs


def test_empty_vocab_grid_points_score_nan_not_crash():
    # Unique words per doc: min_df=2/3 leave an empty vocabulary in every fold.
    texts = [f"uniq{i} word{i}" for i in range(14)]
    y = np.repeat(np.arange(7), 2)
    means = nb.cv_scores(texts, y, n_splits=2, seed=0)["mnb_count"]
    for (_, min_df), m in zip(nb.GRID, means, strict=True):
        assert np.isnan(m) == (min_df > 1)
    fit = nb.tune_variant("mnb_count", means, texts, y)
    assert fit.min_df == 1
    assert fit.vocab_size > 0


@pytest.mark.parametrize("per_class,seed", [(6, 0), (8, 3)])
def test_manual_cv_matches_grid_search_exactly(per_class, seed):
    texts, y = toy(per_class)
    n_splits = nb.n_splits_for(per_class)
    means = nb.cv_scores(texts, y, n_splits, seed)
    for variant in nb.VARIANTS:
        gs = grid_search_scores(variant, texts, y, n_splits, seed)
        ref = gs.cv_results_["mean_test_score"]
        assert np.array_equal(means[variant], ref, equal_nan=True)
        fit = nb.tune_variant(variant, means[variant], texts, y)
        assert (fit.alpha, fit.min_df) == (
            gs.best_params_["nb__alpha"],
            gs.best_params_["vec__min_df"],
        )
        assert fit.cv_score == gs.best_score_


def test_tuned_cell_selects_by_cv_with_tie_break():
    texts, y = toy(6)
    fits, n_splits = nb.tune_all(texts, y, seed=0)
    assert n_splits == 5
    best = nb.select_best(fits)
    top = max(f.cv_score for f in fits)
    assert best.cv_score == top
    assert best is next(f for f in fits if f.cv_score == top)  # first of the tied variants


def test_select_best_ties_go_to_first():
    mk = lambda v, s: nb.Fitted(v, None, s, 1.0, 1, 0.0)  # noqa: E731
    fits = [mk("a", 0.5), mk("b", 0.7), mk("c", 0.7), mk("d", 0.6)]
    assert nb.select_best(fits).variant == "b"


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class FakeMeter:
    def __init__(self, joules_per_s=2.0):
        self.joules_per_s, self.labels = joules_per_s, []

    @contextmanager
    def measure(self, label, idle=None):
        self.labels.append(label)
        m = SimpleNamespace(record=None)
        yield m
        dur = self.clock.t - 0.0
        m.record = SimpleNamespace(
            cpu_pkg_J=self.joules_per_s * dur,
            cpu_pkg_marginal_J=None,
            gpu_J=1.0,
            duration_s=dur,
            energy_id=f"x:{label}",
        )


def test_repeat_loop_runs_until_window_and_divides_energy():
    clock = FakeClock()
    meter = FakeMeter(joules_per_s=2.0)
    meter.clock = clock
    calls = []

    def fn():
        calls.append(1)
        clock.t += 0.3  # each repetition takes 0.3 s
        return len(calls)

    value, e = nb.measured_phase(meter, "fit", None, fn, 10.0, clock)
    assert value == 1  # result of the first call
    assert e["reps"] == len(calls) == 34  # 34 * 0.3 = 10.2 s >= 10 s, 33 * 0.3 = 9.9 s < 10 s
    assert e["window_s"] == pytest.approx(10.2)
    assert e["cpu_J"] == pytest.approx(2.0 * 10.2 / 34)
    assert e["cpu_marginal_J"] is None


def test_slow_phase_runs_once():
    clock = FakeClock()
    n = []

    def fn():
        n.append(1)
        clock.t += 12.0

    _, reps = nb.run_repeated(fn, 10.0, clock)
    assert reps == 1 == len(n)
