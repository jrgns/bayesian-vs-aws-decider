"""Naive Bayes learning curve on the POOL split, evaluated on TEST, with optional energy.

    python -m scripts.nb_curve --measure-energy

For n in N_PER_CLASS (examples per class) and each seed: sample a stratified subset of the pool,
tune {Multinomial,Complement}NB x {Count,Tfidf}Vectorizer by stratified CV inside the subset,
refit, and score every variant on the test split (reporting only, never used for selection).
The final point, "full", is the whole pool (imbalanced as-is): its subset is identical for every
seed and the seeds only change the CV fold shuffle.

Notes recorded in results/nb_meta.json:
- n=1 per class has no CV: all variants use alpha=1.0, min_df=1 and MNB+Count is "best" by rule.
- ComplementNB's predict_proba is a softmax over its complement-weighted scores. It is not a
  calibrated probability in the usual sense, so its Brier/ECE are not comparable to MNB's.
- Energy phases per (n, seed): tune (GridSearchCV over all 4 variants; at n=1 there is no CV, so
  it is the 4 default fits), fit (best config refit), infer (vectorise + predict_proba on all
  test items, best config). Each phase is looped until its window lasts >= MIN_WINDOW_S and the
  energy is divided by the repeat count.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_info, threadpool_limits

from scripts import metrics
from scripts.common import RESULTS_DIR, append_csv, load_settings, setup_logging
from scripts.decider_run import N_CLASSES, load_split
from scripts.energy import Meter

log = logging.getLogger(__name__)

N_PER_CLASS = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000)  # the pre-registered grid
# Added after the main run to refine the crossover between 50 and 100 (same protocol, same seeds).
REFINE_N_PER_CLASS = (60, 70, 80, 90)
FULL = "full"
SEEDS = (0, 1, 2, 3, 4)
ALPHAS = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
MIN_DFS = (1, 2, 3)
DEFAULT_ALPHA, DEFAULT_MIN_DF = 1.0, 1
MAX_FOLDS = 5
MIN_WINDOW_S = 10.0
PHASES = ("tune", "fit", "infer")

# Fixed order: also the tie-break order for "best".
VARIANTS: dict[str, tuple[type, type]] = {
    "mnb_count": (MultinomialNB, CountVectorizer),
    "mnb_tfidf": (MultinomialNB, TfidfVectorizer),
    "cnb_count": (ComplementNB, CountVectorizer),
    "cnb_tfidf": (ComplementNB, TfidfVectorizer),
}


def refined_sizes(preds_dir=None, seeds=SEEDS) -> list[int]:
    """Planned grid plus those refinement sizes whose best-preds exist for every seed, ascending."""
    preds_dir = preds_dir or PREDS_DIR
    extra = [
        n
        for n in REFINE_N_PER_CLASS
        if all((preds_dir / f"{cell_name(n, s)}_best.parquet").exists() for s in seeds)
    ]
    return sorted({*N_PER_CLASS, *extra})


SUBSETS_DIR = RESULTS_DIR / "nb" / "subsets"
PREDS_DIR = RESULTS_DIR / "nb" / "preds"


# ------------------------------------------------------------------------------ subsets


def cell_name(n: int | str, seed: int) -> str:
    return f"n{n}_s{seed}"


def sample_subset(labels: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Row positions (ascending) of n items per class: stratified, seeded, deterministic."""
    rng = np.random.default_rng(seed * 1000 + n)
    keep: list[int] = []
    for c in np.unique(labels):
        pos = np.flatnonzero(labels == c)
        if n > len(pos):
            raise ValueError(f"class {c}: asked for {n}, only {len(pos)} available")
        keep.extend(rng.choice(pos, size=n, replace=False).tolist())
    return np.array(sorted(keep))


# ------------------------------------------------------------------------------ models


def make_pipeline(variant: str, alpha: float = DEFAULT_ALPHA, min_df: int = DEFAULT_MIN_DF):
    nb_cls, vec_cls = VARIANTS[variant]
    return Pipeline(
        [
            ("vec", vec_cls(ngram_range=(1, 2), lowercase=True, min_df=min_df)),
            ("nb", nb_cls(alpha=alpha)),
        ]
    )


@dataclass
class Fitted:
    variant: str
    model: Pipeline
    cv_score: float  # mean CV accuracy of the chosen grid point; NaN when not tuned
    alpha: float
    min_df: int
    fit_s: float  # time to fit the chosen config on the full subset

    @property
    def vocab_size(self) -> int:
        return len(self.model.named_steps["vec"].vocabulary_)


def n_splits_for(min_class_count: int) -> int:
    return min(MAX_FOLDS, min_class_count)


# Grid points in sklearn ParameterGrid order for {"nb__alpha", "vec__min_df"}: keys are sorted
# and the last key varies fastest, so alpha is the outer loop and min_df the inner one.
GRID = [(alpha, min_df) for alpha in ALPHAS for min_df in MIN_DFS]


def cv_scores(texts, y, n_splits: int, seed: int) -> dict[str, np.ndarray]:
    """Mean CV accuracy of every grid point (GRID order) for every variant.

    Equivalent to one GridSearchCV per variant over the same StratifiedKFold split, but each
    vectorizer is fit once per (fold, vectorizer type, min_df) and shared by all alphas and by
    both NB classes. A grid point whose vectorizer (or NB) fit raises, e.g. an empty vocabulary,
    scores NaN, like error_score=np.nan.
    """
    texts = np.asarray(texts, dtype=object)
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(texts, y)
    per_fold = {v: np.full((len(GRID), n_splits), np.nan) for v in VARIANTS}
    for k, (tr, va) in enumerate(folds):
        for vec_cls in (CountVectorizer, TfidfVectorizer):
            for j, min_df in enumerate(MIN_DFS):
                vec = vec_cls(ngram_range=(1, 2), lowercase=True, min_df=min_df)
                try:
                    x_tr = vec.fit_transform(texts[tr])
                    x_va = vec.transform(texts[va])
                except ValueError:
                    continue
                for variant, (nb_cls, vc) in VARIANTS.items():
                    if vc is not vec_cls:
                        continue
                    for i, alpha in enumerate(ALPHAS):
                        try:
                            pred = nb_cls(alpha=alpha).fit(x_tr, y[tr]).predict(x_va)
                        except ValueError:
                            continue
                        per_fold[variant][i * len(MIN_DFS) + j, k] = accuracy_score(y[va], pred)
    return {v: np.average(a, axis=1) for v, a in per_fold.items()}


def tune_variant(variant: str, means: np.ndarray, texts, y) -> Fitted:
    """Best grid point (first of any ties, as GridSearchCV), refit on the whole subset."""
    alpha, min_df = GRID[int(np.nanargmax(means))]
    fit = default_variant(variant, texts, y, alpha, min_df)
    fit.cv_score = float(np.nanmax(means))
    return fit


def default_variant(
    variant: str, texts, y, alpha: float = DEFAULT_ALPHA, min_df: int = DEFAULT_MIN_DF
) -> Fitted:
    """Fit one config on the whole subset (the defaults unless given), timing the fit."""
    model = make_pipeline(variant, alpha, min_df)
    t0 = time.perf_counter()
    model.fit(texts, y)
    fit_s = time.perf_counter() - t0
    return Fitted(variant, model, float("nan"), alpha, min_df, fit_s)


def tune_all(texts, y, seed: int) -> tuple[list[Fitted], int]:
    """All four variants for one subset. Returns the fits and the number of CV folds (0 = none)."""
    n_splits = n_splits_for(int(np.bincount(y).min()))
    if n_splits < 2:
        return [default_variant(v, texts, y) for v in VARIANTS], 0
    means = cv_scores(texts, y, n_splits, seed)
    return [tune_variant(v, means[v], texts, y) for v in VARIANTS], n_splits


def select_best(fits: list[Fitted]) -> Fitted:
    """Highest mean CV accuracy; ties (and the no-CV case) go to the first variant in order."""
    best = fits[0]
    for f in fits[1:]:
        if not np.isnan(f.cv_score) and (np.isnan(best.cv_score) or f.cv_score > best.cv_score):
            best = f
    return best


# ------------------------------------------------------------------------------ energy


def run_repeated(fn: Callable, min_s: float, clock=time.perf_counter):
    """Call fn until at least min_s has elapsed. Returns (first call's result, repeat count)."""
    t0 = clock()
    first = fn()
    reps = 1
    while clock() - t0 < min_s:
        fn()
        reps += 1
    return first, reps


def per_repeat(rec, reps: int) -> dict:
    """Energy of one repetition: the window's energy divided by the repeat count."""

    def div(v):
        return None if v is None else v / reps

    return {
        "reps": reps,
        "cpu_J": div(rec.cpu_pkg_J),
        "cpu_marginal_J": div(rec.cpu_pkg_marginal_J),
        "gpu_J": div(rec.gpu_J),
        "window_s": rec.duration_s,
        "energy_id": rec.energy_id,
    }


def measured_phase(meter, label: str, idle, fn: Callable, min_s: float, clock=time.perf_counter):
    """Run fn repeatedly inside one meter window. Returns (result, per-repeat energy dict)."""
    with meter.measure(label, idle) as m:
        value, reps = run_repeated(fn, min_s, clock)
    return value, per_repeat(m.record, reps)


# ------------------------------------------------------------------------------ evaluation


def evaluate(fit: Fitted, test_texts, y_test) -> tuple[dict, np.ndarray]:
    t0 = time.perf_counter()
    probs = fit.model.predict_proba(test_texts)
    predict_s = time.perf_counter() - t0
    if list(fit.model.classes_) != list(range(N_CLASSES)):
        raise RuntimeError(f"{fit.variant}: unexpected classes {fit.model.classes_}")
    pred = probs.argmax(axis=1)
    return {
        "test_accuracy": metrics.accuracy(y_test, pred),
        "test_macro_f1": metrics.macro_f1(y_test, pred, N_CLASSES),
        "test_brier": metrics.brier(probs, y_test),
        "test_ece": metrics.ece(probs, y_test),
        "predict_s": predict_s,
    }, probs


def write_best_preds(name: str, ids, y_test, probs) -> None:
    out = pd.DataFrame({"item_id": ids, "label": y_test.astype("int64")})
    out["pred"] = probs.argmax(axis=1).astype("int64")
    for j in range(N_CLASSES):
        out[f"probs_{j}"] = probs[:, j].astype(np.float32)
    out["margin"] = metrics.top2_margin(probs).astype(np.float32)
    PREDS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(PREDS_DIR / f"{name}_best.parquet", index=False)


def write_variant_preds(name: str, ids, preds: dict[str, np.ndarray]) -> None:
    out = pd.DataFrame({"item_id": ids})
    for variant, p in preds.items():
        out[variant] = p.astype(np.int8)
    PREDS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(PREDS_DIR / f"{name}_variants.parquet", index=False)


ENERGY_COLS = [f"{p}_{k}" for p in PHASES for k in ("reps", "cpu_J", "cpu_marginal_J")]


def energy_columns(phases: dict[str, dict] | None) -> dict:
    row = dict.fromkeys(ENERGY_COLS)
    for p, e in (phases or {}).items():
        row.update({f"{p}_reps": e["reps"], f"{p}_cpu_J": e["cpu_J"]})
        row[f"{p}_cpu_marginal_J"] = e["cpu_marginal_J"]
    return row


def update_best_configs(path, key: str, best: Fitted) -> None:
    cfg = json.loads(path.read_text()) if path.exists() else {}
    cfg[key] = {"variant": best.variant, "alpha": best.alpha, "min_df": best.min_df}
    path.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")


# ------------------------------------------------------------------------------ main


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sizes", nargs="+", default=[*map(str, N_PER_CLASS), FULL])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--measure-energy", action="store_true")
    ap.add_argument("--idle-seconds", type=float, default=60)
    ap.add_argument("--min-window-s", type=float, default=MIN_WINDOW_S)
    ap.add_argument("--tag", default="", help="suffix for output names (re-runs of single cells)")
    return ap.parse_args(argv)


def run_cell(n, seed, pool, test, args, meter, idle, runs_csv, best_path, idle_id) -> dict:
    full = n == FULL
    if full:
        rows = np.arange(len(pool))  # same subset for every seed; the seed only shuffles CV folds
    else:
        rows = sample_subset(pool["label"].to_numpy(), int(n), seed)
    sub = pool.iloc[rows]
    name = cell_name(n, seed)
    SUBSETS_DIR.mkdir(parents=True, exist_ok=True)
    sub[["item_id"]].to_csv(SUBSETS_DIR / f"{name}{args.tag}.csv", index=False)
    texts, y = sub["text"].tolist(), sub["label"].to_numpy()
    if meter:
        meter.run_id = f"nb_{name}"

    def tune():
        return tune_all(texts, y, seed)

    if meter:
        (fits, n_splits), e_tune = measured_phase(meter, "tune", idle, tune, args.min_window_s)
    else:
        (fits, n_splits), e_tune = tune(), None
    best = select_best(fits)
    log.info(
        "%s: best %s alpha=%g min_df=%d cv=%.4f (%d folds)",
        name, best.variant, best.alpha, best.min_df, best.cv_score, n_splits,
    )  # fmt: skip

    phases = None
    if meter:
        _, e_fit = measured_phase(
            meter, "fit", idle, lambda: clone(best.model).fit(texts, y), args.min_window_s
        )
        _, e_inf = measured_phase(
            meter, "infer", idle, lambda: best.model.predict_proba(test["text"]), args.min_window_s
        )
        phases = {"tune": e_tune, "fit": e_fit, "infer": e_inf}

    y_test, ids = test["label"].to_numpy(), test["item_id"]
    non_best: dict[str, np.ndarray] = {}
    for f in fits:
        stats, probs = evaluate(f, test["text"], y_test)
        is_best = f is best
        if is_best:
            write_best_preds(f"{name}{args.tag}", ids, y_test, probs)
        else:
            non_best[f.variant] = probs.argmax(axis=1)
        energy_ids = ""
        if is_best and phases:
            energy_ids = ";".join([idle_id, *(phases[p]["energy_id"] for p in PHASES)])
        append_csv(
            runs_csv,
            {
                "n_per_class": n,
                "seed": seed,
                "n_train": len(sub),
                "variant": f.variant,
                "is_best": is_best,
                "tuned": n_splits >= 2,
                "cv_folds": n_splits,
                "cv_score": f.cv_score,
                "alpha": f.alpha,
                "min_df": f.min_df,
                "vocab_size": f.vocab_size,
                **stats,
                "fit_s": f.fit_s,
                "energy_ids": energy_ids,
                **energy_columns(phases if is_best else None),
            },
        )
        log.info("%s %-9s acc %.4f f1 %.4f", name, f.variant, stats["test_accuracy"],
                 stats["test_macro_f1"])  # fmt: skip
    write_variant_preds(f"{name}{args.tag}", ids, non_best)
    update_best_configs(best_path, name, best)
    return {"name": name, "best": best.variant}


def meta_path(tag: str, sizes, results_dir=RESULTS_DIR):
    """nb_meta<tag>.json, unless it exists: then a per-sizes file, so the first run's meta stays."""
    path = results_dir / f"nb_meta{tag}.json"
    if path.exists():
        path = results_dir / f"nb_meta{tag}_n{'-'.join(map(str, sizes))}.json"
    return path


def main(argv=None) -> None:
    args = parse_args(argv)
    s = load_settings()
    setup_logging(s.log_level)
    t_start = time.perf_counter()
    # Training subsets come from the pool only. Test is loaded for evaluation of chosen configs.
    pool, test = load_split("pool"), load_split("test")
    log.info("pool %d items, test %d items", len(pool), len(test))
    runs_csv = RESULTS_DIR / f"nb_runs{args.tag}.csv"
    best_path = RESULTS_DIR / f"nb_best_configs{args.tag}.json"

    meter, idle, idle_id = None, None, ""
    cells = [(n if n == FULL else int(n), seed) for n in args.sizes for seed in args.seeds]
    with threadpool_limits(limits=1):
        info = threadpool_info()
        log.info("threadpools: %s", info)
        if args.measure_energy:
            meter = Meter(f"nb_{cell_name(*cells[0])}")  # idle row carries the first cell's id
            idle = meter.idle_baseline(args.idle_seconds)
            idle_id = idle.record.energy_id
        for n, seed in cells:
            run_cell(n, seed, pool, test, args, meter, idle, runs_csv, best_path, idle_id)

    meta = {
        "finished_utc": datetime.now(UTC).isoformat(),
        "runtime_s": time.perf_counter() - t_start,
        "args": vars(args),
        "threadpool_info": info,
        "idle_energy_id": idle_id or None,
        "alphas": ALPHAS,
        "min_dfs": MIN_DFS,
        "notes": {
            "full_pool": "subset identical for all seeds; seeds only vary the CV fold shuffle",
            "n=1": "no CV: defaults alpha=1.0, min_df=1; MNB+Count chosen as best by fixed rule; "
            "the tune phase is the 4 default fits",
            "complement_nb": "predict_proba is a softmax of ComplementNB scores, not a calibrated "
            "probability; its Brier/ECE are not comparable to MultinomialNB's",
            "tie_break": list(VARIANTS),
            "energy": "CPU RAPL package-0 attributed to NB; GPU counter recorded, not attributed",
        },
    }
    meta_path(args.tag, args.sizes).write_text(json.dumps(meta, indent=2) + "\n")
    log.info("done in %.1f s", meta["runtime_s"])


if __name__ == "__main__":
    main()
