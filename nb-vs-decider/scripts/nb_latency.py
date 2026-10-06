"""NB per-item latency at the crossover n (the one measurement added after the benchmark runs).

    python -m scripts.nb_latency

Retrains the seed-0 best config from its saved subset, then predicts the first ITEMS test items
ONE AT A TIME (vectorise + predict_proba per item), timing each with perf_counter on a single
thread, as nb_curve does. No energy is measured. Texts live in data/processed/ (orion only).
Writes results/nb_latency.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_info, threadpool_limits

from scripts.common import RESULTS_DIR, load_settings, setup_logging, write_json
from scripts.decider_run import load_split
from scripts.nb_curve import PREDS_DIR, SUBSETS_DIR, make_pipeline

log = logging.getLogger(__name__)

ITEMS = 1000
WARMUP_ITEMS = 20


def percentiles_ms(seconds: list[float]) -> dict:
    ms = np.asarray(seconds) * 1000
    return {
        "p50_ms": float(np.percentile(ms, 50)),
        "p95_ms": float(np.percentile(ms, 95)),
        "mean_ms": float(ms.mean()),
        "min_ms": float(ms.min()),
        "max_ms": float(ms.max()),
    }


def time_one_at_a_time(model, texts: list[str], warmup: int = WARMUP_ITEMS) -> list[float]:
    for t in texts[:warmup]:
        model.predict_proba([t])
    out = []
    for t in texts:
        t0 = time.perf_counter()
        model.predict_proba([t])
        out.append(time.perf_counter() - t0)
    return out


def batch_per_item_from_runs(n: int) -> dict:
    """Whole-test-set predict_proba time / test size for the best config, from nb_runs.csv."""
    runs = pd.read_csv(RESULTS_DIR / "nb_runs.csv")
    best = runs[runs["is_best"] & (runs["n_per_class"].astype(str) == str(n))]
    n_test = json.loads((RESULTS_DIR / "dataset_summary.json").read_text())["n_items"]["test"]
    per_item_ms = best["predict_s"] / n_test * 1000
    return {
        "source": "results/nb_runs.csv, is_best rows, predict_s / test size",
        "per_seed_ms": per_item_ms.tolist(),
        "mean_ms": float(per_item_ms.mean()),
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=None, help="default: crossover n from crossover.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--items", type=int, default=ITEMS)
    args = ap.parse_args(argv)
    setup_logging(load_settings().log_level)

    n = (
        args.n
        or json.loads((RESULTS_DIR / "crossover.json").read_text())["headline"][
            "crossover_n_per_class"
        ]
    )
    name = f"n{n}_s{args.seed}"
    cfg = json.loads((RESULTS_DIR / "nb_best_configs.json").read_text())[name]
    pool, test = load_split("pool"), load_split("test")
    ids = pd.read_csv(SUBSETS_DIR / f"{name}.csv")
    sub = ids.merge(pool, on="item_id", how="left", validate="one_to_one")
    if sub["text"].isna().any() or len(sub) != 7 * n:
        raise RuntimeError(f"{name}: subset does not match the pool ({len(sub)} rows)")
    log.info("%s: retraining %s on %d items", name, cfg, len(sub))

    with threadpool_limits(limits=1):
        info = threadpool_info()
        model = make_pipeline(cfg["variant"], cfg["alpha"], cfg["min_df"])
        t0 = time.perf_counter()
        model.fit(sub["text"].tolist(), sub["label"].to_numpy())
        fit_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        probs = model.predict_proba(test["text"])
        batch_s = time.perf_counter() - t0
        saved = pd.read_parquet(PREDS_DIR / f"{name}_best.parquet", columns=["item_id", "pred"])
        saved = saved.set_index("item_id").loc[test["item_id"], "pred"].to_numpy()
        agree = float(np.mean(probs.argmax(axis=1) == saved))
        log.info("retrained model agrees with the saved predictions on %.4f of items", agree)

        texts = test["text"].tolist()[: args.items]
        lat = time_one_at_a_time(model, texts)

    write_json(
        "nb_latency.json",
        {
            "n_per_class": n,
            "seed": args.seed,
            "config": cfg,
            "n_train": len(sub),
            "fit_s": fit_s,
            "single_item": {"items": len(lat), "total_s": float(sum(lat)), **percentiles_ms(lat)},
            "batch_whole_test_this_run": {
                "items": len(test),
                "total_s": batch_s,
                "per_item_ms": batch_s / len(test) * 1000,
            },
            "batch_per_item_from_nb_runs": batch_per_item_from_runs(n),
            "retrain_agreement_with_saved_preds": agree,
            "threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
            "threadpool_info": info,
            "loadavg": os.getloadavg(),
            "method": "pipeline.predict_proba([text]) per item, perf_counter, 1 thread, after "
            f"{WARMUP_ITEMS} warmup calls; the first {len(lat)} test items in split order",
        },
    )


if __name__ == "__main__":
    main()
