"""Crossover analysis: the smallest NB training size that matches Decider's zero-shot accuracy.

    python -m scripts.crossover

For every n (examples per class) and seed, an exact McNemar test compares the NB best config's
per-item correctness with a Decider test run on identical item_ids (PLAN section 6).
Crossover rule: the smallest n where NB's mean accuracy >= Decider's AND, for a majority of the
seeds, McNemar is not significant at ALPHA or NB is significantly better.
The full imbalanced pool is reported separately and never enters the "smallest n" search.
Writes results/crossover.csv (one row per run x n x seed) and results/crossover.json.
"""

from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd

from scripts import metrics
from scripts.common import RESULTS_DIR, decider_test_runs, load_settings, setup_logging
from scripts.nb_curve import FULL, N_PER_CLASS, PREDS_DIR, SEEDS
from scripts.summary_stats import bootstrap_mean_ci, mean_ci

log = logging.getLogger(__name__)

ALPHA = 0.05
BOOT_SEED = 20261006
N_CLASSES = 7


# ------------------------------------------------------------------------------ rule (pure)


def direction(nb_only: int, decider_only: int) -> str:
    if nb_only > decider_only:
        return "nb_better"
    return "decider_better" if decider_only > nb_only else "tie"


def seed_qualifies(p_value: float, dir_: str, alpha: float = ALPHA) -> bool:
    """McNemar is not significant, or NB is significantly better."""
    return not (p_value < alpha and dir_ == "decider_better")


def n_qualifies(nb_mean_acc: float, decider_acc: float, seeds_ok: int, n_seeds: int) -> bool:
    """Mean accuracy at least Decider's, and a strict majority of the seeds qualify."""
    return nb_mean_acc >= decider_acc and 2 * seeds_ok > n_seeds


def find_crossover(per_n: list[dict]) -> int | None:
    """Smallest n whose summary qualifies. `per_n` rows need n, qualifies; any order."""
    ok = [r["n"] for r in per_n if r["qualifies"]]
    return min(ok) if ok else None


# ------------------------------------------------------------------------------ data


def load_preds(path) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=["item_id", "label", "pred"])
    return df.set_index("item_id").sort_index()


def correctness(nb: pd.DataFrame, decider: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Per-item correctness of both, after asserting identical item sets and labels."""
    assert set(nb.index) == set(decider.index), "NB and Decider test item sets differ"
    assert nb.index.is_unique and decider.index.is_unique
    assert (nb["label"].to_numpy() == decider["label"].to_numpy()).all(), "labels differ"
    return (nb["pred"] == nb["label"]).to_numpy(), (decider["pred"] == decider["label"]).to_numpy()


def compare(nb_ok: np.ndarray, dec_ok: np.ndarray) -> dict:
    m = metrics.mcnemar_exact(nb_ok, dec_ok)
    b, c = m["a_only_correct"], m["b_only_correct"]
    d = direction(b, c)
    return {
        "nb_accuracy": float(nb_ok.mean()),
        "decider_accuracy": float(dec_ok.mean()),
        "nb_only_correct": b,
        "decider_only_correct": c,
        "p_value": m["p_value"],
        "direction": d,
        "significant": m["p_value"] < ALPHA,
        "seed_qualifies": seed_qualifies(m["p_value"], d),
    }


# ------------------------------------------------------------------------------ analysis


def analyse_run(run_id: str, sizes, seeds) -> tuple[list[dict], list[dict]]:
    """Per-(n, seed) rows and per-n summaries (with the gap to Decider and its CI)."""
    decider = load_preds(RESULTS_DIR / "decider" / f"{run_id}.parquet")
    rows, per_n = [], []
    for n in sizes:
        nb_ok_by_seed = []
        n_rows = []
        for seed in seeds:
            nb = load_preds(PREDS_DIR / f"n{n}_s{seed}_best.parquet")
            nb_ok, dec_ok = correctness(nb, decider)
            nb_ok_by_seed.append(nb_ok)
            n_rows.append({"decider_run": run_id, "n_per_class": n, "seed": seed,
                           "in_search": n != FULL, **compare(nb_ok, dec_ok)})  # fmt: skip
        rows += n_rows
        accs = [r["nb_accuracy"] for r in n_rows]
        mean, lo, hi = mean_ci(accs)
        dec_acc = float(dec_ok.mean())
        gap = dec_ok.astype(float) - np.mean(nb_ok_by_seed, axis=0)  # + means NB is behind
        gap_lo, gap_hi = bootstrap_mean_ci(gap, seed=BOOT_SEED)
        seeds_ok = sum(r["seed_qualifies"] for r in n_rows)
        per_n.append(
            {
                "n": n,
                "nb_mean_accuracy": mean,
                "nb_accuracy_ci_lo": lo,
                "nb_accuracy_ci_hi": hi,
                "decider_accuracy": dec_acc,
                "gap_decider_minus_nb": float(gap.mean()),
                "gap_ci_lo": gap_lo,
                "gap_ci_hi": gap_hi,
                "seeds_qualifying": seeds_ok,
                "n_seeds": len(seeds),
                "qualifies": n_qualifies(mean, dec_acc, seeds_ok, len(seeds)),
                "per_seed": [
                    {
                        k: r[k]
                        for k in (
                            "seed",
                            "nb_only_correct",
                            "decider_only_correct",
                            "p_value",
                            "direction",
                            "significant",
                            "seed_qualifies",
                        )
                    }
                    for r in n_rows
                ],  # fmt: skip
            }
        )
    return rows, per_n


def summarise_run(run_id: str, per_n: list[dict]) -> dict:
    """Crossover and the gap one grid step below it (search covers numeric sizes only)."""
    search = sorted((r for r in per_n if r["n"] != FULL), key=lambda r: r["n"])
    cross = find_crossover(search)
    below = [r for r in search if cross is not None and r["n"] < cross]
    return {
        "decider_run": run_id,
        "crossover_n_per_class": cross,
        "at_crossover": next((r for r in search if r["n"] == cross), None),
        "largest_n_below_crossover": below[-1] if below else None,
        "best_gap_if_never": None if cross is not None else min(
            search, key=lambda r: r["gap_decider_minus_nb"]),
        "per_n": search,
    }  # fmt: skip


def full_pool_report(run_id: str, per_n_full: dict, summary: dict) -> dict:
    """Full-pool point plus the evidence for the prior-shift explanation (seed 0 / n=1000)."""
    ds = json.loads((RESULTS_DIR / "dataset_summary.json").read_text())
    names = [ds["label_map"][str(i)]["name"] for i in range(N_CLASSES)]
    pool = ds["per_class"]["pool"]
    n_pool = sum(pool.values())
    out = {
        **{k: v for k, v in per_n_full.items() if k != "n"},
        "pool_size": n_pool,
        "pool_class_share": {k: v / n_pool for k, v in pool.items()},
        "x_equivalent_per_class": n_pool / N_CLASSES,
        "test_class_share": 1 / N_CLASSES,
        "in_search": False,
    }
    for key, tag in (("full", "nfull_s0"), ("n1000", "n1000_s0")):
        df = load_preds(PREDS_DIR / f"{tag}_best.parquet")
        y, p = df["label"].to_numpy(), df["pred"].to_numpy()
        out[f"{key}_seed0"] = {
            "predicted_share": {names[i]: float((p == i).mean()) for i in range(N_CLASSES)},
            "recall": dict(zip(names, metrics.per_class_recall(y, p, N_CLASSES).tolist(),
                               strict=True)),
        }  # fmt: skip
    out["note"] = (
        "The pool is imbalanced (37% Medical) while the test split is balanced (1/7 per class): "
        "a prior shift. NB learns the pool's class priors, so it over-predicts the majority "
        "class on the balanced test set and loses recall on the small classes."
    )
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-prefix", default="crossover")
    args = ap.parse_args(argv)
    setup_logging(load_settings().log_level)

    sizes, seeds = [*N_PER_CLASS, FULL], list(SEEDS)
    all_rows, summaries = [], {}
    # Headline = the bs=1 run; robustness = the best-throughput batch size (absent if that is 1).
    roles = zip(("headline", "robustness"), decider_test_runs().values(), strict=False)
    for role, run_id in roles:
        rows, per_n = analyse_run(run_id, sizes, seeds)
        all_rows += rows
        full = next(r for r in per_n if r["n"] == FULL)
        summaries[role] = summarise_run(run_id, per_n)
        summaries[role]["full_pool"] = full_pool_report(run_id, full, summaries[role])
        log.info("%s (%s): crossover at n=%s per class", role, run_id,
                 summaries[role]["crossover_n_per_class"])  # fmt: skip

    pd.DataFrame(all_rows).to_csv(RESULTS_DIR / f"{args.out_prefix}.csv", index=False)
    out = {
        "alpha": ALPHA,
        "rule": "smallest n with NB mean accuracy >= Decider's and, for a majority of the seeds, "
        "exact McNemar not significant at alpha or NB significantly better (PLAN section 6); "
        "the full imbalanced pool is excluded from the search",
        "gap_definition": "Decider accuracy minus NB mean accuracy over seeds; the CI is a paired "
        "percentile bootstrap over test items (10,000 resamples) of the per-item difference",
        "seeds": seeds,
        **summaries,
    }
    (RESULTS_DIR / f"{args.out_prefix}.json").write_text(json.dumps(out, indent=2) + "\n")
    log.info("wrote %s.csv and %s.json", args.out_prefix, args.out_prefix)


if __name__ == "__main__":
    main()
