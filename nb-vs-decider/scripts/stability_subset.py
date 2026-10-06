"""Stability subset: 1,000 test items, half near the decision boundary, half the rest.

    python -m scripts.stability_subset

Near-boundary stratum: the NEAR_POOL lowest-margin test items of the main Decider run
(results/decider/test_<labels>_bs1.parquet, `margin_top2`, ties broken by item_id), 500 sampled
proportionally by true label. Rest stratum: 500 items from the other test items, class-balanced
(72 for each of the first 3 labels, 71 for the rest). The result is shuffled once with the seed;
that order is the canonical item order of every stability run. Writes stability/subset_ids.csv
(item_id, label, stratum). Only the main run's ids, labels and margins are read: no texts.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from scripts.common import APP_DIR, RESULTS_DIR, decider_test_runs, load_settings, setup_logging

log = logging.getLogger(__name__)

SEED = 20261007
N_PER_STRATUM = 500
NEAR_POOL = 1000
N_CLASSES = 7
STABILITY_DIR = APP_DIR / "stability"
SUBSET_CSV = STABILITY_DIR / "subset_ids.csv"
NEAR, REST = "near_boundary", "rest"


def proportional_counts(class_sizes: dict[int, int], total: int) -> dict[int, int]:
    """Largest-remainder allocation of `total` over classes, proportional to their sizes."""
    n = sum(class_sizes.values())
    exact = {c: total * s / n for c, s in class_sizes.items()}
    counts = {c: int(np.floor(v)) for c, v in exact.items()}
    left = total - sum(counts.values())
    for c in sorted(exact, key=lambda c: (-(exact[c] - counts[c]), c))[:left]:
        counts[c] += 1
    return counts


def balanced_counts(classes: list[int], total: int) -> dict[int, int]:
    """Equal split; the remainder goes to the first classes in label order."""
    base, extra = divmod(total, len(classes))
    return {c: base + (1 if i < extra else 0) for i, c in enumerate(sorted(classes))}


def _draw(df: pd.DataFrame, counts: dict[int, int], rng: np.random.Generator) -> pd.DataFrame:
    parts = []
    for c in sorted(counts):
        pool = df[df["label"] == c].sort_values("item_id")
        if counts[c] > len(pool):
            raise ValueError(f"class {c}: asked for {counts[c]}, only {len(pool)} available")
        pick = rng.choice(len(pool), size=counts[c], replace=False)
        parts.append(pool.iloc[np.sort(pick)])
    return pd.concat(parts)


def build_subset(
    main: pd.DataFrame,
    seed: int = SEED,
    n_per_stratum: int = N_PER_STRATUM,
    near_pool: int = NEAR_POOL,
) -> pd.DataFrame:
    """main: item_id, label, margin_top2 of the whole test split -> item_id, label, stratum."""
    main = main[["item_id", "label", "margin_top2"]]
    ranked = main.sort_values(["margin_top2", "item_id"], kind="stable")
    low, other = ranked.iloc[:near_pool], ranked.iloc[near_pool:]
    rng = np.random.default_rng(seed)
    near_counts = proportional_counts(low["label"].value_counts().to_dict(), n_per_stratum)
    near = _draw(low, near_counts, rng).assign(stratum=NEAR)
    rest_counts = balanced_counts(sorted(other["label"].unique().tolist()), n_per_stratum)
    rest = _draw(other, rest_counts, rng).assign(stratum=REST)
    both = pd.concat([near, rest])
    order = rng.permutation(len(both))
    out = both.iloc[order][["item_id", "label", "stratum"]].reset_index(drop=True)
    out["label"] = out["label"].astype("int64")
    return out


def main() -> None:
    setup_logging(load_settings().log_level)
    run_id = decider_test_runs()[1]
    main_run = pd.read_parquet(
        RESULTS_DIR / "decider" / f"{run_id}.parquet", columns=["item_id", "label", "margin_top2"]
    )
    sub = build_subset(main_run)
    STABILITY_DIR.mkdir(parents=True, exist_ok=True)
    sub.to_csv(SUBSET_CSV, index=False)
    log.info("wrote %s: %d items from %s", SUBSET_CSV, len(sub), run_id)
    log.info("strata %s", sub["stratum"].value_counts().to_dict())
    log.info("labels by stratum:\n%s", pd.crosstab(sub["stratum"], sub["label"]))


if __name__ == "__main__":
    main()
