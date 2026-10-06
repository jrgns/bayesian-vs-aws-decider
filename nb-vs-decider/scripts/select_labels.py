"""Apply the label-wording selection rule (config/labels/LOG.md) to the dev runs.

    python -m scripts.select_labels

Rule: at most three versions (v1, v2, v3); the highest dev accuracy at batch size 1 wins; a tie
(same accuracy to 4 d.p.) goes to the earlier version. Reads results/decider_runs.csv and
writes results/selected_labels.json. Never looks at the test split.
"""

from __future__ import annotations

import logging

import pandas as pd

from scripts.common import RESULTS_DIR, load_settings, setup_logging, write_json

log = logging.getLogger(__name__)

VERSIONS = ("v1", "v2", "v3")
TIE_DECIMALS = 4


def select_version(scores: dict[str, float]) -> str:
    """Highest accuracy (to TIE_DECIMALS); ties go to the earlier version in VERSIONS order."""
    best = None
    for v in VERSIONS:
        if v in scores and (
            best is None or round(scores[v], TIE_DECIMALS) > round(scores[best], TIE_DECIMALS)
        ):
            best = v
    if best is None:
        raise ValueError(f"no dev scores for any of {VERSIONS}")
    return best


def dev_scores(runs: pd.DataFrame) -> dict[str, float]:
    """Dev accuracy of run `dev_<version>_bs1` for every version that has one."""
    by_id = runs.set_index("run_id")["accuracy"]
    scores = {v: float(by_id[f"dev_{v}_bs1"]) for v in VERSIONS if f"dev_{v}_bs1" in by_id}
    missing = [v for v in VERSIONS if v not in scores]
    if missing:
        raise RuntimeError(f"dev runs missing for {missing}: run them before selecting")
    return scores


def main() -> None:
    setup_logging(load_settings().log_level)
    scores = dev_scores(pd.read_csv(RESULTS_DIR / "decider_runs.csv"))
    selected = select_version(scores)
    log.info("dev accuracy %s -> selected %s", scores, selected)
    write_json(
        "selected_labels.json",
        {
            "selected": selected,
            "dev_accuracy": scores,
            "rule": "highest dev accuracy at bs=1 (to 4 d.p.); ties go to the earlier version; "
            "at most v1, v2, v3 (config/labels/LOG.md)",
        },
    )


if __name__ == "__main__":
    main()
