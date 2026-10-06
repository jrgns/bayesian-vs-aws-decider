"""Stability metrics: stability/runs -> results.csv, flipped_examples.json, versions.json.

    python -m scripts.stability_metrics

One row per condition x stratum (all, near_boundary, rest, reweighted). Pooled rows join the
runs of several conditions: D4 = D1 + D2 pooled, D4_paired = D1 run i vs D2 run i, and
D5+D2 / D6a+D2 / D6b+D2 = the condition pooled with its D2 baseline.

Definitions (docs/PLAN_STABILITY.md):
- flip: item whose predicted label is not identical across all runs; flip_rate is their share
- pairwise_disagreement: mean over all run pairs of the share of items predicted differently
- bitwise_identical: share of items whose raw scores are byte-identical across all runs
  (Decider: raw float32 logits; NB: predict_proba bytes); max_abs_diff: largest per-entry spread
- reweighted: 0.1 x near_boundary + 0.9 x rest; every rate is additive, so it is the weighted
  mean of the stratum rates (accuracy bounds are the min/max over runs of the weighted accuracy)
"""

from __future__ import annotations

import itertools
import json
import logging

import numpy as np
import pandas as pd

from scripts.common import RESULTS_DIR, load_settings, setup_logging
from scripts.stability_subset import NEAR, REST, STABILITY_DIR, SUBSET_CSV

log = logging.getLogger(__name__)

RUNS_DIR = STABILITY_DIR / "runs"
WEIGHTS = {NEAR: 0.1, REST: 0.9}
STRATA = ("all", NEAR, REST, "reweighted")
N_CLASSES = 7
INVENTORY_COMMIT = "3e94e9d84c620ed5a95f1a3310c3decb971e261c"
CONDITIONS = ("N1", "N2", "D1", "D2", "D3", "D3w", "D5", "D6a", "D6b")
SPEED_COLS = ("wall_s", "items_per_s", "batch_p50_ms", "batch_p95_ms")


class Runs:
    """R runs aligned to the canonical item order: preds (R, N), scores (R, N, 7), speed rows."""

    def __init__(self, preds, scores, y, speed: pd.DataFrame | None = None):
        self.preds, self.scores, self.y = np.asarray(preds), np.asarray(scores), np.asarray(y)
        self.speed = speed if speed is not None else pd.DataFrame(columns=list(SPEED_COLS))

    def __len__(self) -> int:
        return self.preds.shape[1]

    def subset(self, mask) -> Runs:
        return Runs(self.preds[:, mask], self.scores[:, mask], self.y[mask], self.speed)

    def join(self, other: Runs) -> Runs:
        return Runs(
            np.concatenate([self.preds, other.preds]),
            np.concatenate([self.scores, other.scores]),
            self.y,
            pd.concat([self.speed, other.speed], ignore_index=True),
        )


# ------------------------------------------------------------------------------ metrics


def flipped(preds: np.ndarray) -> np.ndarray:
    """Per item: not every run predicts the same label. preds: (R, N)."""
    return (preds != preds[:1]).any(axis=0)


def flip_rate(preds: np.ndarray) -> float:
    return float(flipped(preds).mean())


def pairwise_disagreement(preds: np.ndarray) -> float:
    pairs = list(itertools.combinations(range(len(preds)), 2))
    if not pairs:
        return float("nan")
    return float(np.mean([(preds[a] != preds[b]).mean() for a, b in pairs]))


def _bits(scores: np.ndarray) -> np.ndarray:
    """Raw bytes of each score as unsigned ints (NaN payloads and -0.0 are distinguished)."""
    scores = np.ascontiguousarray(scores)
    return scores.view(np.dtype(f"u{scores.dtype.itemsize}"))


def bitwise_identical_share(scores: np.ndarray) -> float:
    """Share of items whose raw scores are byte-identical across all runs. scores: (R, N, K)."""
    bits = _bits(scores)
    return float((bits == bits[:1]).all(axis=(0, 2)).mean())


def max_abs_diff(scores: np.ndarray) -> float:
    s = scores.astype(np.float64)
    return float((s.max(axis=0) - s.min(axis=0)).max())


def accuracy_per_run(runs: Runs) -> np.ndarray:
    return (runs.preds == runs.y[None, :]).mean(axis=1)


def stratum_row(runs: Runs) -> dict:
    acc = accuracy_per_run(runs)
    return {
        "n_items": len(runs),
        "n_runs": len(runs.preds),
        "flip_rate": flip_rate(runs.preds),
        "pairwise_disagreement": pairwise_disagreement(runs.preds),
        "bitwise_identical": bitwise_identical_share(runs.scores),
        "max_abs_diff": max_abs_diff(runs.scores),
        "acc_min": float(acc.min()),
        "acc_max": float(acc.max()),
    }


def condition_rows(name: str, runs: Runs, strata: np.ndarray) -> list[dict]:
    """The four stratum rows of one condition. `strata`: per-item stratum label."""
    rows = {}
    for st in (NEAR, REST):
        rows[st] = stratum_row(runs.subset(strata == st))
    rows["all"] = stratum_row(runs)
    rw = {
        "n_items": rows["all"]["n_items"],
        "n_runs": rows["all"]["n_runs"],
        "max_abs_diff": rows["all"]["max_abs_diff"],
    }
    for key in ("flip_rate", "pairwise_disagreement", "bitwise_identical"):
        rw[key] = sum(w * rows[st][key] for st, w in WEIGHTS.items())
    acc = sum(w * accuracy_per_run(runs.subset(strata == st)) for st, w in WEIGHTS.items())
    rw["acc_min"], rw["acc_max"] = float(acc.min()), float(acc.max())
    rows["reweighted"] = rw
    speed = {
        c: float(runs.speed[c].mean()) if len(runs.speed) else float("nan") for c in SPEED_COLS
    }
    return [{"condition": name, "stratum": st, **rows[st], **speed} for st in STRATA]


def paired_rows(name: str, a: Runs, b: Runs, strata: np.ndarray) -> list[dict]:
    rows = {}
    for st, mask in (
        (NEAR, strata == NEAR),
        (REST, strata == REST),
        ("all", np.ones(len(strata), bool)),
    ):
        pa, pb = a.preds[:, mask], b.preds[:, mask]
        sa, sb = a.scores[:, mask], b.scores[:, mask]
        diff = pa != pb
        acc_a, acc_b = (pa == a.y[mask]).mean(axis=1), (pb == a.y[mask]).mean(axis=1)
        rows[st] = {
            "n_items": int(mask.sum()),
            "n_runs": len(pa),
            "flip_rate": float(diff.any(axis=0).mean()),
            "pairwise_disagreement": float(diff.mean()),
            "bitwise_identical": float((_bits(sa) == _bits(sb)).all(axis=(0, 2)).mean()),
            "max_abs_diff": float(np.abs(sa.astype(np.float64) - sb.astype(np.float64)).max()),
            "acc_min": float(min(acc_a.min(), acc_b.min())),
            "acc_max": float(max(acc_a.max(), acc_b.max())),
        }
    rw = {
        k: sum(w * rows[st][k] for st, w in WEIGHTS.items())
        for k in ("flip_rate", "pairwise_disagreement", "bitwise_identical")
    }
    rows["reweighted"] = {
        "n_items": rows["all"]["n_items"],
        "n_runs": rows["all"]["n_runs"],
        **rw,
        "max_abs_diff": rows["all"]["max_abs_diff"],
        "acc_min": rows["all"]["acc_min"],
        "acc_max": rows["all"]["acc_max"],
    }
    speed = pd.concat([a.speed, b.speed])
    sp = {c: float(speed[c].mean()) if len(speed) else float("nan") for c in SPEED_COLS}
    return [{"condition": name, "stratum": st, **rows[st], **sp} for st in STRATA]


# ------------------------------------------------------------------------------ loading


def load_condition(cond: str, ids: pd.DataFrame) -> Runs | None:
    files = sorted((RUNS_DIR / cond).glob("run_*.parquet"), key=lambda p: int(p.stem.split("_")[1]))
    if not files:
        return None
    kind = "probs" if cond.startswith("N") else "logits"
    preds, scores, speed = [], [], []
    for f in files:
        df = pd.read_parquet(f).set_index("item_id").loc[ids["item_id"]]
        preds.append(df["pred"].to_numpy())
        scores.append(df[[f"{kind}_{j}" for j in range(N_CLASSES)]].to_numpy())
        speed.append({c: df[c].iloc[0] if c in df else np.nan for c in SPEED_COLS})
    return Runs(preds, scores, ids["label"].to_numpy(), pd.DataFrame(speed))


def flipped_examples(
    all_runs: dict[str, Runs], ids: pd.DataFrame, margins: pd.Series, k: int = 5
) -> list[dict]:
    """k flipped items, one per condition first (the lowest-margin flipped item not yet used)."""
    chosen: list[dict] = []
    used: set[int] = set()
    for _ in range(k):
        for cond, runs in all_runs.items():
            if len(chosen) >= k:
                break
            idx = [i for i in np.flatnonzero(flipped(runs.preds)) if i not in used]
            idx.sort(key=lambda i: (margins.iloc[i], i))
            if idx:
                i = idx[0]
                used.add(int(i))
                chosen.append(
                    {
                        "condition": cond,
                        "item_id": ids["item_id"].iloc[i],
                        "true_label": int(ids["label"].iloc[i]),
                        "stratum": ids["stratum"].iloc[i],
                        "margin_top2": float(margins.iloc[i]),
                        "preds_per_run": runs.preds[:, i].tolist(),
                    }
                )
    return chosen


def run_image_ids() -> set[str]:
    """Image ids recorded in the run files (one per run, from DECIDER_IMAGE at run time)."""
    files = sorted(RUNS_DIR.glob("*/run_*.parquet"))
    return {str(pd.read_parquet(f, columns=["image_id"])["image_id"].iloc[0]) for f in files} - {""}


def main() -> None:
    setup_logging(load_settings().log_level)
    ids = pd.read_csv(SUBSET_CSV)
    strata = ids["stratum"].to_numpy()
    runs = {c: r for c in CONDITIONS if (r := load_condition(c, ids)) is not None}
    rows: list[dict] = []
    for c, r in runs.items():
        rows += condition_rows(c, r, strata)
    if "D1" in runs and "D2" in runs:
        rows += condition_rows("D4", runs["D1"].join(runs["D2"]), strata)
        n = min(len(runs["D1"].preds), len(runs["D2"].preds))
        a = Runs(runs["D1"].preds[:n], runs["D1"].scores[:n], ids["label"], runs["D1"].speed)
        b = Runs(runs["D2"].preds[:n], runs["D2"].scores[:n], ids["label"], runs["D2"].speed)
        rows += paired_rows("D4_paired", a, b, strata)
    if "N1" in runs and "N2" in runs:
        rows += condition_rows("N1+N2", runs["N1"].join(runs["N2"]), strata)
    for c in ("D5", "D6a", "D6b"):
        if c in runs and "D2" in runs:
            rows += condition_rows(f"{c}+D2", runs[c].join(runs["D2"]), strata)
    pd.DataFrame(rows).to_csv(STABILITY_DIR / "results.csv", index=False)
    log.info("wrote %s (%d rows)", STABILITY_DIR / "results.csv", len(rows))

    margins = (
        pd.read_parquet(
            RESULTS_DIR / "decider" / "test_v3_bs1.parquet", columns=["item_id", "margin_top2"]
        )
        .set_index("item_id")["margin_top2"]
        .loc[ids["item_id"]]
    )
    pool = {**runs}
    if "D1" in runs and "D2" in runs:
        pool["D4"] = runs["D1"].join(runs["D2"])
    order = [
        c for c in ("D4", "D6b", "D6a", "D5", "D3w", "D3", "D2", "D1", "N2", "N1") if c in pool
    ]
    ex = flipped_examples({c: pool[c] for c in order}, ids, margins)
    from scripts.decider_run import load_split

    text = load_split("test").set_index("item_id")["text"]
    for e in ex:
        e["text_first_120"] = text.loc[e["item_id"]][:120]
    (STABILITY_DIR / "flipped_examples.json").write_text(json.dumps(ex, indent=2) + "\n")

    versions = json.loads((RESULTS_DIR / "versions.json").read_text())
    versions["image_main_run"] = versions.pop("image", None)
    versions["image_stability_run"] = sorted(run_image_ids())
    versions["decider_inventory_commit"] = INVENTORY_COMMIT
    (STABILITY_DIR / "versions.json").write_text(
        json.dumps(versions, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
