"""Evidence for the causes of Decider flips, from the saved runs: stability/causes.json.

    python -m scripts.stability_causes

Token length of an item = its text length in tokens (the prompt around it is constant), so the
padded length of a batch is the longest text in it. Questions answered:
  D1 vs D2   are logits byte-identical for the longer item of a D2 pair (never padded) and for
             the shorter one (padded to the pair's length)?
  D6b vs D2  are logits byte-identical when the item's batch has the same padded length as its
             canonical D2 batch, and when it differs?
  D6a / D3   byte-identical to D2?
  D5         per-run accuracy and the share of predictions on each canonical label.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from scripts.common import load_settings, prepare_checkpoint, setup_logging
from scripts.stability_metrics import RUNS_DIR, Runs, _bits, flipped, load_condition
from scripts.stability_subset import STABILITY_DIR, SUBSET_CSV

log = logging.getLogger(__name__)


def token_lengths(texts: list[str]) -> np.ndarray:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(prepare_checkpoint(load_settings()).path)
    return np.array([len(x) for x in tok(texts, add_special_tokens=False)["input_ids"]])


def same_bytes(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per item: logits of two runs (N, K) are byte-identical."""
    return (_bits(a) == _bits(b)).all(axis=1)


def share(mask: np.ndarray, sel: np.ndarray) -> dict:
    return {"n": int(sel.sum()), "byte_identical": float(mask[sel].mean()) if sel.any() else None}


def by_item(cond: str, run: int, ids: pd.DataFrame) -> pd.DataFrame:
    df = (
        pd.read_parquet(RUNS_DIR / cond / f"run_{run}.parquet")
        .set_index("item_id")
        .loc[ids["item_id"]]
    )
    return df


def main() -> None:
    setup_logging(load_settings().log_level)
    ids = pd.read_csv(SUBSET_CSV)
    from scripts.decider_run import load_split

    text = load_split("test").set_index("item_id").loc[ids["item_id"], "text"].tolist()
    length = token_lengths(text)
    n = len(ids)
    cols = [f"logits_{j}" for j in range(7)]
    out: dict = {"token_length": {"p50": float(np.median(length)), "max": int(length.max())}}

    # D1 vs D2: canonical pairs (2b, 2b+1)
    d1, d2 = by_item("D1", 0, ids)[cols].to_numpy(), by_item("D2", 0, ids)[cols].to_numpy()
    eq = same_bytes(d1, d2)
    partner = np.arange(n) ^ 1
    longer, shorter = length > length[partner], length < length[partner]
    out["D1_vs_D2"] = {
        "all": share(eq, np.ones(n, bool)),
        "longer_in_pair_unpadded": share(eq, longer),
        "shorter_in_pair_padded": share(eq, shorter),
        "equal_length": share(eq, length == length[partner]),
    }

    # D6b vs D2, every run: padded length of the item's batch vs its canonical batch
    d2_pad = np.maximum(length, length[partner])
    same, diff = [], []
    for k in range(5):
        run = by_item("D6b", k, ids)
        order = pd.read_parquet(RUNS_DIR / "D6b" / f"run_{k}.parquet")["item_id"].to_numpy()
        pos = pd.Series(np.arange(n), index=ids["item_id"]).loc[order].to_numpy()
        pad = np.empty(n, int)
        pad[pos] = np.repeat(np.maximum.reduceat(length[pos], np.arange(0, n, 2)), 2)
        eq6 = same_bytes(run[cols].to_numpy(), d2)
        same.append((eq6[pad == d2_pad], (pad == d2_pad).sum()))
        diff.append((eq6[pad != d2_pad], (pad != d2_pad).sum()))
    out["D6b_vs_D2"] = {
        "same_padded_length": {
            "n_item_runs": int(sum(c for _, c in same)),
            "byte_identical": float(np.concatenate([m for m, _ in same]).mean()),
        },
        "different_padded_length": {
            "n_item_runs": int(sum(c for _, c in diff)),
            "byte_identical": float(np.concatenate([m for m, _ in diff]).mean()),
        },
    }

    # D6a, D3 vs D2, and D3 reproducibility
    for cond in ("D6a", "D3"):
        runs = load_condition(cond, ids)
        base = load_condition("D2", ids)
        pooled = Runs(
            np.concatenate([runs.preds, base.preds]),
            np.concatenate([runs.scores, base.scores]),
            ids["label"],
        )
        out[f"{cond}_vs_D2"] = {
            "byte_identical_all_runs": float(
                (_bits(pooled.scores) == _bits(pooled.scores)[:1]).all(axis=(0, 2)).mean()
            ),
            "flips": int(flipped(pooled.preds).sum()),
        }

    # D5
    d5 = load_condition("D5", ids)
    out["D5"] = {
        "accuracy_per_run": (d5.preds == d5.y).mean(axis=1).tolist(),
        "pred_label_share_per_run": [(np.bincount(p, minlength=7) / n).tolist() for p in d5.preds],
        "true_label_share": (np.bincount(d5.y, minlength=7) / n).tolist(),
    }
    (STABILITY_DIR / "causes.json").write_text(json.dumps(out, indent=2) + "\n")
    log.info("wrote causes.json:\n%s", json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
