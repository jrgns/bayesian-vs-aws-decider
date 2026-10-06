"""NB stability runs: the seed-0 best config at the crossover n, on the stability subset.

    python -m scripts.stability_nb --cond N1     # train once, predict 10x in this process
    python -m scripts.stability_nb --cond N2 --run 3   # retrain + predict, one process per run

The config is read from results/nb_best_configs.json (no re-tuning) and trained on the saved
subset results/nb/subsets/n<N>_s0.csv. Threads are pinned to 1 as in nb_curve. Output parquets
use the Decider schema, except that the arrays are float64 exactly as sklearn returns them
(`probs_*` = predict_proba, `logits_*` = predict_joint_log_proba), so byte comparisons are
not weakened by a cast. Rows keep the canonical subset order.
"""

from __future__ import annotations

import argparse
import logging
import os
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_info, threadpool_limits

from scripts.common import RESULTS_DIR, load_settings, setup_logging
from scripts.decider_run import N_CLASSES, check_split_allowed, load_split
from scripts.nb_curve import SUBSETS_DIR, make_pipeline
from scripts.stability_run import RUNS_DIR
from scripts.stability_subset import SUBSET_CSV

log = logging.getLogger(__name__)

N_PER_CLASS = 100
SEED = 0
N1_RUNS = 10


def load_cfg(n: int = N_PER_CLASS, seed: int = SEED) -> dict:
    import json

    return json.loads((RESULTS_DIR / "nb_best_configs.json").read_text())[f"n{n}_s{seed}"]


def train(cfg: dict, n: int = N_PER_CLASS, seed: int = SEED):
    pool = load_split("pool")
    ids = pd.read_csv(SUBSETS_DIR / f"n{n}_s{seed}.csv")
    sub = ids.merge(pool, on="item_id", how="left", validate="one_to_one")
    if sub["text"].isna().any() or len(sub) != N_CLASSES * n:
        raise RuntimeError("training subset does not match the pool")
    model = make_pipeline(cfg["variant"], cfg["alpha"], cfg["min_df"])
    model.fit(sub["text"].tolist(), sub["label"].to_numpy())
    return model


def predict_frame(model, items: pd.DataFrame, meta: dict) -> pd.DataFrame:
    t0 = time.perf_counter()
    probs = model.predict_proba(items["text"])
    wall_s = time.perf_counter() - t0
    x = model.named_steps["vec"].transform(items["text"])
    jll = model.named_steps["nb"].predict_joint_log_proba(x)
    out = pd.DataFrame(
        {
            "item_id": items["item_id"],
            "label": items["label"].astype("int64"),
            "pred": probs.argmax(axis=1).astype("int64"),
        }
    )
    for j in range(N_CLASSES):
        out[f"probs_{j}"] = probs[:, j]
    for j in range(N_CLASSES):
        out[f"logits_{j}"] = jll[:, j]
    out["batch_index"] = np.int32(0)
    out["position_in_batch"] = np.arange(len(out), dtype="int32")
    for k, v in {**meta, "wall_s": wall_s, "items_per_s": len(out) / wall_s}.items():
        out[k] = v
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cond", choices=("N1", "N2"), required=True)
    ap.add_argument("--run", type=int, default=0, help="N2 run index")
    args = ap.parse_args(argv)
    check_split_allowed("test")
    setup_logging(load_settings().log_level)
    runs = range(N1_RUNS) if args.cond == "N1" else [args.run]
    todo = [k for k in runs if not (RUNS_DIR / args.cond / f"run_{k}.parquet").exists()]
    if not todo:
        log.info("skip %s: all outputs exist", args.cond)
        return
    cfg = load_cfg()
    items = pd.read_csv(SUBSET_CSV).merge(
        load_split("test")[["item_id", "text"]], on="item_id", how="left", validate="one_to_one"
    )
    with threadpool_limits(limits=1):
        info = [
            {k: p.get(k) for k in ("user_api", "internal_api", "num_threads", "version")}
            for p in threadpool_info()
        ]
        model = train(cfg)
        meta_base = {
            "cond": args.cond,
            "batch_size": len(items),
            "deterministic": False,
            "warn_only": False,
            "option_shuffle": False,
            "item_order": "canonical",
            "seed": SEED,
            "pythonhashseed": os.environ.get("PYTHONHASHSEED", "random"),
            "threadpool": str(info),
            "config": str(cfg),
            "image_id": os.environ.get("DECIDER_IMAGE", ""),
        }
        for k in todo:
            frame = predict_frame(model, items, {**meta_base, "run": k})
            path = RUNS_DIR / args.cond / f"run_{k}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(path, index=False)
            log.info("wrote %s: acc %.4f", path, float((frame["pred"] == frame["label"]).mean()))


if __name__ == "__main__":
    main()
