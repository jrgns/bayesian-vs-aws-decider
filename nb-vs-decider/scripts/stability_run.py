"""One Decider stability run (one process, one parquet): the official `collect_logits` path.

    python -m scripts.stability_run --cond D2 --run 0

Conditions (docs/PLAN_STABILITY.md), all on stability/subset_ids.csv, test labels, batch pairs
taken consecutively in the run's item order:
    D1 bs=1 default | D2 bs=2 default | D3 bs=2 deterministic mode | D3w same with warn_only
    D5 option order shuffled per run | D6a pair positions swapped within each batch
    D6b whole subset shuffled before batching
Output: stability/runs/<COND>/run_<k>.parquet, rows in the run's processing order, logits and
probs in CANONICAL label order. Deterministic mode is applied before CUDA initialises; if it
raises (an op with no deterministic implementation) the op and error go to
stability/runs/<COND>/error_run_<k>.json and the exit code is 3.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.common import load_settings, setup_logging
from scripts.decider_run import (
    N_CLASSES,
    check_split_allowed,
    infer,
    latency_stats,
    load_labels,
    load_split,
    pool_examples,
    warmup,
)
from scripts.stability_subset import STABILITY_DIR, SUBSET_CSV

log = logging.getLogger(__name__)

LABELS_VERSION = "v3"
WARMUP_ITEMS = 8
CUBLAS_CFG = ":4096:8"
RUNS_DIR = STABILITY_DIR / "runs"


@dataclass(frozen=True)
class Cond:
    batch_size: int
    deterministic: bool = False
    warn_only: bool = False
    option_shuffle: bool = False
    item_order: str = "canonical"  # canonical | swap_within_batch | shuffle_all


CONDITIONS: dict[str, Cond] = {
    "D1": Cond(1),
    "D2": Cond(2),
    "D3": Cond(2, deterministic=True),
    "D3w": Cond(2, deterministic=True, warn_only=True),
    "D5": Cond(2, option_shuffle=True),
    "D6a": Cond(2, item_order="swap_within_batch"),
    "D6b": Cond(2, item_order="shuffle_all"),
}


# ------------------------------------------------------------------------------ pure helpers


def run_path(cond: str, run: int) -> Path:
    return RUNS_DIR / cond / f"run_{run}.parquet"


def item_order(mode: str, run: int, n: int, batch_size: int) -> np.ndarray:
    """Processing order as indices into the canonical item order."""
    order = np.arange(n)
    rng = np.random.default_rng(run)
    if mode == "canonical":
        return order
    if mode == "shuffle_all":
        return rng.permutation(n)
    if mode == "swap_within_batch":
        for b, flip in enumerate(rng.random(-(-n // batch_size)) < 0.5):
            lo, hi = b * batch_size, min((b + 1) * batch_size, n)
            if flip:
                order[lo:hi] = order[lo:hi][::-1]
        return order
    raise ValueError(f"unknown item order {mode!r}")


def option_perm(run: int, n_options: int = N_CLASSES) -> np.ndarray:
    """perm[j] = canonical label shown in slot j."""
    return np.random.default_rng(run).permutation(n_options)


def unpermute_columns(arr: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """Map a (n, k) array whose column j belongs to canonical label perm[j] back to canonical."""
    out = np.empty_like(arr)
    out[:, perm] = arr
    return out


def seed_used(cond: Cond, run: int) -> int:
    """The seed that drives this run's randomness; -1 when the run uses none."""
    if cond.deterministic:
        return 0
    return run if cond.option_shuffle or cond.item_order != "canonical" else -1


def deterministic_error(exc: BaseException) -> dict:
    msg = str(exc)
    m = re.search(r"([\w.:]+) does not have a deterministic implementation", msg)
    return {
        "op": m.group(1) if m else None,
        "error_type": type(exc).__name__,
        "message": msg.strip().splitlines()[0] if msg.strip() else "",
        "traceback": traceback.format_exc(),
    }


# ------------------------------------------------------------------------------ run


def apply_determinism(warn_only: bool) -> dict:
    """Before any CUDA init: workspace config env, deterministic algorithms, cudnn flags, seeds."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", CUBLAS_CFG)
    import torch

    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    return {"cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"]}


def build_examples(texts, labels, cfg: dict, perm: np.ndarray):
    """Examples with the options passed in `perm` order and the label remapped to its slot."""
    from strands_decider.data.format import Example

    lab = cfg["labels"]
    options = [[lab[i]["name"], lab[i]["description"]] for i in range(N_CLASSES)]
    inv = np.argsort(perm)
    shown = [list(options[p]) for p in perm]
    return [
        Example(
            kind="choice",
            state=t,
            instructions=cfg["instructions"],
            options=[list(o) for o in shown],
            label=int(inv[int(y)]),
        )
        for t, y in zip(texts, labels, strict=True)
    ]


def load_items() -> pd.DataFrame:
    ids = pd.read_csv(SUBSET_CSV)
    test = load_split("test")[["item_id", "text"]]
    out = ids.merge(test, on="item_id", how="left", validate="one_to_one")
    if out["text"].isna().any():
        raise RuntimeError("subset items missing from the test split")
    return out


def execute(args, cond: Cond, items: pd.DataFrame, cfg: dict) -> dict:
    s = load_settings()
    from scripts.decider_run import load_model

    info = apply_determinism(cond.warn_only) if cond.deterministic else {}
    n = len(items)
    order = item_order(cond.item_order, args.run, n, cond.batch_size)
    perm = option_perm(args.run) if cond.option_shuffle else np.arange(N_CLASSES)
    proc = items.iloc[order].reset_index(drop=True)
    examples = build_examples(proc["text"], proc["label"], cfg, perm)

    model, ckpt = load_model(s)
    import torch

    warmup(model, pool_examples(cfg, WARMUP_ITEMS), [cond.batch_size])
    t0 = time.perf_counter()
    res = infer(model, examples, cond.batch_size)
    torch.cuda.synchronize()
    wall_s = time.perf_counter() - t0
    probs = unpermute_columns(res.probs, perm)
    logits = unpermute_columns(res.logits, perm)
    out = pd.DataFrame(
        {
            "item_id": proc["item_id"],
            "label": proc["label"].astype("int64"),
            "pred": probs.argmax(axis=1).astype("int64"),
        }
    )
    for j in range(N_CLASSES):
        out[f"probs_{j}"] = probs[:, j].astype(np.float32)
    for j in range(N_CLASSES):
        out[f"logits_{j}"] = logits[:, j].astype(np.float32)
    out["batch_index"] = res.batch_index.astype("int32")
    out["position_in_batch"] = res.position_in_batch.astype("int32")
    meta = {
        "cond": args.cond,
        "run": args.run,
        "batch_size": cond.batch_size,
        "deterministic": cond.deterministic,
        "warn_only": cond.warn_only,
        "option_shuffle": cond.option_shuffle,
        "item_order": cond.item_order,
        "seed": seed_used(cond, args.run),
        "wall_s": wall_s,
        "items_per_s": n / wall_s,
        **latency_stats(res),
        "model_revision": ckpt.model_revision,
        "image_id": os.environ.get("DECIDER_IMAGE", ""),
        "cublas_workspace_config": info.get("cublas_workspace_config", ""),
    }
    for k, v in meta.items():
        out[k] = v
    return {"frame": out, "acc": float((out["pred"] == out["label"]).mean()), "wall_s": wall_s}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cond", choices=sorted(CONDITIONS), required=True)
    ap.add_argument("--run", type=int, required=True)
    args = ap.parse_args(argv)
    check_split_allowed("test")
    setup_logging(load_settings().log_level)
    path = run_path(args.cond, args.run)
    if path.exists():
        log.info("skip %s: exists", path)
        return
    cond = CONDITIONS[args.cond]
    cfg = load_labels(LABELS_VERSION)
    items = load_items()
    log.info(
        "run %s/%d: %s, %d items, labels %s", args.cond, args.run, cond, len(items), LABELS_VERSION
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        done = execute(args, cond, items, cfg)
    except Exception as exc:
        if not (cond.deterministic and not cond.warn_only):
            raise
        err = {"cond": args.cond, "run": args.run, **deterministic_error(exc)}
        (path.parent / f"error_run_{args.run}.json").write_text(json.dumps(err, indent=2) + "\n")
        log.error(
            "%s run %d failed in deterministic mode: op=%s %s",
            args.cond,
            args.run,
            err["op"],
            err["message"],
        )
        sys.exit(3)
    done["frame"].to_parquet(path, index=False)
    log.info(
        "wrote %s: acc %.4f, %.1fs, %.2f items/s",
        path,
        done["acc"],
        done["wall_s"],
        len(items) / done["wall_s"],
    )


if __name__ == "__main__":
    main()
