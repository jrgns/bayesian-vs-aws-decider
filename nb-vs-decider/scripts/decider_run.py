"""Zero-shot Decider on one split via the official batched evaluation path.

    python -m scripts.decider_run --split dev --labels v1 --batch-size 1 --measure-energy

One code path for every batch size: the examples are sliced into batches of N and each slice goes
through `collect_logits(batch_size=N)` + `predictions_from_logits` (temperature and max_length from
the model config), timed with CUDA synchronisation around it.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from scripts import metrics
from scripts.common import (
    APP_DIR,
    DATA_DIR,
    RESULTS_DIR,
    Settings,
    append_csv,
    load_settings,
    prepare_checkpoint,
    setup_logging,
)
from scripts.energy import EnergyRecord, IdleBaseline, Measurement, Meter

log = logging.getLogger(__name__)

SEED = 20261006  # subsets, warmup sampling and bootstrap; the splits themselves use 20261005
N_CLASSES = 7
WARMUP_BATCHES = 3
WARMUP_BATCH_SIZE = 32
SPLITS_DIR = DATA_DIR / "splits"
FINAL_PARQUET = DATA_DIR / "processed" / "wos46985_l1.parquet"
RUNS_CSV = RESULTS_DIR / "decider_runs.csv"
RUNS_DIR = RESULTS_DIR / "decider"


# ------------------------------------------------------------------------------ data


def check_split_allowed(split: str) -> None:
    if split == "test" and os.environ.get("ALLOW_TEST") != "1":
        sys.exit("refusing to run Decider on the test split: ALLOW_TEST=1 is not set")


def load_labels(version: str) -> dict:
    cfg = yaml.safe_load((APP_DIR / "config" / "labels" / f"{version}.yaml").read_text())
    if sorted(cfg["labels"]) != list(range(N_CLASSES)):
        raise ValueError(f"label config {version} must define ids 0..{N_CLASSES - 1}")
    return cfg


def load_split(split: str) -> pd.DataFrame:
    """item_id, text, label for one split, in the frozen split-file order.

    Only the requested split's rows are read from the parquet, so other splits' texts
    (notably test) are never loaded.
    """
    ids = pd.read_csv(SPLITS_DIR / f"{split}.csv")
    texts = pd.read_parquet(
        FINAL_PARQUET, columns=["item_id", "text"], filters=[("item_id", "in", ids["item_id"])]
    )
    out = ids.merge(texts, on="item_id", how="left", validate="one_to_one")
    if out["text"].isna().any():
        raise RuntimeError(f"{split}: texts missing for {int(out['text'].isna().sum())} items")
    return out[["item_id", "text", "label"]]


def stratified_subset(df: pd.DataFrame, n: int, seed: int = SEED) -> pd.DataFrame:
    """n items, as equal per class as possible (remainder to the lowest labels), seeded.

    Keeps the frozen split order, so the subset is a stable function of (split, n, seed).
    """
    classes = sorted(df["label"].unique())
    base, extra = divmod(n, len(classes))
    rng = np.random.default_rng(seed)
    keep: list[int] = []
    for i, c in enumerate(classes):
        pos = np.flatnonzero(df["label"].to_numpy() == c)
        want = base + (1 if i < extra else 0)
        if want > len(pos):
            raise ValueError(f"class {c}: asked for {want}, only {len(pos)} available")
        keep.extend(rng.choice(pos, size=want, replace=False).tolist())
    return df.iloc[sorted(keep)].reset_index(drop=True)


def build_examples(df: pd.DataFrame, cfg: dict) -> list:
    from strands_decider.data.format import Example

    labels = cfg["labels"]
    options = [[labels[i]["name"], labels[i]["description"]] for i in range(N_CLASSES)]
    return [
        Example(
            kind="choice",
            state=text,
            instructions=cfg["instructions"],
            options=[list(o) for o in options],
            label=int(label),
        )
        for text, label in zip(df["text"], df["label"], strict=True)
    ]


def pool_examples(cfg: dict, n: int) -> list:
    """Warmup items: a seeded sample from the POOL split, never from an evaluated split."""
    df = load_split("pool")
    return build_examples(df.sample(n=n, random_state=SEED).reset_index(drop=True), cfg)


# ------------------------------------------------------------------------------ model


def load_model(s: Settings):
    from strands_decider.modeling import StrandsDeciderModel

    ckpt = prepare_checkpoint(s)
    model = StrandsDeciderModel.load(ckpt.path)
    model.eval().to("cuda")
    import torch

    torch.cuda.synchronize()
    return model, ckpt


@contextmanager
def measured(meter: Meter | None, label: str, idle: IdleBaseline | None) -> Iterator[Measurement]:
    """meter.measure(...) when a meter is given, otherwise a no-op that yields an empty slot."""
    if meter is None:
        yield Measurement()
    else:
        with meter.measure(label, idle) as m:
            yield m


# ------------------------------------------------------------------------------ inference


@dataclass
class InferResult:
    logits: np.ndarray  # (n, 7) float32, untempered, as returned by collect_logits
    probs: np.ndarray  # (n, 7) float32, after the model's per-kind temperature
    batch_s: list[float]  # latency of each collect_logits call
    batch_n: list[int]  # items in each batch

    @property
    def pred(self) -> np.ndarray:
        return self.probs.argmax(axis=1)

    @property
    def batch_index(self) -> np.ndarray:
        return np.repeat(np.arange(len(self.batch_n)), self.batch_n)

    @property
    def position_in_batch(self) -> np.ndarray:
        return np.concatenate([np.arange(n) for n in self.batch_n])


def infer(model, examples: list, batch_size: int) -> InferResult:
    """Slice into batches of `batch_size`; run each slice through the official eval path."""
    import torch
    from strands_decider.evaluate import collect_logits, predictions_from_logits

    temps, max_len = model.config.temperature_by_kind, model.config.max_length
    logits, probs, batch_s, batch_n = [], [], [], []
    for start in range(0, len(examples), batch_size):
        chunk = examples[start : start + batch_size]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        lg, labels, n_slots, exs = collect_logits(
            model, chunk, device="cuda", batch_size=batch_size, max_length=max_len
        )
        torch.cuda.synchronize()
        batch_s.append(time.perf_counter() - t0)
        batch_n.append(len(chunk))
        preds = predictions_from_logits(lg, labels, n_slots, exs, temperature=temps)
        if lg.shape[1] != N_CLASSES:
            raise RuntimeError(f"expected {N_CLASSES} logit columns, got {lg.shape[1]}")
        logits.append(lg.numpy())
        probs.append(np.array([p.probs for p in preds], dtype=np.float32))
    return InferResult(
        np.concatenate(logits).astype(np.float32), np.concatenate(probs), batch_s, batch_n
    )


def warmup(model, pool: list, sizes: list[int]) -> None:
    """WARMUP_BATCHES batches of each size in `sizes` (taken from `pool`), results discarded."""
    for size in sizes:
        infer(model, pool[: WARMUP_BATCHES * size], size)


def check_slicing(model, sample: list, batch_size: int) -> dict:
    """Per-slice calls must give the same logits as one big call at the same batch size."""
    import torch
    from strands_decider.evaluate import collect_logits

    big, *_ = collect_logits(
        model, sample, device="cuda", batch_size=batch_size, max_length=model.config.max_length
    )
    sliced = infer(model, sample, batch_size).logits
    big = big.numpy()
    torch.cuda.synchronize()
    return {
        "n": len(sample),
        "batch_size": batch_size,
        "bitwise_identical": bool(np.array_equal(big, sliced)),
        "max_abs_logit_diff": float(np.abs(big - sliced).max()),
    }


def latency_stats(res: InferResult) -> dict:
    batch = np.array(res.batch_s)
    item = batch / np.array(res.batch_n)
    q = lambda a, p: float(np.percentile(a, p) * 1000)  # noqa: E731
    return {
        "batch_p50_ms": q(batch, 50),
        "batch_p95_ms": q(batch, 95),
        "item_p50_ms": q(item, 50),
        "item_p95_ms": q(item, 95),
    }


def energy_summary(rec: EnergyRecord | None, n: int) -> dict:
    """Gross and marginal energy of the inference window, totals and per item."""
    out = {}
    for key in ("gpu_J", "cpu_pkg_J", "gpu_marginal_J", "cpu_pkg_marginal_J"):
        val = None if rec is None else getattr(rec, key)
        out[key] = val
        out[key.replace("_J", "_J_per_item")] = None if val is None else val / n
    return out


# ------------------------------------------------------------------------------ outputs


def write_predictions(run_id: str, split: str, df: pd.DataFrame, res: InferResult) -> Path:
    out = pd.DataFrame(
        {"item_id": df["item_id"], "split": split, "label": df["label"].astype("int64")}
    )
    out["pred"] = res.pred.astype("int64")
    for j in range(N_CLASSES):
        out[f"probs_{j}"] = res.probs[:, j]
    for j in range(N_CLASSES):
        out[f"logits_{j}"] = res.logits[:, j]
    out["margin_top2"] = metrics.top2_margin(res.probs).astype(np.float32)
    out["batch_index"] = res.batch_index.astype("int32")
    out["position_in_batch"] = res.position_in_batch.astype("int32")
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = RUNS_DIR / f"{run_id}.parquet"
    out.to_parquet(path, index=False)
    log.info("wrote %s", path)
    return path


def summary_row(args, run_id, n, stats, res, wall_s, recs, rec_infer) -> dict:
    return {
        "run_id": run_id,
        "split": args.split,
        "labels_version": args.labels,
        "batch_size": args.batch_size,
        "limit": args.limit,
        "n": n,
        **{
            k: stats[k]
            for k in (
                "accuracy",
                "accuracy_ci_lo",
                "accuracy_ci_hi",
                "macro_f1",
                "macro_f1_ci_lo",
                "macro_f1_ci_hi",
                "brier",
                "ece",
            )
        },  # fmt: skip
        "wall_s": wall_s,
        "items_per_s": n / wall_s,
        **latency_stats(res),
        **energy_summary(rec_infer, n),
        "energy_ids": ";".join(r.energy_id for r in recs),
    }


# ------------------------------------------------------------------------------ main


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", choices=("dev", "test"), required=True)
    ap.add_argument("--labels", default="v1")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="stratified subset size")
    ap.add_argument("--measure-energy", action="store_true")
    ap.add_argument("--idle-seconds", type=float, default=60)
    ap.add_argument("--run-id", default=None)
    return ap.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    check_split_allowed(args.split)
    s = load_settings()
    setup_logging(s.log_level)
    run_id = args.run_id or (
        f"{args.split}_{args.labels}_bs{args.batch_size}_{datetime.now(UTC):%Y%m%dT%H%M%S}"
    )
    cfg = load_labels(args.labels)
    df = load_split(args.split)
    if args.limit:
        df = stratified_subset(df, args.limit)
    log.info("run %s: %s items, labels %s, bs %d", run_id, len(df), args.labels, args.batch_size)

    meter = Meter(run_id) if args.measure_energy else None
    recs: list[EnergyRecord] = []
    idle = None
    if meter:
        idle = meter.idle_baseline(args.idle_seconds)
        recs.append(idle.record)

    with measured(meter, "load", idle) as m_load:
        model, ckpt = load_model(s)
    recs += [m_load.record] if m_load.record else []

    sizes = sorted({WARMUP_BATCH_SIZE, args.batch_size})
    pool = pool_examples(cfg, WARMUP_BATCHES * max(sizes))  # >= 2*bs and >= 32 slicing items
    with measured(meter, "warmup", idle) as m_warm:
        warmup(model, pool, sizes)
    recs += [m_warm.record] if m_warm.record else []
    slicing = check_slicing(model, pool[: max(32, 2 * args.batch_size)], args.batch_size)
    log.info("per-slice vs single-call logits: %s", slicing)

    examples = build_examples(df, cfg)
    import torch

    with measured(meter, "inference", idle) as m_inf:
        t0 = time.perf_counter()
        res = infer(model, examples, args.batch_size)
        torch.cuda.synchronize()
        wall_s = time.perf_counter() - t0
    recs += [m_inf.record] if m_inf.record else []

    y = df["label"].to_numpy()
    stats = metrics.summarise(y, res.pred, res.probs, N_CLASSES, seed=SEED)
    row = summary_row(args, run_id, len(df), stats, res, wall_s, recs, m_inf.record)
    write_predictions(run_id, args.split, df, res)
    cm = metrics.confusion_matrix(y, res.pred, N_CLASSES)
    meta = {
        "run_id": run_id,
        "summary": row,
        "confusion_matrix": cm.tolist(),
        "per_class_recall": metrics.per_class_recall(y, res.pred, N_CLASSES).tolist(),
        "class_names": [cfg["labels"][i]["name"] for i in range(N_CLASSES)],
        "slicing_check": slicing,
        "model": {"id": ckpt.model_id, "revision": ckpt.model_revision},
        "load_s": m_load.record.duration_s if m_load.record else None,
        "energy": {r.label: r.to_row() for r in recs},
    }
    (RUNS_DIR / f"{run_id}.json").write_text(json.dumps(meta, indent=2, default=float) + "\n")
    append_csv(RUNS_CSV, row)
    log.info(
        "%s: acc %.4f [%.4f, %.4f] macro-F1 %.4f brier %.4f ece %.4f, %.2f items/s",
        run_id,
        stats["accuracy"],
        stats["accuracy_ci_lo"],
        stats["accuracy_ci_hi"],
        stats["macro_f1"],
        stats["brier"],
        stats["ece"],
        row["items_per_s"],
    )


if __name__ == "__main__":
    main()
