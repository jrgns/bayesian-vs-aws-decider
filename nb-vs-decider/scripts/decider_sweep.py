"""Throughput batch-size sweep on a fixed stratified DEV subset. Never touches the test split.

    python -m scripts.decider_sweep

The best batch size is chosen on throughput only: the smallest size within 3% of the fastest.
Agreement with bs=1 and the max prob difference show what batching does to the predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from datetime import UTC, datetime

import numpy as np

from scripts.common import RESULTS_DIR, load_settings, setup_logging
from scripts.decider_run import (
    SEED,
    WARMUP_BATCHES,
    build_examples,
    check_slicing,
    energy_summary,
    infer,
    latency_stats,
    load_labels,
    load_model,
    load_split,
    measured,
    pool_examples,
    stratified_subset,
    warmup,
)
from scripts.energy import Meter

log = logging.getLogger(__name__)

BATCH_SIZES = (1, 2, 4, 8, 16, 32, 64, 128)
SUBSET_SIZE = 700  # 100 per class
TIE = 0.03


def choose_best(throughput: dict[int, float], tie: float = TIE) -> int:
    """Smallest batch size whose throughput is within `tie` of the fastest."""
    top = max(throughput.values())
    return min(bs for bs, t in throughput.items() if t >= (1 - tie) * top)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--labels", default="v1")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--idle-seconds", type=float, default=60)
    ap.add_argument("--no-energy", action="store_true")
    args = ap.parse_args(argv)

    s = load_settings()
    setup_logging(s.log_level)
    import torch

    run_id = args.run_id or f"sweep_{args.labels}_{datetime.now(UTC):%Y%m%dT%H%M%S}"
    cfg = load_labels(args.labels)
    df = stratified_subset(load_split("dev"), SUBSET_SIZE)
    examples = build_examples(df, cfg)
    pool = pool_examples(cfg, WARMUP_BATCHES * max(BATCH_SIZES))

    meter = None if args.no_energy else Meter(run_id)
    idle = meter.idle_baseline(args.idle_seconds) if meter else None
    model, ckpt = load_model(s)

    rows, ref = [], None
    for bs in BATCH_SIZES:
        row = {"run_id": run_id, "batch_size": bs, "n": len(examples)}
        try:
            torch.cuda.reset_peak_memory_stats()
            warmup(model, pool, [bs])
            slicing = check_slicing(model, pool[: max(32, 2 * bs)], bs)
            with measured(meter, f"bs{bs}", idle) as m:
                t0 = time.perf_counter()
                res = infer(model, examples, bs)
                torch.cuda.synchronize()
                wall_s = time.perf_counter() - t0
        except torch.cuda.OutOfMemoryError as exc:
            log.warning("bs %d: out of memory, stopping the sweep (%s)", bs, str(exc)[:120])
            rows.append({**row, "status": "oom"})
            torch.cuda.empty_cache()
            break
        if ref is None:
            ref = res
        row.update(
            status="ok",
            wall_s=wall_s,
            items_per_s=len(examples) / wall_s,
            **latency_stats(res),
            **energy_summary(m.record, len(examples)),
            agree_with_bs1=float(np.mean(res.pred == ref.pred)),
            max_abs_prob_diff_vs_bs1=float(np.abs(res.probs - ref.probs).max()),
            max_abs_logit_diff_vs_bs1=float(np.abs(res.logits - ref.logits).max()),
            slicing_bitwise_identical=slicing["bitwise_identical"],
            slicing_max_abs_logit_diff=slicing["max_abs_logit_diff"],
            peak_mem_MiB=torch.cuda.max_memory_allocated() / 2**20,
            energy_id=m.record.energy_id if m.record else None,
        )
        rows.append(row)
        log.info(
            "bs %3d: %.2f items/s, batch p50 %.0f ms, agree %.4f", bs, row["items_per_s"],
            row["batch_p50_ms"], row["agree_with_bs1"],
        )  # fmt: skip

    ok = [r for r in rows if r["status"] == "ok"]
    throughput = {r["batch_size"]: r["items_per_s"] for r in ok}
    best = choose_best(throughput)
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with (RESULTS_DIR / "decider_sweep.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    (RESULTS_DIR / "decider_best_bs.json").write_text(
        json.dumps(
            {
                "best_batch_size": best,
                "rule": f"smallest batch size within {TIE:.0%} of the highest items/s; "
                "throughput only, accuracy is not used",
                "throughput_items_per_s": throughput,
                "oom_at": next((r["batch_size"] for r in rows if r["status"] == "oom"), None),
                "subset": {"split": "dev", "n": len(examples), "per_class": SUBSET_SIZE // 7,
                           "seed": SEED, "labels": args.labels},
                "run_id": run_id,
                "model": {"id": ckpt.model_id, "revision": ckpt.model_revision},
            },
            indent=2,
        )
        + "\n"
    )  # fmt: skip
    log.info("best batch size by throughput: %d", best)


if __name__ == "__main__":
    main()
