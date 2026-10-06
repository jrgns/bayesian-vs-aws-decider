"""Energy summary: kWh per 1,000 predictions, load/warmup, NB training, wall-clock, break-even.

    python -m scripts.energy_summary

Reads decider_runs.csv, decider/<run>.json (load + warmup), nb_runs.csv, energy.csv,
nb_latency.json and crossover.json; writes results/energy_summary.csv (long format).
Decider energy = GPU (NVML) + CPU package (RAPL) over the inference window. NB energy = CPU
package only (NB needs no GPU; GPU idle draw is recorded but not attributed).
kWh = J / 3.6e6. "marginal" = gross minus idle power x duration.
"""

from __future__ import annotations

import json
import logging
import math

import pandas as pd

from scripts.common import RESULTS_DIR, decider_test_runs, load_settings, setup_logging
from scripts.nb_curve import FULL, N_PER_CLASS
from scripts.summary_stats import mean_ci

log = logging.getLogger(__name__)

J_PER_KWH = 3.6e6
MODES = ("gross", "marginal")
TRAIN_KINDS = {"tune_fit": ("tune", "fit"), "fit_only": ("fit",)}
BREAK_EVEN_N = (None, 1000)  # None = the crossover n
COLUMNS = ["table", "subject", "metric", "mode", "value", "ci_lo", "ci_hi", "unit", "note"]


def kwh(joules: float) -> float:
    return joules / J_PER_KWH


def break_even(train_j: float, decider_j_per_pred: float, nb_j_per_pred: float) -> float:
    """N* = NB_train / (Decider_infer - NB_infer) predictions; inf if NB is not cheaper."""
    saving = decider_j_per_pred - nb_j_per_pred
    return train_j / saving if saving > 0 else math.inf


class Table:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, table, subject, metric, value, unit, mode="", ci=(None, None), note="") -> None:
        self.rows.append(dict(zip(COLUMNS, [table, subject, metric, mode, value, *ci, unit, note],
                                  strict=True)))  # fmt: skip

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=COLUMNS)


# ------------------------------------------------------------------------------ inputs


class Inputs:
    def __init__(self) -> None:
        r = RESULTS_DIR
        self.decider = pd.read_csv(r / "decider_runs.csv").set_index("run_id")
        self.runs = {f"decider_bs{bs}": run for bs, run in decider_test_runs().items()}
        self.decider_json = {
            k: json.loads((r / "decider" / f"{run}.json").read_text())
            for k, run in self.runs.items()
        }
        runs = pd.read_csv(r / "nb_runs.csv")
        self.nb = runs[runs["is_best"]].copy()
        self.nb["n_per_class"] = self.nb["n_per_class"].astype(str)
        energy = pd.read_csv(r / "energy.csv").set_index("energy_id")
        self.energy_window_s = energy["duration_s"]
        self.latency = json.loads((r / "nb_latency.json").read_text())
        cross = json.loads((r / "crossover.json").read_text())
        self.crossover_n = int(cross["headline"]["crossover_n_per_class"])
        self.n_test = json.loads((r / "dataset_summary.json").read_text())["n_items"]["test"]

    def nb_cell(self, n) -> pd.DataFrame:
        return self.nb[self.nb["n_per_class"] == str(n)].sort_values("seed")


# ------------------------------------------------------------------------------ pieces


def decider_per_pred_j(inp: Inputs, key: str) -> dict[str, dict[str, float]]:
    row = inp.decider.loc[inp.runs[key]]
    return {
        "gross": {"gpu": row["gpu_J_per_item"], "cpu": row["cpu_pkg_J_per_item"]},
        "marginal": {
            "gpu": row["gpu_marginal_J_per_item"],
            "cpu": row["cpu_pkg_marginal_J_per_item"],
        },
    }


def nb_infer_j_per_pred(inp: Inputs, n) -> dict[str, list[float]]:
    cell = inp.nb_cell(n)
    return {
        "gross": (cell["infer_cpu_J"] / inp.n_test).tolist(),
        "marginal": (cell["infer_cpu_marginal_J"] / inp.n_test).tolist(),
    }


def nb_train_j(inp: Inputs, n, kind: str, mode: str) -> list[float]:
    """Per-seed training energy (J). tune_fit = tune + fit phases; fit_only = the refit alone."""
    cell = inp.nb_cell(n)
    suffix = "cpu_J" if mode == "gross" else "cpu_marginal_J"
    return sum(cell[f"{p}_{suffix}"] for p in TRAIN_KINDS[kind]).tolist()


def add_inference(t: Table, inp: Inputs) -> None:
    for key in inp.runs:
        per = decider_per_pred_j(inp, key)
        for mode in MODES:
            total = sum(per[mode].values())
            t.add("inference", key, "kwh_per_1000_predictions", kwh(total * 1000), "kWh", mode)
            for part in ("gpu", "cpu"):
                t.add("inference", key, f"kwh_per_1000_predictions_{part}",
                      kwh(per[mode][part] * 1000), "kWh", mode)  # fmt: skip
    for n in (inp.crossover_n, 1000, FULL):
        for mode, vals in nb_infer_j_per_pred(inp, n).items():
            mean, lo, hi = mean_ci([kwh(v * 1000) for v in vals])
            t.add("inference", f"nb_n{n}", "kwh_per_1000_predictions", mean, "kWh", mode, (lo, hi),
                  "CPU package only; mean and 95% t-CI over 5 seeds")  # fmt: skip


def add_decider_load(t: Table, inp: Inputs) -> None:
    for key, meta in inp.decider_json.items():
        for phase in ("load", "warmup"):
            e = meta["energy"][phase]
            t.add("decider_overhead", key, f"{phase}_duration", e["duration_s"], "s")
            for mode, g, c in (
                ("gross", e["gpu_J"], e["cpu_pkg_J"]),
                ("marginal", e["gpu_marginal_J"], e["cpu_pkg_marginal_J"]),
            ):
                t.add("decider_overhead", key, f"{phase}_energy_kwh", kwh(g + c), "kWh", mode,
                      note="GPU + CPU package; not in the per-prediction figures")  # fmt: skip


def add_nb_training(t: Table, inp: Inputs) -> None:
    for n in (*N_PER_CLASS, FULL):
        for kind in TRAIN_KINDS:
            for mode in MODES:
                mean, lo, hi = mean_ci([kwh(j) for j in nb_train_j(inp, n, kind, mode)])
                t.add("nb_training", f"nb_n{n}", f"train_kwh_{kind}", mean, "kWh", mode, (lo, hi),
                      "CPU package only; mean and 95% t-CI over 5 seeds")  # fmt: skip
        cell = inp.nb_cell(n)
        for phase in ("tune", "fit"):
            secs = [
                inp.energy_window_s[f"nb_n{n}_s{s}:{phase}"] / reps
                for s, reps in zip(cell["seed"], cell[f"{phase}_reps"], strict=True)
            ]
            mean, lo, hi = mean_ci(secs)
            t.add("nb_training", f"nb_n{n}", f"train_wall_s_{phase}", mean, "s", "", (lo, hi),
                  "metering window / repeats")  # fmt: skip


def add_timing(t: Table, inp: Inputs) -> None:
    for key, run in inp.runs.items():
        row = inp.decider.loc[run]
        t.add("timing", key, "wall_clock_test_set", row["wall_s"], "s")
        t.add("timing", key, "latency_p50_per_item", row["item_p50_ms"], "ms")
        t.add("timing", key, "latency_p95_per_item", row["item_p95_ms"], "ms")
    n = inp.crossover_n
    lat = inp.latency
    batch = inp.nb_cell(n)["predict_s"]
    t.add("timing", f"nb_n{n}", "wall_clock_test_set", batch.mean(), "s", "", mean_ci(batch)[1:],
          "vectorise + predict_proba of the whole test set, 1 thread")  # fmt: skip
    t.add("timing", f"nb_n{n}", "batch_per_item_latency", batch.mean() / inp.n_test * 1000, "ms")
    t.add("timing", f"nb_n{n}", "latency_p50_per_item", lat["single_item"]["p50_ms"], "ms", "",
          note="one item at a time, first 1000 test items, seed 0")  # fmt: skip
    t.add("timing", f"nb_n{n}", "latency_p95_per_item", lat["single_item"]["p95_ms"], "ms")


def add_break_even(t: Table, inp: Inputs) -> None:
    for dkey in inp.runs:
        dec = decider_per_pred_j(inp, dkey)
        for n in (inp.crossover_n if x is None else x for x in BREAK_EVEN_N):
            nb_inf = nb_infer_j_per_pred(inp, n)
            for kind in TRAIN_KINDS:
                for mode in MODES:
                    d_j = sum(dec[mode].values())
                    nb_j = sum(nb_inf[mode]) / len(nb_inf[mode])
                    mean, lo, hi = mean_ci(nb_train_j(inp, n, kind, mode))
                    t.add("break_even", f"{dkey}_vs_nb_n{n}", f"break_even_predictions_{kind}",
                          break_even(mean, d_j, nb_j), "predictions", mode,
                          (break_even(lo, d_j, nb_j), break_even(hi, d_j, nb_j)),
                          "N* = NB train / (Decider infer - NB infer), per prediction; "
                          "CI from the training-energy CI")  # fmt: skip


def main() -> None:
    setup_logging(load_settings().log_level)
    inp, t = Inputs(), Table()
    add_inference(t, inp)
    add_decider_load(t, inp)
    add_nb_training(t, inp)
    add_timing(t, inp)
    add_break_even(t, inp)
    out = RESULTS_DIR / "energy_summary.csv"
    t.frame().to_csv(out, index=False)
    log.info("wrote %s (%d rows)", out, len(t.rows))


if __name__ == "__main__":
    main()
