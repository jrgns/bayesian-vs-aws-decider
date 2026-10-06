"""Plots: the learning curve with the crossover, and energy per 1,000 predictions.

    python -m scripts.plots

Reads nb_runs.csv, decider_runs.csv, crossover.json, energy_summary.csv, dataset_summary.json;
writes plots/learning_curve.png and plots/energy.png (150 dpi).
"""

from __future__ import annotations

import json
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.common import (  # noqa: E402
    APP_DIR,
    RESULTS_DIR,
    decider_test_runs,
    load_settings,
    setup_logging,
)
from scripts.nb_curve import FULL, N_PER_CLASS, VARIANTS  # noqa: E402
from scripts.summary_stats import mean_ci  # noqa: E402

log = logging.getLogger(__name__)

PLOTS_DIR = APP_DIR / "plots"
DPI = 150
# Okabe-Ito: colour-blind safe.
BLUE, ORANGE, VERMILLION, GREEN, SKY, PINK, GREY = (
    "#0072B2", "#E69F00", "#D55E00", "#009E73", "#56B4E9", "#CC79A7", "#555555",
)  # fmt: skip
VARIANT_STYLE = {
    "mnb_count": (ORANGE, "-"),
    "mnb_tfidf": (GREEN, "--"),
    "cnb_count": (PINK, "-."),
    "cnb_tfidf": (SKY, ":"),
}


def style_axes(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)


def learning_curve(runs: pd.DataFrame, decider: pd.Series, cross_n: int, pool_size: int) -> None:
    runs = runs.assign(n_per_class=runs["n_per_class"].astype(str))
    xs = list(N_PER_CLASS)
    fig, ax = plt.subplots(figsize=(9, 5.5))

    best = runs[runs["is_best"]]
    stats = [mean_ci(best[best["n_per_class"] == str(n)]["test_accuracy"]) for n in xs]
    m, lo, hi = (np.array(c) for c in zip(*stats, strict=True))
    for variant in VARIANTS:
        v = runs[runs["variant"] == variant]
        ys = [v[v["n_per_class"] == str(n)]["test_accuracy"].mean() for n in xs]
        colour, ls = VARIANT_STYLE[variant]
        ax.plot(xs, ys, color=colour, ls=ls, lw=1.1, label=f"NB {variant} (mean of 5 seeds)")
    ax.fill_between(xs, lo, hi, color=BLUE, alpha=0.25, lw=0)
    ax.plot(xs, m, color=BLUE, lw=2.5, marker="o", ms=4,
            label="NB best config by CV (mean, 95% CI over 5 seeds)")  # fmt: skip

    d_acc, d_lo, d_hi = (decider[k] for k in ("accuracy", "accuracy_ci_lo", "accuracy_ci_hi"))
    ax.axhspan(d_lo, d_hi, color=VERMILLION, alpha=0.2, lw=0)
    ax.axhline(d_acc, color=VERMILLION, lw=2,
               label=f"Decider 2B zero-shot, bs=1 ({d_acc:.3f}, 95% bootstrap CI)")  # fmt: skip

    full = best[best["n_per_class"] == FULL]["test_accuracy"]
    f_mean, f_lo, f_hi = mean_ci(full)
    x_full = pool_size / 7
    ax.errorbar([x_full], [f_mean], yerr=[[f_mean - f_lo], [f_hi - f_mean]], fmt="D", color=BLUE,
                mfc="white", ms=7, capsize=3, label="NB, full pool (imbalanced)")  # fmt: skip
    ax.annotate("full pool (imbalanced)", (x_full, f_mean), textcoords="offset points",
                xytext=(-8, -22), ha="right", fontsize=9, color=BLUE)  # fmt: skip

    cx = xs.index(cross_n)
    ax.axvline(cross_n, color=GREY, ls=":", lw=1)
    ax.plot([cross_n], [m[cx]], "o", ms=11, mfc="none", mec="black", mew=1.8, zorder=5)
    ax.annotate(f"crossover: {cross_n}/class\n({7 * cross_n} labelled examples)",
                (cross_n, m[cx]), textcoords="offset points", xytext=(14, -42), fontsize=9,
                arrowprops={"arrowstyle": "-", "color": "black"})  # fmt: skip

    ax.set_xscale("log")
    ticks = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_xlim(0.8, x_full * 1.6)
    ax.set_ylim(0.1, 0.86)
    ax.set_xlabel("Training examples per class (log scale)")
    ax.set_ylabel("Accuracy on the balanced test split (n = 9,996)")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    style_axes(ax)
    fig.tight_layout()
    out = PLOTS_DIR / "learning_curve.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    log.info("wrote %s", out)


def energy_plot(summary: pd.DataFrame, cross_n: int) -> None:
    inf = summary[
        (summary["table"] == "inference") & (summary["metric"] == "kwh_per_1000_predictions")
    ]
    subjects = [
        (
            f"decider_bs{bs}",
            f"Decider 2B, bs={bs}\n({'headline' if bs == 1 else 'best throughput'})",
        )
        for bs in decider_test_runs()
    ]
    subjects.append((f"nb_n{cross_n}", f"Naive Bayes, {cross_n}/class\n(crossover, CPU only)"))
    vals = {
        mode: [float(inf[(inf["subject"] == s) & (inf["mode"] == mode)]["value"].iloc[0])
               for s, _ in subjects]
        for mode in ("gross", "marginal")
    }  # fmt: skip
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    x, w = np.arange(len(subjects)), 0.36
    for off, mode, colour, hatch in (
        (-w / 2, "gross", BLUE, ""),
        (w / 2, "marginal", ORANGE, "//"),
    ):
        bars = ax.bar(x + off, vals[mode], w, color=colour, hatch=hatch, edgecolor="white",
                      label="marginal (gross minus idle power x time)" if mode == "marginal"
                      else "gross (Decider: GPU + CPU package; NB: CPU package)")  # fmt: skip
        for b, v in zip(bars, vals[mode], strict=True):
            ax.text(
                b.get_x() + b.get_width() / 2,
                v * 1.15,
                f"{v:.2e}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    ax.set_yscale("log")
    ax.set_ylim(1e-7, 1e-1)
    ax.set_xticks(x, [label for _, label in subjects])
    ax.set_ylabel("Energy per 1,000 predictions (kWh, log scale)")
    ratio = vals["gross"][0] / vals["gross"][2]
    ax.set_title(
        f"Decider (bs=1) uses about {ratio:,.0f}x the energy of NB per prediction (gross)",
        fontsize=11,
    )
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    style_axes(ax)
    fig.tight_layout()
    out = PLOTS_DIR / "energy.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    log.info("wrote %s", out)


def main() -> None:
    setup_logging(load_settings().log_level)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    runs = pd.read_csv(RESULTS_DIR / "nb_runs.csv")
    decider = (
        pd.read_csv(RESULTS_DIR / "decider_runs.csv")
        .set_index("run_id")
        .loc[next(iter(decider_test_runs().values()))]
    )
    cross = json.loads((RESULTS_DIR / "crossover.json").read_text())["headline"]
    pool_size = json.loads((RESULTS_DIR / "dataset_summary.json").read_text())["n_items"]["pool"]
    summary = pd.read_csv(RESULTS_DIR / "energy_summary.csv")
    cross_n = int(cross["crossover_n_per_class"])
    learning_curve(runs, decider, cross_n, pool_size)
    energy_plot(summary, cross_n)


if __name__ == "__main__":
    main()
