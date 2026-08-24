"""Compare this run's class distribution against the WISE ablation baselines.

WISE_NULL_CLASSIFICATION_IMPACT.md established what BHRF does with and without
WISE colors, on a paired sample: the SAME objects predicted twice from the same
feature vector, once as-is and once with the eleven WISE colors blanked. That is
the yardstick this run has to be read against -- the whole point of running the
crossmatch was to land on the WISE-present side of it.

Both baselines are recomputed from wise_ablation.csv rather than quoted, so the
bars cannot drift from the source data.

READ THE POPULATION CAVEAT BEFORE READING THE BARS
--------------------------------------------------
The ablation sample is ~3,981 objects that HAVE WISE and a stored 27.5.6 vector
-- well-observed objects. This run is 19.3M objects at n_det >= 2, of which only
80.5% matched AllWISE at all. They are not the same population, so the bars are
not a controlled comparison: only the ablation's two bars are paired. The
Periodic/Stochastic axis is still meaningful because the ablation effect there
is enormous (87% -> 37%) and swamps any population difference. Transient is NOT
-- see the note the report prints.
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ABLATION = HERE.parent / "wise_ablation" / "wise_ablation.csv"

# Same palette as plot_wise_ablation.py, so the figures sit together.
INK, INK2, MUT = "#0b0b0b", "#52514e", "#898781"
GRID, SURF = "#e1e0d9", "#fcfcfb"
C_BASE, C_ABL, C_RUN = "#2a78d6", "#eb6834", "#1f9e6e"

TOP_ORDER = ["Periodic", "Stochastic", "Transient"]
TOP_CLASSIFIER = 6      # lc_classifier_BHRF_forced_phot_top
FLAT_CLASSIFIER = 5     # lc_classifier_BHRF_forced_phot


def load_ablation():
    df = pd.read_csv(ABLATION)
    ok = df[df["flat_base"].notna() & df["flat_abl"].notna()]
    return ok, len(ok)


def load_run():
    """This run's rank-1 shares, from the DB stats the stats script emitted."""
    df = pd.read_csv(HERE / "class_distribution.csv")
    top = df[df["classifier_id"] == TOP_CLASSIFIER]
    flat = df[df["classifier_id"] == FLAT_CLASSIFIER]
    return (top.set_index("class_name")["share_pct"],
            flat.set_index("class_name")["share_pct"])


def plot_top(ok, n, run_top, out):
    base = 100 * ok.top_base.value_counts().reindex(TOP_ORDER, fill_value=0) / n
    abl = 100 * ok.top_abl.value_counts().reindex(TOP_ORDER, fill_value=0) / n
    run = run_top.reindex(TOP_ORDER).fillna(0)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    fig.patch.set_facecolor(SURF)
    ax.set_facecolor(SURF)
    x, w = np.arange(len(TOP_ORDER)), 0.27
    series = [(base, C_BASE, f"ablation baseline — WISE present (n={n:,})"),
              (abl, C_ABL, "ablation — WISE blanked to NaN"),
              (run, C_RUN, "this run — full catalogue, WISE via Xwave")]
    for i, (vals, colour, label) in enumerate(series):
        pos = x + (i - 1) * w
        ax.bar(pos, vals, w, color=colour, label=label)
        for xi, v in zip(pos, vals):
            ax.text(xi, v + 0.8, f"{v:.1f}", ha="center", fontsize=8.5, color=INK2)

    ax.set_xticks(x)
    ax.set_xticklabels(TOP_ORDER, color=INK)
    ax.set_ylabel("% of objects", fontsize=10, color=INK2)
    ax.set_ylim(0, 100)
    ax.set_title(
        "BHRF top head: this run against the WISE ablation\n"
        "Periodic lands with WISE present, not with WISE blanked",
        fontsize=11, color=INK, pad=10, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUT, length=0)
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor=SURF)
    return base, abl, run


def plot_flat(ok, n, run_flat, out, top_n=12):
    base = 100 * ok.flat_base.value_counts() / n
    abl = 100 * ok.flat_abl.value_counts() / n
    order = base.sort_values(ascending=False).head(top_n).index.tolist()
    b = base.reindex(order).fillna(0)
    a = abl.reindex(order).fillna(0)
    r = run_flat.reindex(order).fillna(0)

    fig, ax = plt.subplots(figsize=(11, 5.4))
    fig.patch.set_facecolor(SURF)
    ax.set_facecolor(SURF)
    x, w = np.arange(len(order)), 0.27
    for i, (vals, colour, label) in enumerate(
            [(b, C_BASE, "ablation baseline — WISE present"),
             (a, C_ABL, "ablation — WISE blanked"),
             (r, C_RUN, "this run")]):
        ax.bar(x + (i - 1) * w, vals, w, color=colour, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels(order, color=INK, rotation=35, ha="right")
    ax.set_ylabel("% of objects", fontsize=10, color=INK2)
    ax.set_title(
        f"BHRF flat head, {top_n} most common baseline classes\n"
        "the WISE-null collapse (Periodic-Other/LPV/RSCVn -> CV/Nova, YSO) "
        "did not happen in this run",
        fontsize=11, color=INK, pad=10, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUT, length=0)
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor=SURF)
    return b, a, r


def main():
    ok, n = load_ablation()
    run_top, run_flat = load_run()

    base, abl, run = plot_top(ok, n, run_top,
                              HERE / "run_vs_ablation_top.png")
    plot_flat(ok, n, run_flat, HERE / "run_vs_ablation_flat.png")

    print(f"ablation sample: {n:,} objects\n")
    print(f"{'class':<12} {'WISE present':>13} {'WISE blanked':>13} {'this run':>10}")
    for c in TOP_ORDER:
        print(f"{c:<12} {base[c]:>12.1f}% {abl[c]:>12.1f}% {run[c]:>9.2f}%")
    print("\nwrote run_vs_ablation_top.png, run_vs_ablation_flat.png")


if __name__ == "__main__":
    main()
