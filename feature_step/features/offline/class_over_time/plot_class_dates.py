"""Class distribution across dates from stored BHRF predictions. Top 3-way head +
flat leaf. Random DB sample. Bucketed by lastmjd (last detection ~= when the
prediction was last (re)generated, since BHRF re-runs on each new detection)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRATCH = os.path.dirname(os.path.abspath(__file__))
INK, INK2, MUT = "#0b0b0b", "#52514e", "#898781"
GRID, SURF = "#e1e0d9", "#fcfcfb"
CAT = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]
OTHER = "#898781"

# (date column, pandas period freq, x-axis label) for the bucketing
DATECOL, FREQ, XLABEL = "last_date", "M", "last detection (lastmjd), monthly"

def load(path):
    df = pd.read_csv(path, parse_dates=["first_date", "last_date"])
    df["q"] = df[DATECOL].dt.to_period(FREQ).dt.to_timestamp()
    return df

def order_classes(df, fixed=None, top_n=8):
    if fixed:
        return fixed
    vc = df["class_name"].value_counts()
    keep = list(vc.index[:top_n])
    return keep + (["Other"] if len(vc) > top_n else [])

def make(df, classes, title, fname, colors):
    df = df.copy()
    if "Other" in classes:
        df["cls"] = df["class_name"].where(df["class_name"].isin(classes[:-1]), "Other")
    else:
        df["cls"] = df["class_name"]
    ct = pd.crosstab(df["q"], df["cls"]).reindex(columns=classes, fill_value=0).sort_index()
    frac = ct.div(ct.sum(axis=1), axis=0)
    x = ct.index
    cmap = {c: colors[i] for i, c in enumerate(classes)}

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True,
                                 gridspec_kw={"height_ratios": [2.2, 1]})
    for ax in (a1, a2):
        ax.set_facecolor(SURF)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        ax.spines["left"].set_color(GRID); ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=MUT, length=0)
    fig.patch.set_facecolor(SURF)

    a1.stackplot(x, *[frac[c].values for c in classes],
                 colors=[cmap[c] for c in classes], labels=classes, edgecolor=SURF, linewidth=0.3)
    a1.set_ylim(0, 1); a1.set_ylabel("class share", fontsize=10, color=INK2)
    a1.set_title(title, fontsize=13, color=INK, pad=12, loc="left")
    a1.legend(loc="center left", bbox_to_anchor=(1.005, 0.5), frameon=False,
              fontsize=9, labelcolor=INK2)
    a1.margins(x=0)

    bottom = np.zeros(len(x))
    w = (x[1]-x[0]).days*0.85 if len(x) > 1 else 20
    for c in classes:
        a2.bar(x, ct[c].values, bottom=bottom, width=w, color=cmap[c], label=c)
        bottom += ct[c].values
    a2.set_ylabel("objects (sample)", fontsize=10, color=INK2)
    a2.set_xlabel(XLABEL, fontsize=10, color=INK2)
    a2.yaxis.grid(True, color=GRID, lw=0.6); a2.set_axisbelow(True); a2.margins(x=0)

    fig.tight_layout()
    fig.savefig(f"{SCRATCH}/{fname}", dpi=130, facecolor=SURF, bbox_inches="tight")
    print("wrote", fname, "bins", len(x), "range", str(x.min().date()), "..", str(x.max().date()))

top = load(f"{SCRATCH}/class_dates_top.csv")
flat = load(f"{SCRATCH}/class_dates_flat.csv")

make(top, ["Periodic", "Stochastic", "Transient"],
     "BHRF top-head class composition by last-detection date (top 3-way)",
     "class_dates_top_lastmjd.png", ["#2a78d6", "#1baf7a", "#eb6834"])

make(flat, order_classes(flat, top_n=8),
     "BHRF flat class composition by last-detection date (top 8 + Other)",
     "class_dates_flat_lastmjd.png", CAT + [OTHER])
print("DONE")
