"""Plot the WISE-NaN ablation: does blanking WISE colors move BHRF predictions?
Top-head marginal shift (grouped bars) + top-head transition heatmap."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRATCH = os.path.dirname(os.path.abspath(__file__))
INK, INK2, MUT = "#0b0b0b", "#52514e", "#898781"
GRID, SURF = "#e1e0d9", "#fcfcfb"
C_BASE, C_ABL = "#2a78d6", "#eb6834"

df = pd.read_csv(f"{SCRATCH}/wise_ablation.csv")
ok = df[df["flat_base"].notna() & df["flat_abl"].notna()].copy()
n = len(ok)
flat_same = 100*(ok.flat_base == ok.flat_abl).mean()
top_same = 100*(ok.top_base == ok.top_abl).mean()

# ---- Plot 1: top-head marginal shift ----
order = ["Periodic", "Stochastic", "Transient"]
base = ok.top_base.value_counts().reindex(order, fill_value=0)
abl  = ok.top_abl.value_counts().reindex(order, fill_value=0)
fig, ax = plt.subplots(figsize=(7.5, 5))
fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
x = np.arange(len(order)); w = 0.38
ax.bar(x-w/2, 100*base/n, w, color=C_BASE, label="WISE present (baseline)")
ax.bar(x+w/2, 100*abl/n,  w, color=C_ABL,  label="WISE = NaN (ablated)")
for i, (b, a) in enumerate(zip(base, abl)):
    ax.text(i-w/2, 100*b/n+0.4, f"{100*b/n:.1f}", ha="center", fontsize=9, color=INK2)
    ax.text(i+w/2, 100*a/n+0.4, f"{100*a/n:.1f}", ha="center", fontsize=9, color=INK2)
ax.set_xticks(x); ax.set_xticklabels(order, color=INK)
ax.set_ylabel("% of objects", fontsize=10, color=INK2)
ax.set_title(f"WISE-NaN effect on BHRF top head  (n={n} WISE-populated objects)\n"
             f"top class unchanged for {top_same:.1f}% of objects, flat for {flat_same:.1f}%",
             fontsize=11, color=INK, pad=10, loc="left")
ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
for s in ("top","right","left"): ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(GRID); ax.tick_params(colors=MUT, length=0)
fig.tight_layout(); fig.savefig(f"{SCRATCH}/wise_ablation_top_marginal.png", dpi=130, facecolor=SURF)
print("wrote wise_ablation_top_marginal.png")

# ---- Plot 2: top-head transition heatmap (row-normalized) ----
ct = pd.crosstab(ok.top_base, ok.top_abl).reindex(index=order, columns=order, fill_value=0)
rown = ct.div(ct.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
fig, ax = plt.subplots(figsize=(6, 5))
fig.patch.set_facecolor(SURF)
im = ax.imshow(rown.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(order))); ax.set_xticklabels(order, color=INK2)
ax.set_yticks(range(len(order))); ax.set_yticklabels(order, color=INK2)
ax.set_xlabel("ablated (WISE = NaN)", color=INK2); ax.set_ylabel("baseline (WISE present)", color=INK2)
ax.set_title("Top-head transition (row-normalized)", fontsize=11, color=INK, pad=10)
for i in range(len(order)):
    for j in range(len(order)):
        v = rown.values[i, j]
        ax.text(j, i, f"{v*100:.1f}%\n({ct.values[i,j]})", ha="center", va="center",
                fontsize=9, color="white" if v > 0.5 else INK)
for s in ("top","right","left","bottom"): ax.spines[s].set_visible(False)
ax.tick_params(length=0)
fig.tight_layout(); fig.savefig(f"{SCRATCH}/wise_ablation_top_transition.png", dpi=130, facecolor=SURF)
print("wrote wise_ablation_top_transition.png")
