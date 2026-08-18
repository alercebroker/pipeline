"""Plot NaN (NULL value) rate per feature for alerce.feature versions 27.5.6 vs 27.5.7a32.dev1.

Population estimate from a random TABLESAMPLE SYSTEM (0.05) of the full 7.5B-row
alerce.feature (no cohort — unbiased across all objects that carry each version).
NaN == value IS NULL (no float 'NaN' is ever stored).
Outputs two PNGs: per-feature breakdown + mean-per-version summary.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRATCH = os.path.dirname(os.path.abspath(__file__))
V6, VD = "27.5.6", "27.5.7a32.dev1"
C6, CD = "#2a78d6", "#eb6834"          # validated categorical slots
INK, INK2, MUT = "#0b0b0b", "#52514e", "#898781"
GRID, SURF = "#e1e0d9", "#fcfcfb"
FID_LBL = {0: "", 1: " (g)", 2: " (r)", 12: " (g,r)"}

_exact = os.path.join(SCRATCH, "nan_per_feature_exact.csv")
_src = _exact if os.path.exists(_exact) else os.path.join(SCRATCH, "nan_per_feature_sampled.csv")
EXACT = _src.endswith("exact.csv")
df = pd.read_csv(_src)
df["feat"] = df["name"] + df["fid"].map(FID_LBL).fillna(df["fid"].astype(str))

wide = df.pivot_table(index="feat", columns="version", values="nan_pct").fillna(0.0)
wide = wide.reindex(columns=[V6, VD]).sort_values(VD, ascending=True)
mean6, meand = df[df.version == V6]["nan_pct"].mean(), df[df.version == VD]["nan_pct"].mean()

# ---------- Plot 1: per-feature ----------
n = len(wide)
fig, ax = plt.subplots(figsize=(11, max(6, n * 0.17)))
fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
y = np.arange(n); h = 0.4
ax.barh(y + h/2 + 0.01, wide[V6], height=h, color=C6, label=V6)
ax.barh(y - h/2 - 0.01, wide[VD], height=h, color=CD, label=VD)
ax.set_yticks(y); ax.set_yticklabels(wide.index, fontsize=6, color=INK2)
ax.set_ylim(-1, n)
ax.set_xlabel("% of objects with NaN (NULL) value", fontsize=10, color=INK2)
_sub = "exact full scan of alerce.feature (~2.0B rows)" if EXACT else "random 0.05% population sample of alerce.feature (~2.2k obs/feature)"
ax.set_title(f"NaN rate per feature — {V6} vs {VD}\n{_sub}",
             fontsize=12, color=INK, pad=12)
ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(GRID); ax.tick_params(colors=MUT, length=0)
ax.legend(loc="lower right", frameon=False, fontsize=10)
fig.tight_layout()
fig.savefig(f"{SCRATCH}/nan_per_feature.png", dpi=130, facecolor=SURF)
print("wrote nan_per_feature.png", wide.shape)

# ---------- Plot 2: mean per version ----------
fig, ax = plt.subplots(figsize=(5.2, 4.6))
fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
bars = ax.bar([V6, VD], [mean6, meand], color=[C6, CD], width=0.55)
for b, v in zip(bars, [mean6, meand]):
    ax.text(b.get_x()+b.get_width()/2, v+0.6, f"{v:.1f}%",
            ha="center", va="bottom", fontsize=13, color=INK, fontweight="bold")
ax.set_ylabel("mean % NaN across features", fontsize=10, color=INK2)
ax.set_title(f"Mean NaN rate per feature version\n({'exact full scan' if EXACT else 'random population sample'})", fontsize=12, color=INK, pad=12)
ax.set_ylim(0, max(mean6, meand)*1.25)
ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.spines["left"].set_visible(False); ax.spines["bottom"].set_color(GRID)
ax.tick_params(colors=MUT, length=0); ax.tick_params(axis="x", colors=INK, labelsize=10)
fig.tight_layout()
fig.savefig(f"{SCRATCH}/nan_mean_per_version.png", dpi=130, facecolor=SURF)
print(f"wrote nan_mean_per_version.png  mean {V6}={mean6:.2f}%  {VD}={meand:.2f}%")

# ---------- console: biggest movers ----------
wide["delta"] = wide[VD] - wide[V6]
print("\nTop 15 features by increase in NaN% (dev1 - 27.5.6):")
print(wide.sort_values("delta", ascending=False)[[V6, VD, "delta"]].head(15).round(1).to_string())
print("\nTop 10 features by DECREASE (dev1 lower):")
print(wide.sort_values("delta")[[V6, VD, "delta"]].head(10).round(1).to_string())
