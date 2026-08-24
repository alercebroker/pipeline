"""Class distribution stratified by n_det: what does a short light curve cost?

Two things in this run look nothing like the WISE ablation baselines -- Transient
at 11.6% (baseline 0.2%) and Periodic-Other at 37.6% (baseline 19.6%) -- and the
suspicion in RUN_VS_WISE_ABLATION.md is that both come from the `n_det >= 2` cut
rather than from anything the model did wrong. This is the check. If sparse light
curves are the cause, both should fall away as n_det grows, and the well-observed
stratum should look like the baseline.

JOINABILITY
-----------
`probability` is HASH(oid) over 16 partitions and `object` HASH(oid) over 8. A
Postgres hash partition holds the oids where hash % modulus == remainder, so
hash % 16 == 0 implies hash % 8 == 0: probability_part_0's oids are a SUBSET of
object_part_0's. Joining those two partitions therefore loses nothing -- it is a
complete 1/16 sample of objects, not an intersection of two samples.

Counts are scaled by 16 for readability; every percentage is computed within its
own stratum and is unaffected by the scaling.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sqlalchemy as sa
from sqlalchemy import text

HERE = Path(__file__).resolve().parent
CREDS = HERE.parent / "credentials.json"
SCHEMA, SID, N_PARTS = "multisurvey_ztf", 0, 16
TOP, FLAT = 6, 5

INK, INK2, MUT = "#0b0b0b", "#52514e", "#898781"
GRID, SURF = "#e1e0d9", "#fcfcfb"
# light -> dark as the light curve gets longer
STRATA_COLOURS = ["#f3b562", "#5aa9e6", "#1f9e6e"]
STRATA = ["2-3", "4-7", "8+"]
FINE_TO_STRATUM = {"2-3": "2-3", "4-7": "4-7",
                   "8-15": "8+", "16-31": "8+", "32-63": "8+", "64+": "8+"}


def fetch() -> pd.DataFrame:
    p = json.loads(CREDS.read_text())
    url = (f"postgresql+psycopg2://{p['user']}:{p['password']}"
           f"@{p['host']}:{p['port']}/{p['dbname']}")
    engine = sa.create_engine(url, connect_args={"connect_timeout": 90})
    # Finer bins than the three plotted, so the CSV can answer follow-ups
    # (where exactly does Transient fall off?) without another scan.
    sql = text(f"""
        SELECT p.classifier_id,
               t.class_name,
               CASE WHEN o.n_det <  4 THEN '2-3'
                    WHEN o.n_det <  8 THEN '4-7'
                    WHEN o.n_det < 16 THEN '8-15'
                    WHEN o.n_det < 32 THEN '16-31'
                    WHEN o.n_det < 64 THEN '32-63'
                    ELSE '64+' END AS n_det_bin,
               count(*) AS n
        FROM {SCHEMA}.probability_part_0 p
        JOIN {SCHEMA}.object_part_0 o
          ON o.oid = p.oid AND o.sid = p.sid
        JOIN {SCHEMA}.taxonomy t
          ON t.classifier_id = p.classifier_id AND t.class_id = p.class_id
        WHERE p.sid = :sid AND p.ranking = 1
          AND p.classifier_id IN (:top, :flat)
        GROUP BY 1, 2, 3
    """)
    with engine.connect() as c:
        c.execute(text("SET statement_timeout = '3600s'"))
        c.execute(text("SET work_mem = '512MB'"))
        df = pd.read_sql_query(sql, c, params={"sid": SID, "top": TOP, "flat": FLAT})
    engine.dispose()
    df["n_full"] = df["n"] * N_PARTS
    df["stratum"] = df["n_det_bin"].map(FINE_TO_STRATUM)
    return df


def _shares(df, classifier):
    """-> DataFrame [class_name x stratum] of within-stratum percentages."""
    d = df[df.classifier_id == classifier]
    piv = (d.groupby(["class_name", "stratum"])["n_full"].sum()
             .unstack("stratum").reindex(columns=STRATA).fillna(0))
    return 100 * piv / piv.sum(), piv.sum()


def _style(ax):
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUT, length=0)


def plot_top(df, out):
    share, totals = _shares(df, TOP)
    order = ["Periodic", "Stochastic", "Transient"]
    share = share.reindex(order)

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
    x, w = np.arange(len(order)), 0.26
    for i, s in enumerate(STRATA):
        pos = x + (i - 1) * w
        ax.bar(pos, share[s], w, color=STRATA_COLOURS[i],
               label=f"n_det {s}  ({totals[s]/1e6:.1f}M objects)")
        for xi, v in zip(pos, share[s]):
            ax.text(xi, v + 0.8, f"{v:.1f}", ha="center", fontsize=8.5, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels(order, color=INK)
    ax.set_ylabel("% of objects in that stratum", fontsize=10, color=INK2)
    ax.set_ylim(0, 100)
    ax.set_title("BHRF top head by light-curve length\n"
                 "Transient is a short-light-curve artefact: it falls away as "
                 "n_det grows",
                 fontsize=11, color=INK, pad=10, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
    _style(ax)
    fig.tight_layout(); fig.savefig(out, dpi=130, facecolor=SURF)
    return share, totals


def plot_flat(df, out, top_n=14):
    share, totals = _shares(df, FLAT)
    order = share["8+"].sort_values(ascending=False).head(top_n).index.tolist()
    share = share.reindex(order)

    fig, axes = plt.subplots(len(STRATA), 1, figsize=(11.5, 11), sharex=True)
    fig.patch.set_facecolor(SURF)
    for ax, s, colour in zip(axes, STRATA, STRATA_COLOURS):
        ax.set_facecolor(SURF)
        ax.bar(np.arange(len(order)), share[s], 0.68, color=colour)
        for xi, v in enumerate(share[s]):
            ax.text(xi, v + 0.6, f"{v:.1f}", ha="center", fontsize=8, color=INK2)
        ax.set_title(f"n_det {s}   —   {totals[s]/1e6:.1f}M objects",
                     fontsize=10.5, color=INK, loc="left", pad=6)
        ax.set_ylabel("% of stratum", fontsize=9.5, color=INK2)
        ax.set_ylim(0, max(45, share[s].max() * 1.18))
        _style(ax)
    axes[-1].set_xticks(np.arange(len(order)))
    axes[-1].set_xticklabels(order, color=INK, rotation=35, ha="right")
    fig.suptitle("BHRF flat head by light-curve length "
                 f"({top_n} most common classes among well-observed objects)",
                 fontsize=11.5, color=INK, x=0.008, ha="left", y=0.998)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out, dpi=130, facecolor=SURF)
    return share


def main():
    df = fetch()
    df.to_csv(HERE / "class_by_ndet.csv", index=False)
    top, totals = plot_top(df, HERE / "class_by_ndet_top.png")
    flat = plot_flat(df, HERE / "class_by_ndet_flat.png")

    print("objects per stratum (scaled x16):")
    for s in STRATA:
        print(f"  n_det {s:<4} {totals[s]:>12,.0f}")
    print("\ntop head, % within stratum:")
    print(top.round(2).to_string())
    print("\nflat head, % within stratum:")
    print(flat.round(2).to_string())
    print("\nwrote class_by_ndet_top.png, class_by_ndet_flat.png, class_by_ndet.csv")


if __name__ == "__main__":
    main()
