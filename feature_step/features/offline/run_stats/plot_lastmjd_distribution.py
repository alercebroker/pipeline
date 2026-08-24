"""When were the run's objects last detected? Monthly histogram of `lastmjd`.

`lastmjd` is stored on every probability row, so this is the run's real coverage
curve: not a discovery-date distribution, but a recency one -- how long ago each
classified object was last seen. Read from one `probability` partition (an
unbiased 1/16 of objects, HASH on oid) and scaled up; see offline_db_stats.py
for why partitions are the sampling unit.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import sqlalchemy as sa
from sqlalchemy import text

HERE = Path(__file__).resolve().parent
CREDS = HERE.parent / "credentials.json"
PART, N_PARTS = "probability_part_0", 16
SCHEMA, SID, CLASSIFIER = "multisurvey_ztf", 0, 5

INK, INK2, MUT = "#0b0b0b", "#52514e", "#898781"
GRID, SURF, BAR = "#e1e0d9", "#fcfcfb", "#2a78d6"


def fetch() -> pd.DataFrame:
    p = json.loads(CREDS.read_text())
    url = (f"postgresql+psycopg2://{p['user']}:{p['password']}"
           f"@{p['host']}:{p['port']}/{p['dbname']}")
    engine = sa.create_engine(url, connect_args={"connect_timeout": 90})
    with engine.connect() as c:
        c.execute(text("SET statement_timeout = '1800s'"))
        # MJD 0 is 1858-11-17; floor() first so the cast cannot round a day up.
        df = pd.read_sql_query(text(f"""
            SELECT date_trunc('month',
                       (DATE '1858-11-17' + floor(lastmjd)::int))::date AS month,
                   count(*) AS n
            FROM {SCHEMA}.{PART}
            WHERE sid = :sid AND classifier_id = :cid AND ranking = 1
            GROUP BY 1 ORDER BY 1
        """), c, params={"sid": SID, "cid": CLASSIFIER})
    engine.dispose()
    df["month"] = pd.to_datetime(df["month"])
    df["n_full"] = df["n"] * N_PARTS
    return df


def plot(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5.4))
    fig.patch.set_facecolor(SURF)
    ax.set_facecolor(SURF)
    ax.bar(df["month"], df["n_full"], width=24, color=BAR, align="center")

    lo, hi = df["month"].min(), df["month"].max()
    total = df["n_full"].sum()
    ax.set_title(
        "When the run's objects were last detected  "
        f"(lastmjd, {total/1e6:.1f}M classified objects)\n"
        f"{lo:%Y-%m-%d} to 2026-08-14  —  August is a partial month",
        fontsize=11, color=INK, pad=10, loc="left")
    ax.set_ylabel("objects", fontsize=10, color=INK2)
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1e6:g}M" if v else "0"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(4, 7, 10)))
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUT, length=0)
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor=SURF)


def main():
    df = fetch()
    df.to_csv(HERE / "lastmjd_per_month.csv", index=False)
    plot(df, HERE / "lastmjd_distribution.png")
    print(f"{len(df)} months, {df['month'].min():%Y-%m} to {df['month'].max():%Y-%m}, "
          f"{df['n_full'].sum():,} objects")
    print("wrote lastmjd_distribution.png, lastmjd_per_month.csv")


if __name__ == "__main__":
    main()
