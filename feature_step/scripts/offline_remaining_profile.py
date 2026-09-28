#!/usr/bin/env python
"""What is coming: how heavy are the objects of the units not yet processed?

A unit's cost follows its objects' detection counts, and a sorted oid array
walks through blocks of very different weight. This samples a few hundred oids
per block of units, reads n_det from <schema>.object by primary key, and prints
one line per block: how many of its units are done, the mean / p90 n_det, the
share of heavy objects, and -- calibrated on the blocks already landed -- a
projected seconds-per-oid and hours for what remains.

    poetry run python scripts/offline_remaining_profile.py \\
        --oid-file $RUN/oids/reprocesados.npy --out-dir $RUN/bhrf_reproc \\
        --credentials <read credentials> --workers 96

Read-only: (blocks x --per-block) primary-key lookups, about a minute.
The projection assumes cost per oid scales with mean n_det, which is rough:
the extractors are superlinear in the tail. Read it as "heavier / lighter than
what has landed", not as an ETA to the minute.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parent))
import offline_run_batch as M  # noqa: E402
from features.offline import db  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oid-file", required=True)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--unit-size", type=int, default=5000)
    ap.add_argument("--block", type=int, default=25, help="units per block (default 25)")
    ap.add_argument("--per-block", type=int, default=300, help="oids sampled per block")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--credentials", default=M.DEFAULT_CREDENTIALS)
    args = ap.parse_args()

    oids = np.load(args.oid_file).astype(np.int64)
    n_units = -(-len(oids) // args.unit_size)
    done = {}
    for p in (args.out_dir / "manifests").glob("unit_*.json"):
        m = json.loads(p.read_text())
        done[m["unit"]] = m["elapsed_s"] / m["n_oids"]

    rng = np.random.default_rng(args.seed)
    engine = db._make_engine(args.credentials)
    rows = []
    with engine.connect() as conn:
        conn.execute(text("SET statement_timeout = '120s'"))
        for b0 in range(0, n_units, args.block):
            b1 = min(b0 + args.block, n_units)
            sl = oids[b0 * args.unit_size:b1 * args.unit_size]
            pick = rng.choice(sl, min(args.per_block, len(sl)), replace=False)
            d = pd.read_sql(text(f"SELECT n_det FROM {db.SCHEMA}.object WHERE sid = :sid AND oid = ANY(:oids)"),
                            conn, params={"sid": db.SID, "oids": [int(x) for x in pick]})
            landed = [done[u] for u in range(b0, b1) if u in done]
            rows.append({"b0": b0, "b1": b1 - 1, "units": b1 - b0, "done": len(landed),
                         "n_det_mean": d.n_det.mean(), "n_det_p90": d.n_det.quantile(.9),
                         "ge50": (d.n_det >= 50).mean(), "ge200": (d.n_det >= 200).mean(),
                         "spo_landed": statistics.mean(landed) if landed else None})
    df = pd.DataFrame(rows)

    # calibrate s/oid per unit of mean n_det on blocks with >= 3 landed units
    cal = df[(df.done >= 3) & df.spo_landed.notna()]
    k = (cal.spo_landed / cal.n_det_mean).median() if len(cal) else None
    df["spo_proj"] = df.spo_landed.where(df.spo_landed.notna(), df.n_det_mean * k if k else np.nan)
    df["pending"] = df.units - df.done
    df["hours_proj"] = df.pending * args.unit_size * df.spo_proj / args.workers / 3600

    print(f"{len(oids):,} oids, {n_units} units of {args.unit_size}; {len(done)} done, {n_units - len(done)} pending; "
          f"blocks of {args.block} units, {args.per_block} oids sampled each")
    if k:
        print(f"calibration: median s/oid per unit of mean n_det = {k:.4f} over {len(cal)} landed blocks "
              f"(s/oid ~ {k:.4f} x mean n_det)")
    else:
        print("calibration: no block with >= 3 landed units yet -- projections blank")
    print(f"\n{'units':>11} {'done':>4} {'mean n_det':>10} {'p90':>6} {'>=50':>6} {'>=200':>6} {'s/oid landed':>12} {'s/oid proj':>10} {'hours left':>10}")
    for r in df.itertuples():
        sl = f"{r.spo_landed:12.3f}" if r.spo_landed is not None and not np.isnan(r.spo_landed) else f"{'-':>12}"
        sp = f"{r.spo_proj:10.3f}" if not np.isnan(r.spo_proj) else f"{'-':>10}"
        hp = f"{r.hours_proj:10.1f}" if not np.isnan(r.hours_proj) else f"{'-':>10}"
        flag = "  <- heavy" if r.n_det_mean >= 2 * df.n_det_mean.median() else ""
        print(f"{r.b0:5d}-{r.b1:<5d} {r.done:4d} {r.n_det_mean:10.1f} {r.n_det_p90:6.0f} {r.ge50:6.1%} {r.ge200:6.1%} {sl} {sp} {hp}{flag}")
    if k:
        left = df.hours_proj.sum()
        print(f"\nprojected for the {int(df.pending.sum())} pending units at {args.workers} workers: ~{left:.0f} h "
              f"({left / 24:.1f} days) -- rough, see the docstring")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
