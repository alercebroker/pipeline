#!/usr/bin/env python
"""Are the objects of one run heavier than the objects of another?

Two runs over overlapping oid ranges are not comparable per range if they do
not process the same objects: the cost of a unit is dominated by its heaviest
light curves, so a subset enriched in active objects costs more per oid than
the range average, with nothing slowed down. This takes two oid arrays,
splits them into "only in A", "in both", "only in B", samples each group, and
reads those objects' n_det / n_forced from <schema>.object by primary key.

    poetry run python scripts/offline_compare_object_sets.py \\
        --a $RUN/oids/run.npy --b $RUN/oids/reprocesados.npy \\
        --credentials <read credentials> --sample 20000

Read-only, a few thousand primary-key lookups, under a minute.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parent))
import offline_run_batch as M  # noqa: E402  (pipeline paths)
from features.offline import db  # noqa: E402


def fetch(engine, oids: np.ndarray, chunk: int = 5000) -> pd.DataFrame:
    parts = []
    with engine.connect() as conn:
        conn.execute(text("SET statement_timeout = '120s'"))
        for i in range(0, len(oids), chunk):
            parts.append(pd.read_sql(
                text(f"SELECT oid, n_det, n_forced, firstmjd, lastmjd FROM {db.SCHEMA}.object "
                     "WHERE sid = :sid AND oid = ANY(:oids)"),
                conn, params={"sid": db.SID, "oids": [int(x) for x in oids[i:i + chunk]]}))
    return pd.concat(parts, ignore_index=True)


def describe(label: str, d: pd.DataFrame) -> str:
    q = d.n_det.quantile
    return (f"  {label:24s} {len(d):7,d} sampled | n_det mean {d.n_det.mean():7.1f} "
            f"median {d.n_det.median():5.0f} p90 {q(.9):6.0f} p99 {q(.99):7.0f} | "
            f">=50: {(d.n_det >= 50).mean():5.1%}  >=200: {(d.n_det >= 200).mean():5.2%} | "
            f"n_forced mean {d.n_forced.mean():6.1f} | span mean {(d.lastmjd - d.firstmjd).mean():5.0f} d")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="oid array of run A (e.g. the full run's run.npy)")
    ap.add_argument("--b", required=True, help="oid array of run B (e.g. reprocesados.npy)")
    ap.add_argument("--label-a", default="August run")
    ap.add_argument("--label-b", default="this run")
    ap.add_argument("--sample", type=int, default=20000, help="objects sampled per group")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--credentials", default=M.DEFAULT_CREDENTIALS)
    args = ap.parse_args()

    a = np.load(args.a).astype(np.int64)
    b = np.load(args.b).astype(np.int64)
    only_a = np.setdiff1d(a, b, assume_unique=True)
    both = np.intersect1d(a, b, assume_unique=True)
    only_b = np.setdiff1d(b, a, assume_unique=True)
    print(f"{args.label_a}: {len(a):,} oids   {args.label_b}: {len(b):,} oids")
    print(f"  only in {args.label_a}: {len(only_a):,}   in both: {len(both):,}   only in {args.label_b}: {len(only_b):,}")

    rng = np.random.default_rng(args.seed)
    engine = db._make_engine(args.credentials)
    groups = {}
    for name, arr in ((f"only {args.label_a}", only_a), ("in both", both), (f"only {args.label_b}", only_b)):
        if not len(arr):
            continue
        pick = rng.choice(arr, min(args.sample, len(arr)), replace=False)
        groups[name] = fetch(engine, pick)

    print("\nper group (random sample, object table by primary key):")
    for name, d in groups.items():
        print(describe(name, d))

    ga = groups.get(f"only {args.label_a}")
    gb = pd.concat([g for k, g in groups.items() if k != f"only {args.label_a}"], ignore_index=True)
    if ga is not None and len(gb):
        print(f"\n{args.label_b}'s objects vs objects only {args.label_a} processed:")
        print(f"  n_det    x{gb.n_det.mean() / ga.n_det.mean():.2f}   n_forced x{gb.n_forced.mean() / ga.n_forced.mean():.2f}")
        top = lambda d: d.n_det.sort_values().tail(max(1, len(d) // 20)).sum() / d.n_det.sum()
        print(f"  detections held by the heaviest 5% of objects: {args.label_a}-only {top(ga):.0%}, {args.label_b} {top(gb):.0%}")
        print("\n  n_det histogram (share of objects):")
        edges = [2, 5, 10, 20, 50, 100, 200, 500, 1000, 10**9]
        for lo, hi in zip(edges[:-1], edges[1:]):
            fa = ((ga.n_det >= lo) & (ga.n_det < hi)).mean()
            fb = ((gb.n_det >= lo) & (gb.n_det < hi)).mean()
            print(f"    {lo:5d}-{hi if hi < 10**9 else '':<5}  {args.label_a}-only {fa:6.1%}   {args.label_b} {fb:6.1%}")
    print("\nA unit's cost follows its heaviest light curves, so per-oid costs of the two"
          " runs on the same oid range are only comparable if these distributions match.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
