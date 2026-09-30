#!/usr/bin/env python
"""Did offline_backfill_object_colors.py leave ztf_object in step with feature?

Samples oids from the array the backfill ran over, rebuilds the four colours
each object should carry from its `feature` rows (in pandas, not with the
backfill's SQL) and compares them with `ztf_object`. Reads only, and only by
oid. Exit 0 when nothing mismatches.

    # the whole array
    poetry run python scripts/offline_verify_object_colors.py \
        --oid-file $RUN/oids/all_classified.npy \
        --credentials $CREDS

    # one block of it, e.g. a range progress.jsonl reports few updates for
    poetry run python scripts/offline_verify_object_colors.py ... \
        --lo 36028989782755966 --hi 36028989783066486

The run writes no feature row for a NaN value, so an object only has colour
rows when it has both g and r; the rest the backfill leaves untouched. The band
table says which of the two a block with few updates is: no feature rows at
all (unclassifiable), or rows in a single band.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from features.offline import db, object_writer  # noqa: E402
from offline_run_batch import load_oids  # noqa: E402


def expected_colors(feature_rows: pd.DataFrame, ids: dict) -> pd.DataFrame:
    """One row per object with colour rows, one column per ztf_object colour.

    Same rule as the backfill: only the rows carrying the object's most recent
    updated_date among its colour rows count; an older colour becomes NaN.
    """
    rows = feature_rows[(feature_rows.band == object_writer.PAIR_BAND)
                        & feature_rows.feature_id.isin(list(ids.values()))]
    rows = rows[rows.updated_date == rows.groupby("oid").updated_date.transform("max")]
    return (rows.pivot(index="oid", columns="feature_id", values="value")
                .rename(columns={fid: col for col, fid in ids.items()})
                .reindex(columns=list(ids)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oid-file", required=True,
                    help="the array the backfill ran over.")
    ap.add_argument("--credentials", required=True,
                    help="credentials json; SELECT is enough.")
    ap.add_argument("--schema", default=db.SCHEMA)
    ap.add_argument("--sid", type=int, default=db.SID)
    ap.add_argument("--lo", type=int, default=None, help="sample only oids >= lo.")
    ap.add_argument("--hi", type=int, default=None, help="sample only oids <= hi.")
    ap.add_argument("--sample", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    oids = load_oids(args.oid_file)
    if args.lo is not None:
        oids = oids[oids >= args.lo]
    if args.hi is not None:
        oids = oids[oids <= args.hi]
    if len(oids) == 0:
        print("no oid of the array falls in [lo, hi]", file=sys.stderr)
        return 2
    rng = np.random.default_rng(args.seed)
    sample = [int(x) for x in rng.choice(oids, min(args.sample, len(oids)), replace=False)]

    ids = object_writer.resolve_color_feature_ids(
        db.fetch_feature_name_lut(args.credentials, sid=args.sid, schema=args.schema))
    cols = list(ids)

    engine = db._make_engine(args.credentials)
    params = {"oids": sample, "sid": args.sid}
    with engine.connect() as conn:
        feat = pd.read_sql(
            text(f"SELECT oid, feature_id, band, value, updated_date "
                 f"FROM {args.schema}.feature WHERE oid = ANY(:oids) AND sid = :sid"),
            conn, params=params)
        obj = pd.read_sql(
            text(f"SELECT oid, n_det FROM {args.schema}.object "
                 "WHERE oid = ANY(:oids) AND sid = :sid"),
            conn, params=params)
        got = pd.read_sql(
            text(f"SELECT oid, {', '.join(cols)} FROM {args.schema}.ztf_object "
                 "WHERE oid = ANY(:oids)"),
            conn, params={"oids": sample}).set_index("oid")
    db.dispose_engines()

    exp = expected_colors(feat, ids)
    both = exp.index.intersection(got.index)
    # ztf_object's columns are REAL and feature.value is double: compare at
    # float4 precision, NULL == NULL.
    same = np.isclose(exp.loc[both].to_numpy(dtype="float64"),
                      got.loc[both, cols].to_numpy(dtype="float64"),
                      rtol=1e-5, atol=1e-6, equal_nan=True)
    bad = ~same.all(axis=1)
    no_object = len(exp.index.difference(got.index))

    print(f"oids to sample from          : {len(oids):,}  (sampled {len(sample):,})")
    print(f"with any feature row         : {feat.oid.nunique():,}")
    print("\nn_det of the sample:")
    print(obj.n_det.describe().round(1).to_string())
    print("\nbands that have feature rows, per object:")
    bands = feat.groupby("oid").band.agg(lambda s: ",".join(map(str, sorted(set(s)))))
    print(bands.value_counts().to_string())

    print(f"\nwith colour rows in feature  : {len(exp):,}")
    print(f"  no ztf_object row          : {no_object:,}")
    print(f"  all 4 colours match        : {int((~bad).sum()):,}")
    print(f"  MISMATCH                   : {int(bad.sum()):,}")
    if len(exp):
        color_rows = feat[(feat.band == object_writer.PAIR_BAND)
                          & feat.feature_id.isin(list(ids.values()))]
        print("\nupdated_date of the colour rows (objects):")
        print(color_rows.groupby("updated_date").oid.nunique().to_string())
        print("\nNULL fraction per colour in ztf_object:")
        print(got.loc[both, cols].isna().mean().round(3).to_string())
    if bad.any():
        print("\nfirst mismatches (expected from feature | ztf_object):")
        for oid in both[bad][:10]:
            print(oid, exp.loc[oid].round(5).tolist(), "|",
                  got.loc[oid, cols].round(5).tolist())
    return 1 if bad.any() or no_object else 0


if __name__ == "__main__":
    raise SystemExit(main())
