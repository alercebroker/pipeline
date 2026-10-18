#!/usr/bin/env python
"""Validate offline_reclassify_batch.py: its shard vs the probabilities the DB already holds.

Run the reclassifier with the SAME model the stored rows came from (squidward
2.1.0, classifier_version 210) over a few hundred oids, without --load-db, and
hand this script the shard it wrote. If the stored feature rows plus the model
reproduce the stored probabilities, the whole chain (feature read, newest-date
filter, LUT-name translation, frame, model call, per-oid split, row building)
is validated end to end before the 2.2.0 run writes anything.

Prints labeled lines and exits 1 on any difference beyond --tol. The DB column
is a float32 `real`, so a tolerance of 1e-6 absorbs the storage rounding;
rankings are compared exactly and lastmjd exactly.

    python scripts/offline_reclassify_compare.py \\
        $RUN/reclass_210_check/probabilities/unit_0000000.parquet \\
        --credentials features/offline/credentials.json --schema multisurvey
"""
import argparse  # CLI
import sys  # path and exit code
from pathlib import Path  # paths

PIPE = Path(__file__).resolve().parents[2]  # .../pipeline
for _p in (PIPE / "feature_step", PIPE / "lc_classifier", PIPE / "libs" / "idmapper",
           PIPE / "libs" / "apf"):  # enough to import features.offline.db
    sys.path.insert(0, str(_p))  # importable without an install

import pandas as pd  # frames
from sqlalchemy import text  # bound query

from features.offline import db  # engine and oid casting

KEY = ["oid", "classifier_id", "class_id"]  # one probability row per key


def compare_rows(shard: pd.DataFrame, stored: pd.DataFrame, tol: float) -> dict:
    """Shard rows vs stored rows on (oid, classifier_id, class_id): counts and worst differences.

    Both frames carry oid, classifier_id, class_id, probability, ranking, lastmjd.
    Returns a dict of labeled numbers; `verdict` is "ok" when every shard row has
    a stored twin within `tol` on probability and equal on ranking and lastmjd.
    """
    merged = shard.merge(stored, on=KEY, how="left", suffixes=("", "_db"))  # every shard row, its twin or NaN
    missing = merged["probability_db"].isna()  # shard rows with no stored row
    both = merged[~missing]  # rows that can be compared
    dprob = (both["probability"] - both["probability_db"]).abs()  # per-row difference
    out = {
        "oids_compared": int(shard["oid"].nunique()),  # objects in the shard
        "rows_shard": int(len(shard)),  # rows the reclassifier built
        "rows_db": int(len(stored)),  # rows the DB holds for those oids
        "missing_in_db": int(missing.sum()),  # shard rows without a stored twin
        "extra_in_db": int(len(stored) - len(both)),  # stored rows the shard did not produce
        "max_abs_dprob": float(dprob.max()) if len(both) else 0.0,  # worst probability gap
        "rows_over_tol": int((dprob > tol).sum()),  # rows beyond the tolerance
        "rank_mismatches": int((both["ranking"] != both["ranking_db"]).sum()),  # rankings must be identical
        "lastmjd_mismatches": int((both["lastmjd"] != both["lastmjd_db"]).sum()),  # copied, so identical
    }
    clean = (out["missing_in_db"] == 0 and out["extra_in_db"] == 0 and out["rows_over_tol"] == 0
             and out["rank_mismatches"] == 0 and out["lastmjd_mismatches"] == 0)  # every check
    out["verdict"] = "ok" if clean else "DIFFERENT"  # the one line to read
    return out


def fetch_stored(credentials: str, oids: list, version: int, schema: str) -> pd.DataFrame:
    """Stored probability rows of `oids` with `classifier_version` = version, by primary key."""
    query = text(f"""
        SELECT oid, classifier_id, class_id, probability, ranking, lastmjd
        FROM {schema}.probability
        WHERE oid = ANY(:oids) AND sid = :sid AND classifier_version = :version
    """)  # PK prefix (oid, sid): index lookups; the version filter is on the fetched rows
    with db._make_engine(credentials).connect() as conn:  # read account
        return pd.read_sql_query(query, conn, params={"oids": db._py_oids(oids), "sid": db.SID,
                                                      "version": version})  # plain ints


def main():
    """Read the shard, fetch its oids' stored rows, print the comparison, exit 1 on a difference."""
    ap = argparse.ArgumentParser(description="Compare a reclassify shard with the stored probabilities.")
    ap.add_argument("shard", help="probabilities/unit_*.parquet written by offline_reclassify_batch.py")
    ap.add_argument("--credentials", required=True, help="credentials JSON (read is enough)")
    ap.add_argument("--schema", default="multisurvey")  # where the stored rows are
    ap.add_argument("--version", type=int, default=210,
                    help="classifier_version of the stored rows to compare against")  # 2.1.0 -> 210
    ap.add_argument("--tol", type=float, default=1e-6,
                    help="max |probability difference| accepted (real is float32)")  # storage rounding
    args = ap.parse_args()  # the flags

    shard = pd.read_parquet(args.shard)  # what the reclassifier built
    if len(shard) == 0:  # nothing to compare
        print("verdict: EMPTY shard")  # say so
        return 1  # a failure
    stored = fetch_stored(args.credentials, sorted(shard["oid"].unique().tolist()),
                          args.version, args.schema)  # their stored twins
    result = compare_rows(shard, stored, args.tol)  # the numbers
    for key, value in result.items():  # one labeled line each
        print(f"{key}: {value}")
    return 0 if result["verdict"] == "ok" else 1  # exit code


if __name__ == "__main__":  # script entry point
    sys.exit(main())  # the exit code
