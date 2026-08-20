#!/usr/bin/env python
"""Sample multisurvey oids that ALSO have stored legacy BHRF probabilities.

Only ~2.4M of the 130M ZTF objects carry `lc_classifier_BHRF_forced_phot`
probabilities in `alerce.probability` (version 2.1.0), so sampling the catalog
uniformly yields almost nothing comparable. This samples from the CLASSIFIED
side instead (TABLESAMPLE over the small `_top` partition), maps the ZTF string
oids to multisurvey bigints, joins `multisurvey_ztf.object` for n_det, and
stratifies by n_det.

Also records `alerce.object.ndet/lastmjd` next to the multisurvey values: a
comparison is only sound when multisurvey holds the object's COMPLETE light
curve, so downstream analysis must split on that.

Writes a JSON sample file consumed by offline_compare_probabilities_batch.py.

    python feature_step/scripts/offline_sample_classified_oids.py --n 100
"""
import argparse
import json
import sys
from pathlib import Path

PIPE = Path(__file__).resolve().parents[2]   # .../pipeline
for p in (PIPE / "feature_step", PIPE / "lc_classifier", PIPE / "libs" / "idmapper"):
    sys.path.insert(0, str(p))

import numpy as np
import pandas as pd
from sqlalchemy import text

from features.offline import db
from idmapper.mapper import catalog_oid_to_masterid

DEFAULT_CREDENTIALS = str(PIPE / "feature_step" / "features" / "offline" / "credentials.json")
# Smallest BHRF partition (3 classes/object) -> cheapest source of classified oids.
TOP_PARTITION = "alerce.lc_classifier_bhrf_forced_phot_top"
# n_det strata edges (lower inclusive, upper exclusive); last is open-ended.
STRATUM_EDGES = [(1, 20), (20, 100), (100, 300), (300, 600), (600, 1200), (1200, 10**9)]


def _stratum_label(lo, hi):
    return f"{lo}-{hi}" if hi < 10**9 else f"{lo}+"


def sample_classified_oids(credentials, version, pool_pct, seed):
    """TABLESAMPLE the BHRF top partition -> distinct ZTF string oids."""
    engine = db._make_engine(credentials)
    query = text(f"""
        SELECT DISTINCT oid
        FROM {TOP_PARTITION} TABLESAMPLE SYSTEM (:pct) REPEATABLE (:seed)
        WHERE classifier_version = :version
    """)
    with engine.connect() as conn:
        df = pd.read_sql_query(query, conn,
                               params={"pct": pool_pct, "seed": seed, "version": version})
    return df["oid"].tolist()


def enrich_with_object_rows(credentials, ztf_oids):
    """Map ZTF oids -> bigints, then join multisurvey_ztf.object + alerce.object."""
    pairs = []
    for z in ztf_oids:
        try:
            pairs.append((int(catalog_oid_to_masterid("ZTF", z, True)), z))
        except Exception:  # noqa: BLE001 - non-ZTF-shaped oid strings are skipped
            continue
    if not pairs:
        return pd.DataFrame()
    bigints = [p[0] for p in pairs]
    ztf_by_bigint = dict(pairs)

    engine = db._make_engine(credentials)
    with engine.connect() as conn:
        ms = pd.read_sql_query(
            text(f"""
                SELECT oid, n_det, meanra, meandec, lastmjd, firstmjd
                FROM {db.SCHEMA}.object
                WHERE sid = :sid AND oid = ANY(:oids)
            """),
            conn, params={"sid": db.SID, "oids": bigints},
        )
        al = pd.read_sql_query(
            text("""
                SELECT oid AS ztf_oid, ndet AS alerce_ndet,
                       lastmjd AS alerce_lastmjd, firstmjd AS alerce_firstmjd
                FROM alerce.object
                WHERE oid = ANY(:oids)
            """),
            conn, params={"oids": [p[1] for p in pairs]},
        )
    if ms.empty:
        return ms
    ms["ztf_oid"] = ms["oid"].map(ztf_by_bigint)
    return ms.merge(al, on="ztf_oid", how="left")


def stratify(df, n_total, seed):
    """Pick ~n_total rows spread evenly across the n_det strata."""
    rng = np.random.default_rng(seed)
    per = max(1, n_total // len(STRATUM_EDGES))
    picked = []
    for lo, hi in STRATUM_EDGES:
        band = df[(df["n_det"] >= lo) & (df["n_det"] < hi)]
        if band.empty:
            continue
        take = min(per, len(band))
        idx = rng.choice(band.index.to_numpy(), size=take, replace=False)
        sub = df.loc[idx].copy()
        sub["stratum"] = _stratum_label(lo, hi)
        picked.append(sub)
    if not picked:
        return pd.DataFrame()
    out = pd.concat(picked, ignore_index=True)
    # Top up from the leftover pool if strata were thin, keeping the sample at n_total.
    if len(out) < n_total:
        rest = df[~df["oid"].isin(out["oid"])]
        if not rest.empty:
            take = min(n_total - len(out), len(rest))
            idx = rng.choice(rest.index.to_numpy(), size=take, replace=False)
            extra = df.loc[idx].copy()
            extra["stratum"] = "topup"
            out = pd.concat([out, extra], ignore_index=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=100, help="oids to sample (default 100)")
    ap.add_argument("--version", default="2.1.0", help="stored classifier_version")
    ap.add_argument("--pool-pct", type=float, default=0.3,
                    help="TABLESAMPLE percent of the top partition (default 0.3)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--credentials", default=DEFAULT_CREDENTIALS)
    ap.add_argument("--out", default=str(PIPE / "feature_step" / "features" / "offline"
                                         / "oids" / "compare_sample.json"))
    args = ap.parse_args()

    print(f"TABLESAMPLE {args.pool_pct}% of {TOP_PARTITION} (version={args.version})...")
    ztf_oids = sample_classified_oids(args.credentials, args.version, args.pool_pct, args.seed)
    print(f"  pool: {len(ztf_oids)} classified ZTF oids")
    if not ztf_oids:
        sys.exit("error: TABLESAMPLE returned no oids; raise --pool-pct")

    print("Joining multisurvey_ztf.object + alerce.object...")
    df = enrich_with_object_rows(args.credentials, ztf_oids)
    print(f"  {len(df)} of the pool are present in {db.SCHEMA}.object")
    if df.empty:
        sys.exit("error: none of the sampled oids are in multisurvey_ztf.object")

    print("\nn_det distribution of the classified pool:")
    print(df["n_det"].describe().to_string())

    sample = stratify(df, args.n, args.seed)
    print(f"\nstratified sample: {len(sample)} oids")
    print(sample.groupby("stratum")["n_det"].agg(["count", "min", "median", "max"]).to_string())

    # LC completeness: multisurvey lastmjd vs the live alerce lastmjd.
    both = sample["alerce_lastmjd"].notna()
    if both.any():
        d = (sample.loc[both, "lastmjd"] - sample.loc[both, "alerce_lastmjd"]).abs()
        print(f"\nLC completeness (|ms.lastmjd - alerce.lastmjd|): "
              f"<=1d: {(d <= 1).sum()}/{both.sum()}, median gap {d.median():.1f}d")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sample.to_json(out, orient="records", indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
