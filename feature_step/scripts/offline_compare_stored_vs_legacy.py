#!/usr/bin/env python
"""Rank-1 agreement between the offline run's stored probabilities and legacy's.

Both sides come from the database, so this is a join, not a recomputation. The
older comparator (offline_compare_probabilities_batch.py) reruns the whole
pipeline per object to build the offline side, which is why that study covered
100 oids; once a batch run has loaded its output with --load-db, the same study
runs over every object the run touched.

The two tables do not share a vocabulary. Ours is keyed by bigint oid with
smallint classifier_id/class_id; alerce.probability is keyed by ZTF string oid
with text classifier_name/class_name. Both translations happen here: class_id
through <schema>.taxonomy (the authority, not the fixture), oid through idmapper.

    MODEL_PATH is not needed -- nothing is classified here.

    poetry run python scripts/offline_compare_stored_vs_legacy.py \
        --oid-file $RUN/oids/sample63k.npy
"""
import argparse
import json
import sys
from pathlib import Path

PIPE = Path(__file__).resolve().parents[2]
for p in (PIPE / "feature_step", PIPE / "lc_classifier", PIPE / "libs" / "idmapper"):
    sys.path.insert(0, str(p))

import numpy as np
import pandas as pd
from sqlalchemy import text

from features.offline import db
from features.offline.probability_compare import (
    CLASSIFIER_NAME_BY_ID, BHRF_CLASSIFIER_NAMES, rank1_agreement, _rank1)
from features.offline.probability_writer import CLASSIFIER_IDS, classifier_version_to_smallint
from idmapper.mapper import decode_masterid

DEFAULT_CREDENTIALS = str(PIPE / "feature_step" / "features" / "offline" / "credentials.json")
# Short names for the report; the DB carries the long lc_classifier_BHRF_* ones.
SHORT = {5: "flat", 6: "top", 7: "transient", 8: "stochastic", 9: "periodic"}


def load_oids(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        return np.load(path)
    return np.loadtxt(path, dtype=np.int64, ndmin=1)


def fetch_ours(credentials, oids, version_smallint, schema):
    """Our rank-1 class per (oid, classifier), reduced chunk by chunk.

    Reduced inside the loop on purpose: the full frame is ~45 rows per object,
    which for a 63k-oid probe is 2.8M rows held only to throw away 44 of every
    45.
    """
    engine = db._make_engine(credentials)
    tax = db.fetch_taxonomy_maps(credentials, CLASSIFIER_IDS, schema=schema)
    name_of = {cid: {v: k for k, v in m.items()} for cid, m in tax.items()}

    sql = text(f"""
        SELECT oid, classifier_id, class_id, probability
        FROM {schema}.probability
        WHERE oid = ANY(:oids) AND classifier_version = :ver
    """)
    out = []
    for chunk in np.array_split(oids, max(1, len(oids) // 5000)):
        with engine.connect() as conn:
            df = pd.read_sql_query(sql, conn, params={
                "oids": [int(o) for o in chunk], "ver": int(version_smallint)})
        if df.empty:
            continue
        df["classifier_name"] = df["classifier_id"].map(SHORT)
        df["class_name"] = [name_of[c][i] for c, i in
                            zip(df["classifier_id"], df["class_id"])]
        df["ranking"] = 0   # unused; rank 1 is derived from probability
        out.append(_rank1(df[["oid", "classifier_name", "class_name",
                              "probability", "ranking"]], "ours"))
    if not out:
        return pd.DataFrame(columns=["oid", "classifier_name", "class_ours", "prob_ours"])
    return pd.concat(out, ignore_index=True)


def fetch_legacy(credentials, ztf_by_oid, version):
    """Legacy rank-1 class per (oid, classifier), keyed back to the bigint oid."""
    engine = db._make_engine(credentials)
    long_to_short = {CLASSIFIER_NAME_BY_ID[i]: SHORT[i] for i in CLASSIFIER_IDS}
    sql = text(f"""
        SELECT oid AS ztf_oid, classifier_name, class_name, probability
        FROM {db.ALERCE_SCHEMA}.probability
        WHERE oid = ANY(:oids) AND classifier_version = :ver
          AND classifier_name = ANY(:names)
    """)
    items = list(ztf_by_oid.items())
    out = []
    for i in range(0, len(items), 5000):
        block = items[i:i + 5000]
        back = {z: o for o, z in block}
        with engine.connect() as conn:
            df = pd.read_sql_query(sql, conn, params={
                "oids": [z for _, z in block], "ver": version,
                "names": BHRF_CLASSIFIER_NAMES})
        if df.empty:
            continue
        df["oid"] = df["ztf_oid"].map(back)
        df["classifier_name"] = df["classifier_name"].map(long_to_short)
        df["ranking"] = 0
        out.append(_rank1(df[["oid", "classifier_name", "class_name",
                              "probability", "ranking"]], "legacy"))
    if not out:
        return pd.DataFrame(columns=["oid", "classifier_name", "class_legacy", "prob_legacy"])
    return pd.concat(out, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oid-file", required=True, help=".npy or newline .txt of bigint oids")
    ap.add_argument("--version", default="2.1.0", help="classifier version, both sides")
    ap.add_argument("--credentials", default=DEFAULT_CREDENTIALS)
    ap.add_argument("--schema", default=db.SCHEMA)
    ap.add_argument("--json-out", default=None, help="write the per-oid frame here (parquet)")
    args = ap.parse_args()

    oids = load_oids(args.oid_file)
    print(f"oids en el archivo   : {len(oids):,}")

    ours = fetch_ours(args.credentials, oids, classifier_version_to_smallint(args.version),
                      args.schema)
    n_ours = ours["oid"].nunique() if len(ours) else 0
    print(f"clasificados por nosotros: {n_ours:,}")
    if not n_ours:
        print("nada que comparar: la corrida no dejo filas para estos oids.")
        return 1

    ztf_by_oid = {int(o): decode_masterid(np.int64(o))[1] for o in ours["oid"].unique()}
    legacy = fetch_legacy(args.credentials, ztf_by_oid, args.version)
    print(f"con probabilidad legacy  : {legacy['oid'].nunique() if len(legacy) else 0:,}")

    # rank1_agreement re-reduces; both sides are already one row per pair, so the
    # reduction is a no-op and the frames just need their long-form column names.
    per_oid, summary = rank1_agreement(
        ours.rename(columns={"class_ours": "class_name", "prob_ours": "probability"})
            .assign(ranking=0),
        legacy.rename(columns={"class_legacy": "class_name", "prob_legacy": "probability"})
              .assign(ranking=0),
    )

    print(f"\n{'clasificador':<12} {'comparados':>11} {'coinciden':>10} {'tasa':>8}")
    for cid in CLASSIFIER_IDS:
        s = summary["by_classifier"].get(SHORT[cid])
        if s:
            print(f"{SHORT[cid]:<12} {s['n_both']:>11,} {s['n_agree']:>10,} "
                  f"{s['rate'] * 100:>7.1f}%")
    print(f"\nsolo nosotros: {summary['n_only_ours']:,}   "
          f"solo legacy: {summary['n_only_legacy']:,}")

    flat = per_oid[(per_oid["classifier_name"] == "flat") & (~per_oid["agree"])]
    if len(flat):
        print(f"\ntop 15 desacuerdos del clasificador plano ({len(flat):,} en total)")
        pairs = (flat.groupby(["class_legacy", "class_ours"]).size()
                 .sort_values(ascending=False).head(15))
        for (leg, our), n in pairs.items():
            print(f"  {n:>7,}  legacy {leg:<16} -> nuestro {our}")

    if args.json_out:
        per_oid.to_parquet(args.json_out)
        print(f"\nper-oid escrito en {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
