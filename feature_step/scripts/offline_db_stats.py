#!/usr/bin/env python
"""Post-run statistics read back out of the database.

The manifests say what the run *computed* (see BHRF_RUN_RESULTS.md). This says
what the database now *holds*: the NaN rate of every feature, the class
distribution of the predictions, and the AllWISE hit rate. Read-only -- every
statement here is a SELECT.

WHY PARTITIONS ARE THE SAMPLING UNIT
------------------------------------
`feature` (32 parts, ~1.42B rows, ~132 GB) and `probability` (16 parts, ~2.29B
rows, ~429 GB) are HASH partitioned on `oid`, and neither has an index on
`version`, `feature_id` or `classifier_id`. Any per-feature or per-class
aggregate is therefore a sequential scan, and over the whole table that is a
multi-hour read.

Hash partitioning is the way out. `oid -> partition` is a hash, so partition K
is an unbiased pseudo-random 1/32 of the OBJECTS -- and, unlike TABLESAMPLE, it
holds *every* row of the objects it contains. That matters: a feature's NaN rate
is a per-object property, and a row-level sample would split one object's 73
rows across the sampled/unsampled boundary. One `feature` partition is ~600k
complete objects, which pins a percentage to well under 0.1 pt.

(TABLESAMPLE is not an option anyway: Postgres rejects it on a partitioned
parent table. It would have to be applied per partition, which is what this
does, only coarser.)

So `--partitions 1` (the default) is not a rough estimate; it is an exact count
over 1/32 of the catalogue. `--partitions 32` / `--exact` reads everything and
takes hours. `xmatch` is small (2.5 GB, unpartitioned) and is always exact.

USAGE
    python scripts/offline_db_stats.py                     # 1 partition, minutes
    python scripts/offline_db_stats.py --partitions 4      # tighter, ~4x slower
    python scripts/offline_db_stats.py --exact             # all of it, hours

Writes one CSV per section plus a markdown summary into --out-dir.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import sqlalchemy as sa
from sqlalchemy import text

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from features.offline import db  # noqa: E402
from features.offline.probability_writer import CLASSIFIER_IDS  # noqa: E402
from features.offline.xmatch import ALLWISE_CATID, SID_ZTF  # noqa: E402

FEATURE_PARTS = 32
PROBABILITY_PARTS = 16


# --------------------------------------------------------------------------- #
#  connection
# --------------------------------------------------------------------------- #
def connect(credentials: str, timeout_s: int):
    """Read-only engine with a generous statement_timeout.

    The default timeout on these servers kills a partition scan halfway, which
    looks like a hang rather than a limit -- so it is set explicitly, per
    session, to something a real scan can finish inside.
    """
    engine = db._make_engine(credentials)
    conn = engine.connect()
    conn.execute(text(f"SET statement_timeout = '{timeout_s}s'"))
    conn.execute(text("SET work_mem = '256MB'"))   # helps the hash aggregates
    return conn


def _table(schema: str, base: str, part: int | None) -> str:
    return f"{schema}.{base}" if part is None else f"{schema}.{base}_part_{part}"


def _parts(n_parts: int, requested: int) -> list:
    """Partition numbers to scan, or [None] to hit the parent table."""
    return [None] if requested >= n_parts else list(range(requested))


# --------------------------------------------------------------------------- #
#  sections
# --------------------------------------------------------------------------- #
def inventory(conn, schema: str) -> pd.DataFrame:
    """Row counts and on-disk size per table, from the planner's statistics.

    pg_stat's n_live_tup, not count(*): it is free, and at this scale an exact
    count of `probability` is a 429 GB read to confirm a number the manifests
    already state exactly.
    """
    return pd.read_sql_query(text("""
        SELECT split_part(c.relname, '_part_', 1) AS table_name,
               count(*)                            AS partitions,
               sum(s.n_live_tup)                   AS approx_rows,
               pg_size_pretty(sum(pg_total_relation_size(c.oid))) AS total_size
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        LEFT JOIN pg_stat_user_tables s ON s.relid = c.oid
        WHERE n.nspname = :schema
          AND c.relkind = 'r'
          AND split_part(c.relname, '_part_', 1)
              IN ('feature', 'probability', 'xmatch', 'object')
        GROUP BY 1 ORDER BY 1
    """), conn, params={"schema": schema})


def class_distribution(conn, schema: str, parts: list, sid: int,
                       classifier_ids: list) -> pd.DataFrame:
    """Rank-1 class counts and confidence per classifier.

    `ranking = 1` is the predicted class, so this is the class distribution of
    the run. Scoped to classifier_id: `probability` holds 2.29B rows, of which
    only ~870M are BHRF -- the stamp classifiers and LSST share the table.

    A partial index `(ranking) WHERE ranking = 1` exists on every partition, but
    it carries neither classifier_id nor class_id, so the planner still has to
    visit the heap. Whether it uses the index or seq-scans, the cost is the same
    order; the partition restriction is what actually makes this finish.
    """
    frames = []
    for part in parts:
        tbl = _table(schema, "probability", part)
        frames.append(pd.read_sql_query(text(f"""
            SELECT classifier_id, class_id,
                   count(*)                                          AS n,
                   avg(probability)                                  AS mean_prob,
                   percentile_cont(0.5) WITHIN GROUP (ORDER BY probability) AS median_prob
            FROM {tbl}
            WHERE sid = :sid AND ranking = 1
              AND classifier_id = ANY(:cids)
            GROUP BY 1, 2
        """), conn, params={"sid": sid, "cids": classifier_ids}))
    df = (pd.concat(frames, ignore_index=True)
            .groupby(["classifier_id", "class_id"], as_index=False)
            .agg(n=("n", "sum"),
                 mean_prob=("mean_prob", "mean"),
                 median_prob=("median_prob", "mean")))
    df["share_pct"] = 100 * df["n"] / df.groupby("classifier_id")["n"].transform("sum")
    return df.sort_values(["classifier_id", "n"], ascending=[True, False])


def object_and_match_counts(conn, schema: str, parts: list, sid: int,
                            catid: int) -> tuple:
    """(objects, objects with an AllWISE match) over the scanned partitions.

    One pass gives both because they share the same expensive step: the DISTINCT
    over the partition's oids. The join back to `xmatch` is cheap -- it is a
    2.5 GB table with a unique index on (oid, sid, catid).

    The object count is the denominator the NaN rate needs (see nan_per_feature),
    and the matched count is the only honest way to state an AllWISE hit rate
    from the database alone: it is measured over the objects that actually have
    features, which is the population the crossmatch mattered for.
    """
    n_objects = n_matched = 0
    for part in parts:
        tbl = _table(schema, "feature", part)
        row = conn.execute(text(f"""
            WITH objs AS (SELECT DISTINCT oid FROM {tbl} WHERE sid = :sid)
            SELECT (SELECT count(*) FROM objs) AS n_objects,
                   (SELECT count(*) FROM objs o
                      JOIN {schema}.xmatch x
                        ON x.oid = o.oid AND x.sid = :sid AND x.catid = :catid)
                       AS n_matched
        """), {"sid": sid, "catid": catid}).one()
        n_objects += row[0]
        n_matched += row[1]
    return n_objects, n_matched


def nan_per_feature(conn, schema: str, parts: list, sid: int,
                    n_objects: int) -> pd.DataFrame:
    """NaN rate per (feature, band), as MISSING ROWS over objects.

    A NaN is never stored. `prepare_ao_features_for_db` drops NaN/inf before the
    writer ever sees them (features/offline/missing_ztf/README.md), so this
    schema has no NULL `value` at all -- counting `value IS NULL` returns 0.00%
    for all 215 features and means nothing.

    What a NaN looks like here is the ABSENCE of the row. So the rate is

        nan_pct = 100 * (1 - rows_for_this_feature / objects)

    which is what the old wide `alerce.feature` measured as `value IS NULL`,
    where every (feature, band) had a row per object whether or not it had a
    value. The two are directly comparable -- a missing row here is a NULL there.

    `n_null` is kept as a guard: it must stay 0. Anything else means the schema
    started storing NULLs and this definition needs revisiting.
    """
    frames = []
    for part in parts:
        tbl = _table(schema, "feature", part)
        frames.append(pd.read_sql_query(text(f"""
            SELECT feature_id, band, version,
                   count(*)                                     AS n,
                   count(*) FILTER (WHERE value IS NULL)        AS n_null
            FROM {tbl}
            WHERE sid = :sid
            GROUP BY 1, 2, 3
        """), conn, params={"sid": sid}))
    df = (pd.concat(frames, ignore_index=True)
            .groupby(["feature_id", "band", "version"], as_index=False)
            .agg(n=("n", "sum"), n_null=("n_null", "sum")))
    df["n_objects"] = n_objects
    df["present_pct"] = 100 * df["n"] / n_objects
    df["nan_pct"] = 100 - df["present_pct"]
    return df.sort_values("nan_pct", ascending=False)


def allwise_hits(conn, schema: str, sid: int, catid: int) -> dict:
    """AllWISE link rows and match distances -- the whole table, always exact.

    `xmatch` is 2.5 GB with one row per matched oid, so there is nothing to
    sample. The distance percentiles are the check that the cone search ran at
    the radius it was supposed to: everything must sit under the 1.005" radius
    the run used.
    """
    row = conn.execute(text(f"""
        SELECT count(*)                AS n_rows,
               count(DISTINCT oid)     AS n_oids,
               min(dist)               AS dist_min,
               avg(dist)               AS dist_mean,
               max(dist)               AS dist_max,
               percentile_cont(0.50) WITHIN GROUP (ORDER BY dist) AS dist_p50,
               percentile_cont(0.90) WITHIN GROUP (ORDER BY dist) AS dist_p90,
               percentile_cont(0.99) WITHIN GROUP (ORDER BY dist) AS dist_p99
        FROM {schema}.xmatch
        WHERE sid = :sid AND catid = :catid
    """), {"sid": sid, "catid": catid}).mappings().one()
    return dict(row)


# --------------------------------------------------------------------------- #
#  name lookups
# --------------------------------------------------------------------------- #
# Same queries as db.fetch_taxonomy_maps / db.fetch_feature_name_lut, but run on
# the connection this script already holds. The db.* helpers each open their own
# connection from a pooled engine capped at one, so calling them here -- inside
# the `with conn` -- deadlocks against ourselves until the pool times out.
def taxonomy_names(conn, schema: str, classifier_ids: list) -> dict:
    """-> {(classifier_id, class_id): class_name}, straight from the DB.

    Never from a local fixture: a drifted fixture would label the counts with
    the wrong classes and nothing downstream would catch it.
    """
    rows = conn.execute(text(
        f"SELECT classifier_id, class_id, class_name FROM {schema}.taxonomy "
        "WHERE classifier_id = ANY(:cids)"), {"cids": classifier_ids}).mappings()
    return {(int(r["classifier_id"]), int(r["class_id"])): r["class_name"]
            for r in rows}


def feature_names(conn, schema: str, sid: int) -> dict:
    """-> {feature_id: feature_name} from <schema>.feature_name_lut."""
    rows = conn.execute(text(
        f"SELECT feature_id, feature_name FROM {schema}.feature_name_lut "
        "WHERE sid = :sid"), {"sid": sid}).mappings()
    return {int(r["feature_id"]): r["feature_name"] for r in rows}


# --------------------------------------------------------------------------- #
#  report
# --------------------------------------------------------------------------- #
def _md_table(df: pd.DataFrame, floatfmt: str = ".4g") -> str:
    """Render a DataFrame as a GitHub markdown table.

    Hand-rolled rather than pandas' markdown writer, which needs `tabulate` --
    an extra dependency the project does not otherwise carry, pulled in for
    nothing but formatting a report.
    """
    def cell(v):
        if v is None or (isinstance(v, float) and v != v):
            return ""
        if isinstance(v, float):
            return format(v, floatfmt)
        if isinstance(v, int) and not isinstance(v, bool):
            return f"{v:,}"
        return str(v)

    cols = [str(c) for c in df.columns]
    rows = [[cell(v) for v in rec] for rec in df.itertuples(index=False, name=None)]
    numeric = {i for i, c in enumerate(df.columns)
               if pd.api.types.is_numeric_dtype(df[c])}
    align = ["---:" if i in numeric else "---" for i in range(len(cols))]
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(align) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def _scale(part_list, n_parts) -> float:
    """Factor that turns a count over the scanned partitions into a full-table
    estimate. 1.0 when the parent table was read."""
    return 1.0 if part_list == [None] else n_parts / len(part_list)


def write_report(out_dir: Path, inv, classes, nans, wise, n_objects, n_matched,
                 feat_parts, prob_parts, timings, schema) -> Path:
    def _pct(part_list, n_parts):
        return "all" if part_list == [None] else f"{len(part_list)}/{n_parts}"

    lines = [
        "# Offline BHRF run — database statistics",
        "",
        f"Read back from `{schema}` after the run. Companion to "
        "`BHRF_RUN_RESULTS.md`, which covers what the run computed; this covers "
        "what the database holds.",
        "",
        f"- `feature` partitions scanned: **{_pct(feat_parts, FEATURE_PARTS)}**",
        f"- `probability` partitions scanned: **{_pct(prob_parts, PROBABILITY_PARTS)}**",
        "- `xmatch`: whole table (exact)",
        "",
        "Partitions are HASH on `oid`, so a scanned partition is an unbiased "
        "pseudo-random subset of objects holding *all* of each object's rows. "
        "Percentages are therefore unbiased regardless of how many were read; "
        "absolute counts over a subset are scaled back up and marked as such.",
        "",
        "## Table inventory",
        "",
        _md_table(inv),
        "",
        "Row counts are the planner's `n_live_tup` estimates, not exact counts.",
        "",
        "## Class distribution (rank-1 predictions)",
        "",
        "`ranking = 1` is the predicted class. Classifier 6 is the top-level "
        "head (Periodic / Stochastic / Transient); 7, 8 and 9 are the "
        "conditional heads under Transient, Stochastic and Periodic; 5 is the "
        "flat 21-class classifier. **The heads run on every object, not only on "
        "the ones the top level routed to them**, so a row under classifier 7 "
        "means \"if this object were a transient, it would be a TDE\" -- not "
        "that it is one. Only classifier 6 and classifier 5 read as population "
        "compositions on their own.",
        "",
        _md_table(classes),
        "",
        "## AllWISE crossmatch",
        "",
        f"- link rows (`catid={ALLWISE_CATID}`, exact, whole table): "
        f"**{wise['n_rows']:,}**",
        f"- distinct oids matched (exact): **{wise['n_oids']:,}**",
        "",
        "Hit rate among the objects that have features, measured on the scanned "
        "`feature` partitions by joining their oids to `xmatch`:",
        "",
        f"- objects with features: **{n_objects:,}**",
        f"- of those, matched to AllWISE: **{n_matched:,}**",
        f"- **hit rate: {100 * n_matched / n_objects:.2f}%**",
        "",
        "The whole-table `xmatch` count is LARGER than the number of objects "
        "with features, so a rate computed against the latter would exceed "
        "100%. That is not an inconsistency: the crossmatch runs before the "
        "classifiability check, so objects later dropped for too few real "
        "detections keep an `xmatch` row and have no features. The rate above "
        "avoids the problem by counting only objects present in both.",
        "",
        "Match distance (arcsec):",
        "",
        f"| min | p50 | p90 | p99 | max | mean |",
        f"|---:|---:|---:|---:|---:|---:|",
        f"| {wise['dist_min']:.4f} | {wise['dist_p50']:.4f} | {wise['dist_p90']:.4f} "
        f"| {wise['dist_p99']:.4f} | {wise['dist_max']:.4f} | {wise['dist_mean']:.4f} |",
        "",
        "## NaN rate per feature",
        "",
        "**A NaN is a missing row, not a NULL.** `prepare_ao_features_for_db` "
        "drops NaN/inf before the writer sees them, so `value` is never NULL in "
        "this schema and `count(value IS NULL)` returns 0.00% for every feature. "
        "The rate is therefore",
        "",
        "```",
        "nan_pct = 100 * (1 - rows_for_this_feature / objects)",
        "```",
        "",
        "which is the same quantity the old wide `alerce.feature` measured as "
        "`value IS NULL` \u2014 there every (feature, band) had a row per object "
        "whether or not it had a value, so a missing row here is a NULL there. "
        "The *definitions* match; the numbers are NOT interchangeable with the "
        "~47% in `nan_distribution/README.md`, which was measured over a "
        "different population and a different feature set. Full table in "
        "`nan_per_feature.csv`.",
        "",
        f"- objects in the scanned partitions: **{n_objects:,}** "
        f"(~{int(round(n_objects * _scale(feat_parts, FEATURE_PARTS))):,} "
        f"over all 32)",
        f"- features x bands: **{len(nans):,}**",
        f"- rows counted: **{nans['n'].sum():,}**",
        f"- rows with a NULL value (must be 0): **{nans['n_null'].sum():,}**",
        f"- mean NaN% across features: **{nans['nan_pct'].mean():.2f}%**",
        f"- features >99% NaN: **{(nans['nan_pct'] > 99).sum()}**",
        f"- features <1% NaN: **{(nans['nan_pct'] < 1).sum()}**",
        f"- median NaN% across features: **{nans['nan_pct'].median():.2f}%**",
        "",
        "(Only one overall figure is quoted because the mean across features "
        "and the row-level total are the same number by construction: "
        "mean(1 - n_f/N) == 1 - sum(n_f)/(F*N).)",
        "",
        "### 25 highest NaN rate",
        "",
        _md_table(nans.head(25), floatfmt=".2f"),
        "",
        "## Timings",
        "",
        "| section | seconds |",
        "|---|---:|",
        *[f"| {k} | {v:.1f} |" for k, v in timings.items()],
        "",
    ]
    path = out_dir / "DB_STATS.md"
    path.write_text("\n".join(lines))
    return path


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--credentials",
                    default=str(HERE.parent / "features/offline/credentials.json"),
                    help="read credentials json.")
    ap.add_argument("--schema", default=db.SCHEMA)
    ap.add_argument("--sid", type=int, default=SID_ZTF)
    ap.add_argument("--partitions", type=int, default=1,
                    help="partitions to scan per partitioned table (default 1). "
                         "Each is an unbiased 1/N sample of objects.")
    ap.add_argument("--exact", action="store_true",
                    help="scan every partition. Hours, not minutes.")
    ap.add_argument("--timeout", type=int, default=14400,
                    help="statement_timeout in seconds (default 4h).")
    ap.add_argument("--out-dir",
                    default=str(HERE.parent / "features/offline/run_stats"))
    ap.add_argument("--skip-features", action="store_true",
                    help="skip the NaN section (the most expensive scan).")
    args = ap.parse_args(argv)

    n = FEATURE_PARTS if args.exact else args.partitions
    feat_parts = _parts(FEATURE_PARTS, n)
    prob_parts = _parts(PROBABILITY_PARTS, min(n, PROBABILITY_PARTS))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timings = {}
    conn = connect(args.credentials, args.timeout)
    try:
        t = time.perf_counter()
        inv = inventory(conn, args.schema)
        timings["inventory"] = time.perf_counter() - t
        print(inv.to_string(index=False), flush=True)

        t = time.perf_counter()
        wise = allwise_hits(conn, args.schema, args.sid, ALLWISE_CATID)
        timings["xmatch"] = time.perf_counter() - t
        print(f"xmatch: {wise['n_oids']:,} oids matched "
              f"({timings['xmatch']:.1f}s)", flush=True)

        t = time.perf_counter()
        classes = class_distribution(conn, args.schema, prob_parts, args.sid,
                                     CLASSIFIER_IDS)
        timings["probability"] = time.perf_counter() - t
        print(f"probability: {len(classes)} (classifier, class) pairs "
              f"({timings['probability']:.1f}s)", flush=True)

        id_to_name = taxonomy_names(conn, args.schema, CLASSIFIER_IDS)
        classes.insert(2, "class_name",
                       [id_to_name.get((int(r.classifier_id), int(r.class_id)), "?")
                        for r in classes.itertuples()])
        classes.to_csv(out_dir / "class_distribution.csv", index=False)

        t = time.perf_counter()
        n_objects, n_matched = object_and_match_counts(
            conn, args.schema, feat_parts, args.sid, ALLWISE_CATID)
        timings["objects"] = time.perf_counter() - t
        print(f"objects: {n_objects:,} with features, {n_matched:,} with an "
              f"AllWISE match ({100 * n_matched / n_objects:.2f}%) "
              f"({timings['objects']:.1f}s)", flush=True)

        nans = pd.DataFrame(columns=["feature_id", "band", "version", "n",
                                     "n_null", "n_objects", "present_pct",
                                     "nan_pct"])
        if not args.skip_features:
            t = time.perf_counter()
            nans = nan_per_feature(conn, args.schema, feat_parts, args.sid,
                                   n_objects)
            timings["feature"] = time.perf_counter() - t
            lut = feature_names(conn, args.schema, args.sid)
            nans.insert(1, "feature_name",
                        nans["feature_id"].map(lambda i: lut.get(int(i), "?")))
            nans.to_csv(out_dir / "nan_per_feature.csv", index=False)
            print(f"feature: {len(nans)} (feature, band) pairs, "
                  f"{nans['n'].sum():,} rows ({timings['feature']:.1f}s)",
                  flush=True)
    finally:
        conn.close()
        db.dispose_engines()

    inv.to_csv(out_dir / "table_inventory.csv", index=False)
    (out_dir / "allwise.json").write_text(
        json.dumps({k: (float(v) if v is not None else None)
                    for k, v in wise.items()}
                   | {"n_objects_with_features": n_objects,
                      "n_objects_matched": n_matched},
                   indent=2))
    path = write_report(out_dir, inv, classes, nans, wise, n_objects,
                        n_matched, feat_parts, prob_parts, timings, args.schema)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
