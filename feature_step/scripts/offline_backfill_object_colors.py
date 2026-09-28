#!/usr/bin/env python
"""Backfill ztf_object's colour columns from the feature rows an offline run wrote.

The live feature step sends an `update-ztf-object-features` scribe command per
object, and the scribe sets g_r_max, g_r_mean, g_r_max_corr and g_r_mean_corr on
`ztf_object` from the band-12 features of the same name. The offline run
(offline_run_batch.py --load-db) writes `feature`, `probability` and `xmatch`
only, so every object it processed still carries whatever those four columns
held before. This copies them across, in oid ranges, without recomputing
anything. Details and the SQL: features/offline/object_writer.py.

    # plan only: ranges, resolved feature ids, the SQL. Connects read-only.
    python scripts/offline_backfill_object_colors.py \
        --oid-file features/offline/oids/run.npy --out-dir $RUN/object_colors

    # first pass: two ranges, then check a few objects by hand
    python scripts/offline_backfill_object_colors.py ... --execute --max-chunks 2

    # the rest. Same command; finished ranges are skipped via progress.jsonl.
    python scripts/offline_backfill_object_colors.py ... --execute

Run it under tmux like the run itself. One transaction per range, so an
interruption loses at most the range in flight, and the SAME command resumes.
The oid array defines the ranges, so a different --oid-file (e.g. the tail
run's) needs a fresh --out-dir, exactly like offline_run_batch.py.

The write account needs UPDATE on <schema>.ztf_object on top of the SELECT the
run already required; setup checks the latter, not the former.
"""
import argparse
import sys
import time
from pathlib import Path

from sqlalchemy import text

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from features.offline import db, object_writer  # noqa: E402
from features.offline.xmatch import SID_ZTF  # noqa: E402
from offline_run_batch import load_oids  # noqa: E402


def connect(credentials: str, timeout_s: int):
    """A connection with an explicit statement_timeout: a range that stalls
    should fail loudly, not hang the tmux session for a day."""
    engine = db._make_engine(credentials)
    conn = engine.connect()
    conn.execute(text(f"SET statement_timeout = '{timeout_s}s'"))
    conn.commit()
    return conn


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oid-file", required=True,
                    help=".npy or text oid list; the run's own run.npy.")
    ap.add_argument("--credentials", required=True,
                    help="WRITE credentials json (UPDATE on ztf_object).")
    ap.add_argument("--schema", default=db.SCHEMA)
    ap.add_argument("--sid", type=int, default=SID_ZTF)
    ap.add_argument("--chunk-size", type=int, default=100_000,
                    help="oids per range / per transaction (default 100k).")
    ap.add_argument("--out-dir", required=True,
                    help="holds progress.jsonl; one per oid array.")
    ap.add_argument("--execute", action="store_true",
                    help="actually UPDATE. Without it: plan only.")
    ap.add_argument("--max-chunks", type=int, default=None,
                    help="apply at most this many new ranges, then stop.")
    ap.add_argument("--timeout", type=int, default=3600,
                    help="statement_timeout per range in seconds (default 1h).")
    args = ap.parse_args(argv)

    oids = load_oids(args.oid_file)
    ranges = object_writer.make_ranges(oids, args.chunk_size)
    print(f"{len(oids):,} oid(s) -> {len(ranges)} range(s) of <= {args.chunk_size:,}",
          flush=True)

    lut = db.fetch_feature_name_lut(args.credentials, sid=args.sid, schema=args.schema)
    ids = object_writer.resolve_color_feature_ids(lut)
    print("feature ids: " + ", ".join(f"{c}={i}" for c, i in ids.items()), flush=True)

    sql = object_writer.build_backfill_sql(args.schema, ids)
    out_dir = Path(args.out_dir)
    progress = out_dir / "progress.jsonl"

    if not args.execute:
        print(sql)
        summary = object_writer.run_backfill(None, sql, ranges, args.sid, progress,
                                             execute=False)
        print(f"dry run: {summary['ranges']} range(s) would be updated; "
              "add --execute to apply", flush=True)
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    conn = connect(args.credentials, args.timeout)
    t0 = time.perf_counter()
    try:
        summary = object_writer.run_backfill(conn, sql, ranges, args.sid, progress,
                                             execute=True, max_chunks=args.max_chunks)
    finally:
        conn.close()
        db.dispose_engines()
    print(f"done: {summary['updated']:,} object(s) updated, "
          f"{summary['skipped']} range(s) already done, "
          f"{time.perf_counter() - t0:.0f}s; progress in {progress}", flush=True)
    return 0


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    sys.exit(main())
