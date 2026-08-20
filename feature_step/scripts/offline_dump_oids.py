#!/usr/bin/env python
"""Dump every oid of <schema>.object to .npy, one partition at a time, in parallel.

`multisurvey_ztf.object` is a hash-partitioned table (~130M rows / 16 GB of heap).
We only want the `oid` column (bigint), so this script takes the two shortcuts
that matter:

  * `COPY ... TO STDOUT (FORMAT binary)` instead of a normal query — the extended
    query protocol spends 11 bytes of DataRow header per row, COPY binary spends
    6 (Int16 field count + Int32 field length). 14 B/row vs 28 B/row in text mode.
  * one connection per partition, run concurrently — the partitions are
    independent, so the server-side scans parallelize almost linearly and the
    run ends up bounded by network bandwidth rather than by the DB.

The PK `object_part_N_pkey` is a btree on (oid, sid), so `SELECT oid` is
index-only-scannable. Whether the planner *uses* it depends on the visibility
map; partitions that were not recently vacuumed fall back to a heap scan. Run
with --explain to see what the planner actually picked per partition.

Run from the pipeline root:

    python scripts/offline_dump_oids.py --dry-run          # estimate only, no transfer
    python scripts/offline_dump_oids.py --limit 1000000    # calibration run
    python scripts/offline_dump_oids.py --jobs 8 --merge   # the real thing
"""
import argparse
import concurrent.futures as cf
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import psycopg2

PIPE = Path(__file__).resolve().parents[2]  # .../pipeline
DEFAULT_CREDENTIALS = str(PIPE / "feature_step" / "features" / "offline" / "credentials.json")
DEFAULT_OUT = str(PIPE / "feature_step" / "features" / "offline" / "oids")

# Binary COPY framing (see PostgreSQL docs, "Binary Format"):
#   header  = 11-byte signature + Int32 flags + Int32 header-extension length
#   tuple   = Int16 field count + per field (Int32 length + payload)
#   trailer = Int16 -1
COPY_HEADER_LEN = 19
COPY_TRAILER_LEN = 2
# Packed, big-endian: numpy defaults to align=False so itemsize is exactly 14.
TUPLE_DTYPE = np.dtype([("nfields", ">i2"), ("length", ">i4"), ("oid", ">i8")])
assert TUPLE_DTYPE.itemsize == 14


def load_credentials(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        p = json.load(f)
    if not p.get("password"):
        sys.exit(f"error: {path} has no password set")
    return p


def connect(creds: dict):
    return psycopg2.connect(
        host=creds["host"], port=creds.get("port", 5432),
        dbname=creds["dbname"], user=creds["user"], password=creds["password"],
    )


def list_partitions(conn, schema: str, table: str) -> list:
    """Return [(name, est_rows, heap_bytes, pct_all_visible), ...] for each partition.

    Falls back to the table itself when it is not partitioned.
    """
    sql = """
        SELECT c.relname,
               c.reltuples::bigint,
               pg_relation_size(c.oid),
               COALESCE(round(100.0 * c.relallvisible / NULLIF(c.relpages, 0), 1), 0)
        FROM pg_class c
        JOIN pg_namespace n  ON n.oid = c.relnamespace
        JOIN pg_inherits i   ON i.inhrelid = c.oid
        JOIN pg_class parent ON parent.oid = i.inhparent
        WHERE n.nspname = %s AND parent.relname = %s
        ORDER BY c.relname
    """
    with conn.cursor() as cur:
        cur.execute(sql, (schema, table))
        parts = cur.fetchall()
        if parts:
            return [(r[0], r[1], r[2], float(r[3])) for r in parts]
        # Not partitioned — treat the table as its own single "partition".
        cur.execute(
            """SELECT c.relname, c.reltuples::bigint, pg_relation_size(c.oid),
                      COALESCE(round(100.0*c.relallvisible/NULLIF(c.relpages,0),1), 0)
               FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
               WHERE n.nspname = %s AND c.relname = %s""",
            (schema, table),
        )
        row = cur.fetchone()
        if not row:
            sys.exit(f"error: {schema}.{table} not found")
        return [(row[0], row[1], row[2], float(row[3]))]


def _select_sql(schema: str, part: str, sid, limit) -> str:
    sql = f'SELECT oid FROM "{schema}"."{part}"'
    if sid is not None:
        sql += f" WHERE sid = {int(sid)}"
    if limit:
        sql += f" LIMIT {int(limit)}"
    return sql


def explain_partition(conn, schema: str, part: str, sid, limit) -> str:
    with conn.cursor() as cur:
        cur.execute("EXPLAIN " + _select_sql(schema, part, sid, limit))
        return cur.fetchone()[0].strip()


def parse_copy_binary(path: Path) -> np.ndarray:
    """Parse a single-bigint-column binary COPY dump into an int64 array."""
    size = path.stat().st_size
    body = size - COPY_HEADER_LEN - COPY_TRAILER_LEN
    if body < 0 or body % TUPLE_DTYPE.itemsize:
        raise ValueError(
            f"{path.name}: unexpected COPY payload ({size} bytes; "
            f"body {body} not a multiple of {TUPLE_DTYPE.itemsize})"
        )
    count = body // TUPLE_DTYPE.itemsize
    with open(path, "rb") as f:
        f.seek(COPY_HEADER_LEN)
        raw = np.fromfile(f, dtype=TUPLE_DTYPE, count=count)
    if count:
        # Cheap sanity check: exactly one non-NULL 8-byte field per tuple.
        head = raw[: min(count, 1000)]
        if not (np.all(head["nfields"] == 1) and np.all(head["length"] == 8)):
            raise ValueError(f"{path.name}: unexpected tuple layout, not a plain bigint column")
    return raw["oid"].astype(np.int64)


def dump_partition(creds, schema, part, sid, limit, outdir: Path, keep_raw: bool) -> dict:
    """COPY one partition out, parse it, save <part>.npy. Returns timing stats."""
    out_npy = outdir / f"{part}.npy"
    sql = _select_sql(schema, part, sid, limit)
    copy_sql = f"COPY ({sql}) TO STDOUT (FORMAT binary)"

    t0 = time.perf_counter()
    raw_path = outdir / f"{part}.copybin"
    fd, tmp = tempfile.mkstemp(dir=str(outdir), prefix=f".{part}.", suffix=".part")
    os.close(fd)
    tmp = Path(tmp)
    try:
        conn = connect(creds)
        try:
            with conn.cursor() as cur, open(tmp, "wb") as f:
                cur.copy_expert(copy_sql, f)
        finally:
            conn.close()
        t_copy = time.perf_counter() - t0
        wire_bytes = tmp.stat().st_size

        t1 = time.perf_counter()
        oids = parse_copy_binary(tmp)
        np.save(out_npy, oids)
        t_parse = time.perf_counter() - t1

        if keep_raw:
            tmp.replace(raw_path)
    finally:
        if tmp.exists():
            tmp.unlink()

    return {
        "partition": part, "rows": int(oids.size), "wire_bytes": wire_bytes,
        "copy_s": t_copy, "parse_s": t_parse, "path": out_npy,
    }


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--credentials", default=DEFAULT_CREDENTIALS)
    ap.add_argument("--schema", default=os.getenv("OFFLINE_DB_SCHEMA", "multisurvey_ztf"))
    ap.add_argument("--table", default="object")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output directory for the .npy files")
    ap.add_argument("--jobs", type=int, default=8, help="partitions to pull concurrently")
    ap.add_argument("--sid", type=int, default=None, help="filter by sid (default: no filter)")
    ap.add_argument("--limit", type=int, default=None,
                    help="rows per partition — for a cheap calibration run")
    ap.add_argument("--merge", action="store_true",
                    help="also write all_oids.npy with every partition concatenated")
    ap.add_argument("--keep-raw", action="store_true", help="keep the raw .copybin dumps")
    ap.add_argument("--explain", action="store_true",
                    help="print the planner's chosen node per partition, then exit")
    ap.add_argument("--dry-run", action="store_true",
                    help="print size/time estimates without transferring anything")
    args = ap.parse_args()

    creds = load_credentials(args.credentials)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    conn = connect(creds)
    try:
        parts = list_partitions(conn, args.schema, args.table)
        if args.explain:
            print(f"planner choice for SELECT oid FROM {args.schema}.<partition>:\n")
            for name, _, _, _ in parts:
                print(f"  {name:16s} {explain_partition(conn, args.schema, name, args.sid, args.limit)}")
            return
    finally:
        if not args.dry_run:
            conn.close()

    est_rows = sum(p[1] for p in parts)
    if args.limit:
        est_rows = min(est_rows, args.limit * len(parts))
    est_wire = est_rows * TUPLE_DTYPE.itemsize
    est_npy = est_rows * 8

    print(f"{args.schema}.{args.table}: {len(parts)} partitions, ~{est_rows:,} rows\n")
    print(f"  {'partition':18s} {'est. rows':>14s} {'heap':>9s} {'all-visible':>12s}")
    for name, rows, heap, vis in parts:
        flag = "" if vis >= 95 else "   <- heap fetches likely"
        print(f"  {name:18s} {rows:>14,} {human(heap):>9s} {vis:>11.1f}%{flag}")
    print()
    print(f"  over the wire (COPY binary, 14 B/row) : ~{human(est_wire)}")
    print(f"  on disk as int64 .npy (8 B/row)       : ~{human(est_npy)}")
    print(f"  peak RAM (one partition per job)      : ~{human(max(p[1] for p in parts) * 22 * min(args.jobs, len(parts)))}")

    if args.dry_run:
        print("\n  network time at 1 Gbit  : "
              f"~{est_wire * 8 / 1e9 / 0.85:,.0f} s")
        print("  network time at 100 Mbit: "
              f"~{est_wire * 8 / 1e8 / 0.85:,.0f} s")
        print("\n(dry run — nothing transferred; drop --dry-run to go)")
        conn.close()
        return

    print(f"\npulling with {args.jobs} concurrent connections -> {outdir}\n")
    t0 = time.perf_counter()
    results, failures = [], []
    with cf.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(dump_partition, creds, args.schema, name, args.sid,
                        args.limit, outdir, args.keep_raw): name
            for name, _, _, _ in parts
        }
        for fut in cf.as_completed(futures):
            name = futures[fut]
            try:
                r = fut.result()
            except Exception as exc:  # keep the other partitions going
                failures.append((name, exc))
                print(f"  {name:18s} FAILED: {type(exc).__name__}: {exc}")
                continue
            results.append(r)
            rate = r["wire_bytes"] / r["copy_s"] if r["copy_s"] else 0
            print(f"  {name:18s} {r['rows']:>12,} rows  {human(r['wire_bytes']):>9s}  "
                  f"copy {r['copy_s']:6.1f}s ({human(rate)}/s)  parse {r['parse_s']:5.1f}s")
    elapsed = time.perf_counter() - t0

    total_rows = sum(r["rows"] for r in results)
    total_wire = sum(r["wire_bytes"] for r in results)
    print(f"\n  {len(results)}/{len(parts)} partitions in {elapsed:,.1f}s")
    print(f"  {total_rows:,} oids, {human(total_wire)} over the wire, "
          f"{human(total_wire / elapsed if elapsed else 0)}/s aggregate")

    if args.merge and results:
        if failures:
            print("\n  skipping --merge: some partitions failed, the merge would be incomplete")
        else:
            t1 = time.perf_counter()
            merged = outdir / "all_oids.npy"
            arrays = [np.load(r["path"], mmap_mode="r") for r in sorted(results, key=lambda r: r["partition"])]
            out = np.empty(total_rows, dtype=np.int64)
            pos = 0
            for a in arrays:
                out[pos:pos + a.size] = a
                pos += a.size
            np.save(merged, out)
            print(f"  merged -> {merged} ({human(merged.stat().st_size)}, {time.perf_counter() - t1:.1f}s)")

    if failures:
        sys.exit(f"\n{len(failures)} partition(s) failed")


if __name__ == "__main__":
    main()
