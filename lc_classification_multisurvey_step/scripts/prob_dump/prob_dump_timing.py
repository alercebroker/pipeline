"""Timed COPY of one multisurvey.probability partition filtered on classifier_version.

Dumps the squidward (BHRF 2.1.0, classifier_version=210) rows of a single hash
partition to a gzip CSV and prints rows, elapsed time and a x16 projection for
the whole table. Read-only on the DB; writes one file under ~/dumps/.

Setup: bash setup.sh (creates .venv here)
Usage: .venv/bin/python prob_dump_timing.py <readonly_credentials.json> [partition] [version]
"""
import gzip  # gzip output file
import json  # credentials file is psycopg2 kwargs in json
import os  # paths and file sizes
import sys  # argv
import time  # timing

import psycopg2  # DB driver


class Counter:
    """File-like wrapper that counts rows (newlines) and raw bytes before compression."""

    def __init__(self, f):
        self.f = f  # underlying gzip file
        self.rows = 0  # newlines seen = CSV rows
        self.raw = 0  # uncompressed bytes seen

    def write(self, b):
        """Count then forward a chunk from COPY TO STDOUT."""
        self.rows += b.count(b"\n")  # one newline per CSV row
        self.raw += len(b)  # uncompressed size
        self.f.write(b)  # forward to gzip


def main():
    """Run the timed COPY and print labeled metrics."""
    cred = sys.argv[1]  # path to readonly credentials json
    part = int(sys.argv[2]) if len(sys.argv) > 2 else 0  # hash partition number
    version = int(sys.argv[3]) if len(sys.argv) > 3 else 210  # classifier_version to dump
    out = os.path.expanduser("~/dumps/prob_part_%d_v%d.csv.gz" % (part, version))  # output file
    os.makedirs(os.path.dirname(out), exist_ok=True)  # ensure ~/dumps exists

    kw = json.load(open(cred))  # psycopg2 kwargs, never printed
    kw["connect_timeout"] = 10  # fail fast on firewall
    conn = psycopg2.connect(**kw)  # open connection
    conn.set_session(readonly=True)  # read-only session
    cur = conn.cursor()  # cursor for COPY
    cur.execute("SET statement_timeout = 0")  # full partition scan takes minutes

    sql = (  # COPY of the filtered partition as CSV
        "COPY (SELECT oid, sid, classifier_id, class_id, probability, ranking, lastmjd "
        "FROM multisurvey.probability_part_%d WHERE classifier_version = %d) "
        "TO STDOUT WITH (FORMAT csv)" % (part, version)
    )
    t0 = time.time()  # start clock
    with gzip.open(out, "wb", compresslevel=1) as g:  # fast compression
        w = Counter(g)  # counting wrapper
        cur.copy_expert(sql, w, size=1 << 20)  # 1 MB chunks
    dt = time.time() - t0  # elapsed seconds
    gz = os.path.getsize(out)  # compressed size

    print("partition: %d" % part)  # which partition
    print("rows: %d" % w.rows)  # rows dumped
    print("elapsed_s: %.0f" % dt)  # the number we want
    print("rows_per_s: %.0f" % (w.rows / max(dt, 1e-9)))  # throughput
    print("csv_mb: %.0f" % (w.raw / 1e6))  # uncompressed size
    print("gz_mb: %.0f" % (gz / 1e6))  # compressed size
    print("bytes_per_row_csv: %.1f  gz: %.1f" % (w.raw / max(w.rows, 1), gz / max(w.rows, 1)))  # per-row weight
    print("x16_estimate: %.0f min, csv %.0f GB, gz %.0f GB" % (dt * 16 / 60, w.raw * 16 / 1e9, gz * 16 / 1e9))  # serial projection


if __name__ == "__main__":
    main()  # entry point
