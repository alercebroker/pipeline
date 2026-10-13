"""Timed COPY of one multisurvey.probability partition filtered on classifier_version, to parquet.

Dumps the squidward (BHRF 2.1.0, classifier_version=210) rows of a single hash
partition to a parquet file and prints rows, elapsed time and a x16 projection
for the whole table. Read-only on the DB; writes one file under ~/dumps/.
Plain two-step: COPY the whole result as CSV into memory (~3 GB per partition),
then parse it with pyarrow and write one parquet file.

Setup: bash setup.sh (creates .venv here)
Usage: .venv/bin/python prob_dump_timing.py <readonly_credentials.json> [partition] [version]
"""
import io  # in-memory CSV buffer
import json  # credentials file is psycopg2 kwargs in json
import os  # paths and file sizes
import sys  # argv
import time  # timing

import psycopg2  # DB driver
import pyarrow as pa  # arrow types
import pyarrow.csv as pacsv  # CSV parser
import pyarrow.parquet as pq  # parquet writer

COLUMNS = ["oid", "sid", "classifier_id", "class_id", "probability", "ranking", "lastmjd"]  # COPY column order
SCHEMA = pa.schema(  # explicit types: real stays float32, small ints stay int16
    [
        ("oid", pa.int64()),  # bigint
        ("sid", pa.int16()),  # smallint
        ("classifier_id", pa.int16()),  # smallint
        ("class_id", pa.int16()),  # smallint
        ("probability", pa.float32()),  # real
        ("ranking", pa.int16()),  # smallint
        ("lastmjd", pa.float64()),  # double precision
    ]
)


def main():
    """Run the timed COPY-to-parquet and print labeled metrics."""
    cred = sys.argv[1]  # path to readonly credentials json
    part = int(sys.argv[2]) if len(sys.argv) > 2 else 0  # hash partition number
    version = int(sys.argv[3]) if len(sys.argv) > 3 else 210  # classifier_version to dump
    out = os.path.expanduser("~/dumps/prob_part_%d_v%d.parquet" % (part, version))  # output file
    tmp = out + ".tmp"  # written here, renamed on success so a partial file is never mistaken for done
    os.makedirs(os.path.dirname(out), exist_ok=True)  # ensure ~/dumps exists

    kw = json.load(open(cred))  # psycopg2 kwargs, never printed
    kw["connect_timeout"] = 10  # fail fast on firewall
    conn = psycopg2.connect(**kw)  # open connection
    conn.set_session(readonly=True)  # read-only session
    cur = conn.cursor()  # cursor for COPY
    cur.execute("SET statement_timeout = 0")  # full partition scan takes minutes

    sql = (  # COPY of the filtered partition as CSV, no header
        "COPY (SELECT %s FROM multisurvey.probability_part_%d WHERE classifier_version = %d) "
        "TO STDOUT WITH (FORMAT csv)" % (", ".join(COLUMNS), part, version)
    )
    t0 = time.time()  # start clock
    buf = io.BytesIO()  # whole CSV result in memory
    cur.copy_expert(sql, buf, size=1 << 20)  # step 1: COPY from the server
    t_copy = time.time() - t0  # seconds spent in COPY
    buf.seek(0)  # rewind for the parser
    table = pacsv.read_csv(  # step 2: parse with fixed schema
        buf,
        read_options=pacsv.ReadOptions(column_names=COLUMNS),  # no header in the CSV
        convert_options=pacsv.ConvertOptions(column_types=SCHEMA),  # no type inference
    )
    del buf  # free the 3 GB text
    pq.write_table(table, tmp, compression="zstd", row_group_size=4_000_000)  # step 3: parquet
    os.replace(tmp, out)  # rename marks completion
    dt = time.time() - t0  # elapsed seconds
    rows = table.num_rows  # rows dumped
    size = os.path.getsize(out)  # parquet size

    print("partition: %d" % part)  # which partition
    print("rows: %d" % rows)  # rows dumped
    print("elapsed_s: %.0f  (copy %.0f, parse+write %.0f)" % (dt, t_copy, dt - t_copy))  # the number we want
    print("rows_per_s: %.0f" % (rows / max(dt, 1e-9)))  # throughput
    print("parquet_mb: %.0f" % (size / 1e6))  # file size
    print("bytes_per_row: %.1f" % (size / max(rows, 1)))  # per-row weight
    print("x16_estimate: %.0f min, parquet %.0f GB" % (dt * 16 / 60, size * 16 / 1e9))  # serial projection


if __name__ == "__main__":
    main()  # entry point
