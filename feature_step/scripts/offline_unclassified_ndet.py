#!/usr/bin/env python
"""Why were the unclassified objects unclassified? Their n_det, from the DB.

offline_run_batch.py counts unclassifiable objects but never names them: the
manifest carries n_unclassifiable and nothing else, and only failed oids reach
errors/unit_*.jsonl. So "are these mostly objects with almost no detections?"
-- the one question that decides whether a 25% skip rate is expected or a bug --
cannot be answered from the run's own output at all.

It IS answerable by subtraction. A unit is a contiguous slice of the oid array
(unit i IS oids[i*unit_size:(i+1)*unit_size], which is what run.json's
fingerprint pins down), and its probability shard names every oid that
classified. The difference is the skipped set, exactly.

Then ask the database for their detection counts.

NOTE ON WHAT IS COUNTED. This reports RAW rows from detection JOIN
ztf_detection, before discard_bogus_detections drops non-forced epochs with
rb < 0.55. That is deliberate: an object is unclassifiable precisely when zero
real detections survive that filter (min_detections defaults to 1), so the
post-filter count is 0 for every object here by construction and a histogram of
it says nothing. The raw count is what separates the two explanations:

  * piled up at 0-2  -- objects with nothing to work with, correctly skipped.
  * a fat tail at 10+ -- objects that had plenty and lost them all to the rb
    cut, which is a different problem and worth chasing.

Only units with a probability shard can be used, so a --no-shards run gives
this nothing to work with.

    python offline_unclassified_ndet.py <out-dir> --oid-file <run.npy> --units 30
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import sqlalchemy as sa

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features.offline import db          # noqa: E402

CHUNK = 500          # matches the run's default minibatch, to stay gentle
BUCKETS = [(0, "0"), (1, "1"), (2, "2"), (3, "3"), (4, "4"), (5, "5"),
           (10, "6-10"), (20, "11-20"), (50, "21-50"), (10**9, ">50")]


def unclassified_oids(out_dir: Path, oid_file: str, n_units: int, seed: int):
    """Oids in the sampled units that finished but produced no probability row.

    Subtraction, not inference: the shard is the complete list of oids that
    classified, and the unit's oid slice is fixed by run.json's fingerprint."""
    unit_size = json.loads((out_dir / "run.json").read_text())["unit_size"]
    all_oids = np.load(oid_file, mmap_mode="r")
    shards = sorted((out_dir / "probabilities").glob("unit_*.parquet"))
    if not shards:
        sys.exit(f"no probability shards in {out_dir}/probabilities -- the run "
                 f"used --no-shards, so the classified oids are not on disk")
    random.seed(seed)
    picked = random.sample(shards, min(n_units, len(shards)))

    skipped, seen = [], 0
    for path in picked:
        index = int(path.stem.split("_")[1])
        unit = np.asarray(all_oids[index * unit_size:(index + 1) * unit_size])
        classified = np.unique(pq.read_table(path, columns=["oid"])["oid"].to_numpy())
        seen += len(unit)
        skipped.append(np.setdiff1d(unit, classified))
    return np.concatenate(skipped), len(picked), seen


def detection_counts(oids: np.ndarray, credentials: str) -> dict:
    """oid -> raw detection rows. Oids absent from the result have zero."""
    query = sa.text(f"""
        SELECT d.oid AS oid, count(*) AS n
        FROM {db.SCHEMA}.detection d
        JOIN {db.SCHEMA}.ztf_detection z
          ON d.oid = z.oid AND d.measurement_id = z.measurement_id
        WHERE d.oid = ANY(:oids) AND d.sid = :sid
        GROUP BY d.oid""")
    counts = {}
    engine = db._make_engine(credentials)
    with engine.connect() as conn:
        for start in range(0, len(oids), CHUNK):
            chunk = [int(o) for o in oids[start:start + CHUNK]]
            for oid, n in conn.execute(query, {"oids": chunk, "sid": db.SID}):
                counts[oid] = n
    return counts


def histogram(oids: np.ndarray, counts: dict) -> None:
    tally = {label: 0 for _, label in BUCKETS}
    for oid in oids:
        n = counts.get(int(oid), 0)
        for hi, label in BUCKETS:
            if n <= hi:
                tally[label] += 1
                break
    total = len(oids)
    print(f"\n{'n_det':>7} {'objects':>10} {'%':>7}   (raw rows, before the "
          f"rb < 0.55 cut)")
    for _, label in BUCKETS:
        n = tally[label]
        if n:
            print(f"{label:>7} {n:>10,} {100 * n / total:6.1f}%  "
                  f"{'#' * int(60 * n / total)}")
    with_none = tally["0"]
    few = sum(tally[l] for l in ("0", "1", "2"))
    print(f"\n  no detection rows at all : {with_none:,} ({100*with_none/total:.1f}%)")
    print(f"  <= 2 rows                : {few:,} ({100*few/total:.1f}%)")
    tail = total - sum(tally[l] for l in ("0", "1", "2", "3", "4", "5"))
    print(f"  6 or more rows           : {tail:,} ({100*tail/total:.1f}%)"
          f"  <- these lost everything to the rb cut; chase this if it is large")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir", help="the run's --out-dir")
    ap.add_argument("--oid-file", required=True,
                    help="the SAME oid list the run used (run.json pins it)")
    ap.add_argument("--units", type=int, default=30,
                    help="how many units to sample (default: %(default)s)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--credentials", default="features/offline/credentials.json")
    args = ap.parse_args()

    oids, n_units, n_seen = unclassified_oids(
        Path(args.out_dir), args.oid_file, args.units, args.seed)
    print(f"{n_units} units sampled -> {n_seen:,} oids, "
          f"{len(oids):,} unclassified ({100 * len(oids) / n_seen:.1f}%)")
    if not len(oids):
        print("nothing skipped in this sample")
        return 0
    histogram(oids, detection_counts(oids, args.credentials))
    return 0


if __name__ == "__main__":
    sys.exit(main())
