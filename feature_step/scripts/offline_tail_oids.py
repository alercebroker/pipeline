#!/usr/bin/env python
"""Build the oid list for a TAIL run: what the completed run did not cover.

The full run (SERVER_QUICKSTART.md step 11) was driven by a frozen snapshot of
the catalogue -- `features/offline/oids/run.npy`, materialized once by
`offline_setup.py` and pinned by SHA-1 in `<out-dir>/run.json`
(BHRF_RUN_RESULTS.md §5). `select_oids` carries no date filter, so anything that
entered `object`, or crossed the `n_det` cut, AFTER that file was written is
simply absent from the run and nothing reports its absence.

This script produces the difference, in two flavours:

  new       oids that satisfy the cut TODAY and are not in the baseline array.
            This is "objects we never processed" and is the default.

  updated   oids that ARE in the baseline but whose `object.lastmjd` has moved
            past --since-mjd, i.e. objects whose light curve grew since the run
            and whose stored features/probabilities are therefore stale. Off by
            default, because it is a different question and a much larger set.
            Reprocessing them is safe: all three writers upsert
            (ON CONFLICT ... DO UPDATE), so a second pass overwrites the old
            rows rather than duplicating them.

The output is a sorted int64 .npy that `offline_run_batch.py --oid-file` reads
directly, plus a `<out>.json` report saying exactly what was diffed against
what -- the baseline's SHA-1 included, so the tail list can be traced back to
the run it complements.

    # what would the tail be? (queries the DB, writes nothing)
    python scripts/offline_tail_oids.py --dry-run

    # objects never processed
    python scripts/offline_tail_oids.py --out $RUN/oids/tail.npy

    # ...plus everything whose LC grew since the run's data horizon
    python scripts/offline_tail_oids.py --out $RUN/oids/tail.npy \
        --since-date 2026-08-14 --run-dir $RUN/bhrf_run

The tail is a DIFFERENT oid array, so it needs a fresh --out-dir: unit index N
means `oids[N*unit_size:...]` of one specific array, and `run.json` refuses a
resume across arrays for exactly that reason.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

PIPE = Path(__file__).resolve().parents[2]  # .../pipeline
for _p in (PIPE / "feature_step", PIPE / "lc_classifier", PIPE / "libs" / "idmapper",
           PIPE / "libs" / "xmatch_client", PIPE / "alerce_classifiers"):
    sys.path.insert(0, str(_p))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # offline_run_batch

from features.offline import db  # noqa: E402

OFFLINE = PIPE / "feature_step" / "features" / "offline"
DEFAULT_CREDENTIALS = str(OFFLINE / "credentials.json")
DEFAULT_BASELINE = str(OFFLINE / "oids" / "run.npy")
# Same default as offline_setup.py: the baseline was built with n_det >= 2, and
# diffing against a list built with a different cut compares two different
# questions.
DEFAULT_MIN_N_DET = 2

MJD_EPOCH = dt.date(1858, 11, 17)  # MJD 0


def mjd_from_date(day: dt.date) -> float:
    return float((day - MJD_EPOCH).days)


def sha1_of(oids: np.ndarray) -> str:
    """Same digest offline_run_batch.run_fingerprint stores in run.json."""
    return hashlib.sha1(np.ascontiguousarray(oids).tobytes()).hexdigest()


def select_oids_since(credentials: str, min_n_det: int, since_mjd: float) -> np.ndarray:
    """oids over the cut whose light curve extends past `since_mjd`.

    `object` has no index on lastmjd, so this is a scan of the 8 partitions --
    the same order of cost as the unfiltered selection, which is why it only
    runs when asked for.
    """
    sql = f"""
        SELECT oid FROM {db.SCHEMA}.object
        WHERE sid = :sid AND n_det >= :min_n_det AND lastmjd > :since
    """
    engine = db._make_engine(credentials)
    with engine.connect() as conn:
        conn = conn.execution_options(stream_results=True)
        chunks = [c["oid"].to_numpy(dtype=np.int64)
                  for c in pd.read_sql_query(text(sql), conn,
                                             params={"sid": db.SID,
                                                     "min_n_det": min_n_det,
                                                     "since": since_mjd},
                                             chunksize=1_000_000)]
    if not chunks:
        return np.empty(0, dtype=np.int64)
    out = np.concatenate(chunks)
    out.sort()
    return out


def check_baseline_fingerprint(baseline: np.ndarray, run_dir: Path) -> str:
    """Confirm this .npy is the array the finished run actually consumed.

    Nothing else ties them together: run.npy is a plain file that could have
    been rebuilt, and a rebuilt baseline would make the diff silently too
    small -- objects processed by the run would look like new ones (harmless,
    just wasted work) or, worse, a baseline built with a LARGER cut would hide
    objects that were never processed.
    """
    path = run_dir / "run.json"
    if not path.exists():
        return f"{path} does not exist -- baseline not verified"
    fp = json.loads(path.read_text())
    got, want = sha1_of(baseline), fp.get("oid_sha1")
    if got == want:
        return f"matches {path} (n_oids={fp.get('n_oids'):,}, oid_sha1={got[:12]})"
    raise SystemExit(
        "\nBASELINE MISMATCH: the .npy is not the list that run produced.\n"
        f"  baseline : {len(baseline):,} oids, sha1={got}\n"
        f"  {path}: {fp.get('n_oids'):,} oids, sha1={want}\n"
        "  Diffing against the wrong baseline gives the wrong tail. Use the\n"
        "  run.npy that built this run, or drop --run-dir to skip the check.\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--credentials", default=DEFAULT_CREDENTIALS)
    ap.add_argument("--baseline", default=DEFAULT_BASELINE,
                    help="the oid .npy the completed run consumed.")
    ap.add_argument("--run-dir",
                    help="the finished --out-dir; verifies the baseline against its run.json.")
    ap.add_argument("--min-n-det", type=int, default=DEFAULT_MIN_N_DET,
                    help="must be the cut the baseline was built with (default 2).")
    since = ap.add_mutually_exclusive_group()
    since.add_argument("--since-mjd", type=float,
                       help="also take baseline oids with object.lastmjd > this (stale LCs).")
    since.add_argument("--since-date",
                       help="same, as YYYY-MM-DD (the run's data horizon was 2026-08-14).")
    ap.add_argument("--out", default=str(OFFLINE / "oids" / "tail.npy"),
                    help="where to write the tail array (.npy).")
    ap.add_argument("--dry-run", action="store_true",
                    help="query and report, write nothing.")
    args = ap.parse_args()

    since_mjd = args.since_mjd
    if args.since_date:
        since_mjd = mjd_from_date(dt.date.fromisoformat(args.since_date))
        print(f"--since-date {args.since_date} -> MJD {since_mjd:.1f}")

    baseline = np.load(args.baseline).astype(np.int64)
    baseline.sort()
    print(f"baseline: {len(baseline):,} oids from {args.baseline}")
    fp_note = "not checked (no --run-dir)"
    if args.run_dir:
        fp_note = check_baseline_fingerprint(baseline, Path(args.run_dir))
        print(f"          {fp_note}")

    import offline_run_batch as R
    print(f"selecting oids with n_det >= {args.min_n_det} from {db.SCHEMA}.object "
          "(a scan of the table, takes a while) ...", flush=True)
    current = R.select_oids(args.credentials, args.min_n_det)
    print(f"current:  {len(current):,} oids over the cut")

    new = np.setdiff1d(current, baseline, assume_unique=True)
    # Objects in the baseline that no longer satisfy the cut. Should be zero --
    # n_det only grows and rows are not deleted -- so a non-zero count means the
    # baseline was built against a different table state and is worth saying.
    gone = np.setdiff1d(baseline, current, assume_unique=True)
    print(f"new:      {len(new):,} oids not in the baseline")
    if len(gone):
        print(f"WARNING:  {len(gone):,} baseline oids are no longer over the cut")

    stale = np.empty(0, dtype=np.int64)
    if since_mjd is not None:
        print(f"selecting oids with lastmjd > {since_mjd} (another scan) ...", flush=True)
        moved = select_oids_since(args.credentials, args.min_n_det, since_mjd)
        # Only the ones the run already processed: the rest are in `new` already.
        stale = np.intersect1d(moved, baseline, assume_unique=True)
        print(f"updated:  {len(stale):,} baseline oids with lastmjd > {since_mjd}")

    tail = np.union1d(new, stale).astype(np.int64)
    print(f"tail:     {len(tail):,} oids")

    report = {
        "schema": db.SCHEMA, "min_n_det": args.min_n_det,
        "baseline": os.path.abspath(args.baseline), "baseline_n": int(len(baseline)),
        "baseline_sha1": sha1_of(baseline), "baseline_check": fp_note,
        "current_n": int(len(current)),
        "n_new": int(len(new)), "n_gone": int(len(gone)),
        "since_mjd": since_mjd, "n_updated": int(len(stale)),
        "n_tail": int(len(tail)), "tail_sha1": sha1_of(tail),
        "out": os.path.abspath(args.out),
    }

    if args.dry_run:
        print("\n--dry-run: nothing written\n" + json.dumps(report, indent=2))
        return 0
    if not len(tail):
        print("\nnothing to do: the catalogue has not moved since the baseline. "
              "Nothing written.")
        return 0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Through a file handle: np.save(path) appends ".npy" unless the name already
    # ends in it, so a staging name would land as "tail.npy.tmp.npy" and the
    # rename below would find nothing (same trap as offline_setup.step_oids).
    tmp = out.with_suffix(".npy.tmp")
    with open(tmp, "wb") as fh:
        np.save(fh, tail)
    os.replace(tmp, out)
    Path(str(out) + ".json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {len(tail):,} oids -> {out}")
    print(f"      report -> {out}.json")
    print("\nnext (note the FRESH --out-dir: unit indices belong to one oid array):\n"
          f"  poetry run python scripts/offline_run_batch.py \\\n"
          f"      --oid-file {out} --out-dir $RUN/bhrf_tail \\\n"
          f"      --workers 64 --features \\\n"
          f"      --load-db --write-credentials features/offline/credentials.json --no-shards")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
