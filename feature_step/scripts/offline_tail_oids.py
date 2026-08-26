#!/usr/bin/env python
"""Build the oid list for a TAIL run: what the completed run does not cover.

The full run (SERVER_QUICKSTART.md step 11) was driven by a frozen snapshot of
the catalogue -- `features/offline/oids/run.npy`, materialized once by
`offline_setup.py` and pinned by SHA-1 in `<out-dir>/run.json`
(BHRF_RUN_RESULTS.md §5). `select_oids` carries no date filter, so the run
misses two different things, and they need two different queries:

  new       oids over the cut that are not in the baseline array -- objects that
            did not exist, or had not reached `n_det >= 2`, when it was built.
            Answered by a diff against a fresh full selection.

  changed   objects the run DID process whose light curve has grown since. Their
            rows in `feature` and `probability` were computed from a shorter
            curve and are stale. Answered by `object.lastmjd > <date>`, which is
            an index scan (`ix_object_lastmjd`) and therefore minutes, not the
            hour the full selection takes.

Reprocessing either is safe: `feature`, `probability` and `xmatch` all write
ON CONFLICT ... DO UPDATE, so a second pass overwrites its own rows.

WHICH DATE SIGNAL
    --since-mjd / --since-date filter `object.lastmjd`, the MJD of the last
    detection: "objects observed again after that date". This is the one to use.
    The run's data horizon was MJD 61266.52 (2026-08-14), so
    `--since-mjd 61266.52` is exactly "everything the run could not have seen".

    --updated-since filters `object.updated_date` instead, the day the row was
    last written by the magstats scribe. It catches late-arriving data whose MJD
    is OLD (a backfill), which the lastmjd filter cannot see. It costs a
    sequential scan of the 8 partitions -- the column is a `Date` with no index
    -- and it is NULL for any row never updated since insert. Use it in addition
    to a lastmjd filter when you suspect a backfill, not instead of it.

    --drop-covered then does the exact per-object check: it reads back
    `probability.lastmjd` -- the MJD the run actually classified each object at
    -- and drops the candidates whose stored value already covers their current
    lastmjd. Cheap (hash index on `probability.oid`, and only over candidates),
    but it can drop an object whose forced photometry extended past its last
    detection, so it is off by default. Off, the tail is a superset; on, it is
    tighter and may miss those.

    python scripts/offline_tail_oids.py --dry-run --run-dir $RUN/bhrf_run
    python scripts/offline_tail_oids.py --since-mjd 61266.52 --out $RUN/oids/tail.npy
    python scripts/offline_tail_oids.py --since-date 2026-08-14 --no-full-diff \
        --out $RUN/oids/tail.npy          # minutes: index scan only, no full scan

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
from features.offline.probability_writer import CLASSIFIER_IDS  # noqa: E402

OFFLINE = PIPE / "feature_step" / "features" / "offline"
DEFAULT_CREDENTIALS = str(OFFLINE / "credentials.json")
DEFAULT_BASELINE = str(OFFLINE / "oids" / "run.npy")
# Same default as offline_setup.py: the baseline was built with n_det >= 2, and
# diffing against a list built with a different cut compares two different
# questions.
DEFAULT_MIN_N_DET = 2
# oids per probability lookup. The index is a hash on oid, so this is one index
# probe per oid either way; the batch size only trades round trips for the size
# of the ANY() array.
PROB_BATCH = 20_000

MJD_EPOCH = dt.date(1858, 11, 17)  # MJD 0


def mjd_from_date(day: dt.date) -> float:
    return float((day - MJD_EPOCH).days)


def sha1_of(oids: np.ndarray) -> str:
    """Same digest offline_run_batch.run_fingerprint stores in run.json."""
    return hashlib.sha1(np.ascontiguousarray(oids).tobytes()).hexdigest()


def _stream(conn, sql, params) -> pd.DataFrame:
    frames = list(pd.read_sql_query(text(sql), conn, params=params, chunksize=1_000_000))
    if not frames:
        return pd.DataFrame({"oid": np.empty(0, np.int64),
                             "lastmjd": np.empty(0, np.float64)})
    return pd.concat(frames, ignore_index=True)


def select_changed(credentials: str, min_n_det: int, since_mjd=None,
                   updated_since=None) -> pd.DataFrame:
    """(oid, lastmjd) for objects over the cut that moved after the given point.

    Two filters, OR-ed by the caller running whichever were asked for:

      lastmjd > :since       -- served by ix_object_lastmjd. For a recent date
                                this is a handful of index pages; for an old one
                                the planner flips to a seq scan, which is correct
                                but costs what the full selection costs.
      updated_date > :day    -- no index on the column, so always a seq scan of
                                the 8 partitions. It is the only way to see a
                                backfill of old-MJD detections.
    """
    out = []
    engine = db._make_engine(credentials)
    with engine.connect() as conn:
        conn = conn.execution_options(stream_results=True)
        if since_mjd is not None:
            out.append(_stream(conn, f"""
                SELECT oid, lastmjd FROM {db.SCHEMA}.object
                WHERE sid = :sid AND n_det >= :min_n_det AND lastmjd > :since
            """, {"sid": db.SID, "min_n_det": min_n_det, "since": since_mjd}))
        if updated_since is not None:
            out.append(_stream(conn, f"""
                SELECT oid, lastmjd FROM {db.SCHEMA}.object
                WHERE sid = :sid AND n_det >= :min_n_det AND updated_date > :day
            """, {"sid": db.SID, "min_n_det": min_n_det, "day": updated_since}))
    if not out:
        return pd.DataFrame({"oid": np.empty(0, np.int64),
                             "lastmjd": np.empty(0, np.float64)})
    changed = pd.concat(out, ignore_index=True).drop_duplicates("oid")
    changed["oid"] = changed["oid"].astype(np.int64)
    return changed.sort_values("oid", ignore_index=True)


def stored_probability_lastmjd(credentials: str, oids: np.ndarray) -> pd.DataFrame:
    """(oid, p_lastmjd) = the MJD the run classified each oid at.

    `probability_writer` stores the light curve's lastmjd on every row it writes,
    so this is what the run SAW, per object -- the only exact answer to "is this
    object's classification stale", and it needs no assumption about when the
    run happened or what the catalogue looked like then.

    max() over the five BHRF classifier ids: all five frames of one object are
    written in the same pass with the same value, so the max is that value and
    survives an object that is missing a sub-classifier's rows.
    """
    engine = db._make_engine(credentials)
    frames = []
    with engine.connect() as conn:
        for start in range(0, len(oids), PROB_BATCH):
            batch = [int(o) for o in oids[start:start + PROB_BATCH]]
            frames.append(pd.read_sql_query(text(f"""
                SELECT oid, max(lastmjd) AS p_lastmjd
                FROM {db.SCHEMA}.probability
                WHERE oid = ANY(:oids) AND sid = :sid AND classifier_id = ANY(:cids)
                GROUP BY oid
            """), conn, params={"oids": batch, "sid": db.SID,
                                "cids": list(CLASSIFIER_IDS)}))
    if not frames:
        return pd.DataFrame({"oid": np.empty(0, np.int64),
                             "p_lastmjd": np.empty(0, np.float64)})
    got = pd.concat(frames, ignore_index=True)
    got["oid"] = got["oid"].astype(np.int64)
    return got


def drop_covered(credentials: str, changed: pd.DataFrame) -> tuple:
    """Keep the candidates whose current lastmjd is past what the run classified.

    An oid with NO probability row is KEPT: the run either skipped it as
    unclassifiable (too few real detections -- 6.9M of them) or never saw it, and
    in both cases the new detections are the reason to look again.
    """
    stored = stored_probability_lastmjd(credentials, changed["oid"].to_numpy())
    merged = changed.merge(stored, on="oid", how="left")
    stale = merged["p_lastmjd"].isna() | (merged["lastmjd"] > merged["p_lastmjd"])
    return merged.loc[stale, ["oid", "lastmjd"]].reset_index(drop=True), int((~stale).sum())


def check_baseline_fingerprint(baseline: np.ndarray, run_dir: Path) -> str:
    """Confirm this .npy is the array the finished run actually consumed.

    Nothing else ties them together: run.npy is a plain file that could have been
    rebuilt, and a rebuilt baseline makes the diff silently too small -- objects
    the run never processed would look like objects it did.
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
                       help="objects with object.lastmjd > this (the run's horizon: 61266.52).")
    since.add_argument("--since-date",
                       help="same, as YYYY-MM-DD (00:00 UT of that day).")
    ap.add_argument("--updated-since",
                    help="also objects with object.updated_date > YYYY-MM-DD (seq scan; "
                         "catches backfills of old-MJD data).")
    ap.add_argument("--drop-covered", action="store_true",
                    help="drop candidates whose stored probability.lastmjd already "
                         "covers their current lastmjd.")
    ap.add_argument("--no-full-diff", action="store_true", dest="no_full_diff",
                    help="skip the full-catalogue selection; find new oids only among "
                         "the changed ones. Minutes instead of an hour, and misses a "
                         "new object whose detections are all older than --since.")
    ap.add_argument("--out", default=str(OFFLINE / "oids" / "tail.npy"),
                    help="where to write the tail array (.npy).")
    ap.add_argument("--dry-run", action="store_true",
                    help="query and report, write nothing.")
    args = ap.parse_args()

    since_mjd = args.since_mjd
    if args.since_date:
        since_mjd = mjd_from_date(dt.date.fromisoformat(args.since_date))
        print(f"--since-date {args.since_date} -> MJD {since_mjd:.1f}")
    if since_mjd is None and args.updated_since is None and args.no_full_diff:
        ap.error("--no-full-diff needs --since-mjd/--since-date or --updated-since: "
                 "without a date filter and without the full selection there is "
                 "nothing to diff.")

    baseline = np.load(args.baseline).astype(np.int64)
    baseline.sort()
    print(f"baseline: {len(baseline):,} oids from {args.baseline}")
    fp_note = "not checked (no --run-dir)"
    if args.run_dir:
        fp_note = check_baseline_fingerprint(baseline, Path(args.run_dir))
        print(f"          {fp_note}")

    # --- changed: the objects whose light curve moved -----------------------
    changed = pd.DataFrame({"oid": np.empty(0, np.int64),
                            "lastmjd": np.empty(0, np.float64)})
    n_covered = 0
    if since_mjd is not None or args.updated_since:
        what = []
        if since_mjd is not None:
            what.append(f"lastmjd > {since_mjd}")
        if args.updated_since:
            what.append(f"updated_date > {args.updated_since} (seq scan, slower)")
        print(f"selecting objects with {' or '.join(what)} ...", flush=True)
        changed = select_changed(args.credentials, args.min_n_det, since_mjd,
                                 args.updated_since)
        print(f"changed:  {len(changed):,} objects moved")
        if args.drop_covered and len(changed):
            print("reading back probability.lastmjd for those oids ...", flush=True)
            changed, n_covered = drop_covered(args.credentials, changed)
            print(f"          {n_covered:,} already classified at or past their "
                  f"current lastmjd -> dropped, {len(changed):,} left")

    changed_oids = changed["oid"].to_numpy(dtype=np.int64)
    changed_oids.sort()
    in_base = np.isin(changed_oids, baseline, assume_unique=True)
    updated = changed_oids[in_base]        # processed before, now stale
    new = changed_oids[~in_base]           # never processed, and recently active

    # --- new: the full-catalogue diff ---------------------------------------
    n_current = n_gone = None
    if not args.no_full_diff:
        import offline_run_batch as R
        print(f"selecting oids with n_det >= {args.min_n_det} from {db.SCHEMA}.object "
              "(a scan of the table, takes a while) ...", flush=True)
        current = R.select_oids(args.credentials, args.min_n_det)
        n_current = len(current)
        print(f"current:  {n_current:,} oids over the cut")
        new = np.union1d(new, np.setdiff1d(current, baseline, assume_unique=True))
        # Baseline oids no longer over the cut. Should be zero -- n_det only grows
        # and rows are not deleted -- so a non-zero count means the baseline was
        # built against a different table state, and that is worth saying.
        n_gone = int(len(np.setdiff1d(baseline, current, assume_unique=True)))
        if n_gone:
            print(f"WARNING:  {n_gone:,} baseline oids are no longer over the cut")

    print(f"new:      {len(new):,} oids never processed")
    print(f"updated:  {len(updated):,} oids processed before, light curve grew")

    tail = np.union1d(new, updated).astype(np.int64)
    print(f"tail:     {len(tail):,} oids")

    report = {
        "schema": db.SCHEMA, "min_n_det": args.min_n_det,
        "baseline": os.path.abspath(args.baseline), "baseline_n": int(len(baseline)),
        "baseline_sha1": sha1_of(baseline), "baseline_check": fp_note,
        "since_mjd": since_mjd, "updated_since": args.updated_since,
        "drop_covered": bool(args.drop_covered), "n_covered_dropped": n_covered,
        "full_diff": not args.no_full_diff, "current_n": n_current, "n_gone": n_gone,
        "n_changed": int(len(changed_oids)),
        "n_new": int(len(new)), "n_updated": int(len(updated)),
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
