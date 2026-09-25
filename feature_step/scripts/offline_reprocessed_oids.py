#!/usr/bin/env python
"""Build the oid list for a run driven by an EXTERNAL list of ZTF names.

The full run (SERVER_QUICKSTART.md step 11) and the tail run (step 12) both
select their objects from the catalogue. This one is handed the objects: a
reprocessing campaign's manifest, a colleague's text file -- a list of ZTF names
("ZTF18abjgybv"), unsorted, with repeats and nulls. It writes the `.npy` that
`offline_run_batch.py --oid-file` consumes.

Three things happen to the list, in order:

  1. NAMES -> OIDS. The multisurvey bigint is idmapper's: survey id 1 in the top
     byte, then year * 26**7, then the seven letters in base 26. Every name is
     validated first, because a mistyped name does not fail downstream -- it is
     an oid with no detections, silently counted as unclassifiable.

  2. --eligible: keep only what the catalogue can classify. The runner reads
     detections for whatever oid it is given; an object missing from
     <schema>.object, or under the n_det cut, costs a round trip and lands in
     the "no detections" counter. Same query as the full run's `select_oids`
     (a bitmap scan over the 8 partitions, minutes), intersected client-side.

  3. Label against the baseline. `run.npy` is the full run's input; the split
     into "never processed" and "processed before" says whether this run mostly
     refreshes stale rows or mostly classifies new objects. It changes nothing.

    python scripts/offline_reprocessed_oids.py --ztf-list oids.txt --dry-run
    python scripts/offline_reprocessed_oids.py --ztf-list alerts.parquet --column oid \
        --eligible --run-dir $RUN/bhrf_run --out $RUN/oids/reprocesados.npy

The result is a DIFFERENT oid array from the full run's, so it needs a fresh
--out-dir: unit index N means `oids[N*unit_size:...]` of one specific array,
and `run.json` refuses a resume across arrays for exactly that reason.
"""
import argparse
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

from features.offline import db  # noqa: E402

OFFLINE = PIPE / "feature_step" / "features" / "offline"
DEFAULT_CREDENTIALS = str(OFFLINE / "credentials.json")
DEFAULT_BASELINE = str(OFFLINE / "oids" / "run.npy")
# Same default as offline_setup.py and offline_tail_oids.py: the cut that makes
# an object eligible at all.
DEFAULT_MIN_N_DET = 2

# idmapper.mapper: SURVEY_IDS["ZTF"] << (63 - SURVEY_PREFIX_LEN_BITS)
ZTF_SURVEY_BITS = np.int64(1 << 55)
YEAR_BASE = np.int64(26 ** 7)


# --- 1. names -> oids ---------------------------------------------------------

def read_ztf_ids(path, column: str = "oid") -> np.ndarray:
    """ZTF names from a newline .txt or a parquet column. Nulls and blank lines
    dropped; nothing else touched (no dedupe, no sort -- see build_oid_array)."""
    p = Path(path)
    if p.suffix == ".parquet":
        import pyarrow.parquet as pq
        col = pq.ParquetFile(p).read(columns=[column])[column]
        return np.asarray(col.drop_null().to_numpy(zero_copy_only=False), dtype=str)
    with open(p) as fh:
        return np.array([ln.strip() for ln in fh if ln.strip()], dtype=str)


def encode_ztf_ids(names: np.ndarray) -> np.ndarray:
    """Vectorised idmapper.ztf.encode_ztf_to_masterid_without_survey, with the
    survey prefix added (== catalog_oid_to_masterid("ZTF", name, validate=True)).

    Validation is the same as idmapper's is_ztf_oid_valid: 'ZTF', two digits,
    seven lowercase ascii letters. The first offender is named in the error."""
    names = np.asarray(names, dtype=str)
    if len(names) == 0:
        return np.empty(0, dtype=np.int64)
    # Fixed-width bytes: one row of 12 uint8 per name. A name that is not 12
    # ascii bytes cannot be laid out this way, so it is caught before the view.
    lengths = np.char.str_len(names)
    bad = np.flatnonzero(lengths != 12)
    if len(bad):
        _reject(names[bad[0]])
    raw = names.astype("S12")
    if (np.char.str_len(raw) != 12).any():           # non-ascii shrank on encode
        _reject(names[np.flatnonzero(np.char.str_len(raw) != 12)[0]])
    b = raw.view(np.uint8).reshape(len(raw), 12)
    ok = ((b[:, 0] == ord("Z")) & (b[:, 1] == ord("T")) & (b[:, 2] == ord("F"))
          & (b[:, 3:5] >= ord("0")).all(axis=1) & (b[:, 3:5] <= ord("9")).all(axis=1)
          & (b[:, 5:] >= ord("a")).all(axis=1) & (b[:, 5:] <= ord("z")).all(axis=1))
    bad = np.flatnonzero(~ok)
    if len(bad):
        _reject(names[bad[0]])
    year = (b[:, 3].astype(np.int64) - ord("0")) * 10 + (b[:, 4].astype(np.int64) - ord("0"))
    letters = b[:, 5:].astype(np.int64) - ord("a")
    powers = (26 ** np.arange(6, -1, -1)).astype(np.int64)
    return ZTF_SURVEY_BITS + year * YEAR_BASE + letters @ powers


def _reject(name):
    raise ValueError(f"Invalid ZTF object ID: {name!r} "
                     "(expected 'ZTF', two digits, seven lowercase letters)")


def build_oid_array(names: np.ndarray) -> np.ndarray:
    """Unique, ascending int64 -- the shape offline_run_batch.py expects."""
    return np.unique(encode_ztf_ids(names))


# --- 2. eligible: what the catalogue can classify ------------------------------

def select_catalogue(credentials: str, min_n_det: int) -> np.ndarray:
    """oids of <schema>.object with n_det >= min_n_det. Same query as
    offline_run_batch.select_oids; the sort is numpy's for the same reason."""
    sql = f"""
        SELECT oid FROM {db.SCHEMA}.object
        WHERE sid = :sid AND n_det >= :min_n_det
    """
    engine = db._make_engine(credentials)
    with engine.connect() as conn:
        conn = conn.execution_options(stream_results=True)
        chunks = [c["oid"].to_numpy(dtype=np.int64)
                  for c in pd.read_sql_query(text(sql), conn,
                                             params={"sid": db.SID, "min_n_det": min_n_det},
                                             chunksize=1_000_000)]
    if not chunks:
        return np.empty(0, dtype=np.int64)
    out = np.concatenate(chunks)
    out.sort()
    return out


def keep_eligible(oids: np.ndarray, catalogue: np.ndarray) -> tuple:
    """(oids present in catalogue, ascending; how many were not)."""
    kept = oids[np.isin(oids, catalogue)]
    return np.sort(kept), int(len(oids) - len(kept))


# --- 3. the baseline label -----------------------------------------------------

def sha1_of(oids: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(oids).tobytes()).hexdigest()


def split_against_baseline(oids: np.ndarray, baseline: np.ndarray) -> tuple:
    """(never processed, processed before). Labels only; changes nothing."""
    in_base = np.isin(oids, baseline)
    return int((~in_base).sum()), int(in_base.sum())


def check_baseline_fingerprint(baseline: np.ndarray, run_dir: Path) -> str:
    path = run_dir / "run.json"
    if not path.exists():
        return f"{path} does not exist -- baseline not verified"
    fp = json.loads(path.read_text())
    got, want = sha1_of(np.sort(baseline)), fp.get("oid_sha1")
    if got == want:
        return f"matches {path}"
    raise SystemExit(
        "\nBASELINE MISMATCH: the .npy is not the list that run produced.\n"
        f"  baseline : {len(baseline):,} oids, sha1={got}\n"
        f"  run.json : {fp.get('n_oids'):,} oids, sha1={want}\n"
        "  A rebuilt run.npy would mislabel the split. Pass the array the run used.")


# --- main ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ztf-list", required=True,
                    help="ZTF names: newline .txt, or .parquet (see --column).")
    ap.add_argument("--column", default="oid",
                    help="parquet column holding the names (default: oid).")
    ap.add_argument("--eligible", action="store_true",
                    help=f"keep only oids in {db.SCHEMA}.object with n_det >= --min-n-det.")
    ap.add_argument("--credentials", default=DEFAULT_CREDENTIALS)
    ap.add_argument("--min-n-det", type=int, default=DEFAULT_MIN_N_DET,
                    help="the run's cut (default 2, same as offline_setup.py).")
    ap.add_argument("--baseline", default=DEFAULT_BASELINE,
                    help="the full run's oid array; labels the list, changes nothing.")
    ap.add_argument("--run-dir",
                    help="the finished --out-dir; verifies the baseline against its run.json.")
    ap.add_argument("--out", default=str(OFFLINE / "oids" / "reprocesados.npy"),
                    help="the .npy for offline_run_batch.py --oid-file.")
    ap.add_argument("--dry-run", action="store_true",
                    help="convert and report, write nothing.")
    args = ap.parse_args()

    names = read_ztf_ids(args.ztf_list, args.column)
    print(f"read:     {len(names):,} names from {args.ztf_list}")
    oids = build_oid_array(names)
    print(f"encoded:  {len(oids):,} unique oids "
          f"({len(names) - len(oids):,} repeats dropped)")

    n_ineligible = None
    if args.eligible:
        print(f"selecting {db.SCHEMA}.object where n_det >= {args.min_n_det} ...", flush=True)
        catalogue = select_catalogue(args.credentials, args.min_n_det)
        oids, n_ineligible = keep_eligible(oids, catalogue)
        print(f"eligible: {len(oids):,} in the catalogue under the cut, "
              f"{n_ineligible:,} dropped (absent or n_det < {args.min_n_det})")

    baseline_note, n_new, n_repeat = "not read", None, None
    if Path(args.baseline).exists():
        baseline = np.load(args.baseline).astype(np.int64)
        baseline_note = f"{len(baseline):,} oids from {args.baseline}"
        if args.run_dir:
            baseline_note += " -- " + check_baseline_fingerprint(baseline, Path(args.run_dir))
        n_new, n_repeat = split_against_baseline(oids, baseline)
        print(f"baseline: {baseline_note}")
        print(f"          {n_new:,} never processed, {n_repeat:,} processed before "
              "(rows will be overwritten)")
    else:
        print(f"baseline: {args.baseline} not found -- list not labelled")

    print(f"list:     {len(oids):,} oids to process")

    report = {
        "schema": db.SCHEMA, "source": os.path.abspath(args.ztf_list),
        "n_names": int(len(names)), "eligible": bool(args.eligible),
        "min_n_det": args.min_n_det if args.eligible else None,
        "n_ineligible_dropped": n_ineligible,
        "baseline": baseline_note, "n_new": n_new, "n_repeat": n_repeat,
        "n_oids": int(len(oids)), "oid_sha1": sha1_of(oids),
        "out": os.path.abspath(args.out),
    }

    if args.dry_run:
        print("\n--dry-run: nothing written\n" + json.dumps(report, indent=2))
        return 0
    if not len(oids):
        print("\nnothing to do: the list is empty after filtering. Nothing written.")
        return 0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Through a file handle: np.save(path) appends ".npy" unless the name already
    # ends in it (same trap as offline_setup.step_oids).
    tmp = out.with_suffix(".npy.tmp")
    with open(tmp, "wb") as fh:
        np.save(fh, oids)
    os.replace(tmp, out)
    Path(str(out) + ".json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {len(oids):,} oids -> {out}")
    print(f"      report -> {out}.json")
    print("\nnext (note the FRESH --out-dir: unit indices belong to one oid array):\n"
          f"  poetry run python scripts/offline_run_batch.py \\\n"
          f"      --oid-file {out} --out-dir $RUN/bhrf_reproc \\\n"
          f"      --workers 64 --features \\\n"
          f"      --load-db --write-credentials {DEFAULT_CREDENTIALS} --no-shards")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
