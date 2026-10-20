#!/usr/bin/env python
"""Select the oids that still have >= N ZTF detections after the rb cut.

Input: an oid list (.npy or newline .txt). Output: the sorted subset whose
objects keep at least --min-detections detections once every detection with
rb < 0.55 is dropped (rb NULL counts as good), as a .npy plus a .json sidecar
with the counts.

Two passes, both read-only:
  1. pre-filter: keep the oids whose <schema>.object.n_det (sid 0) is already
     >= N; an object below N cannot reach N after a cut that only removes.
  2. rb count: per contiguous unit of --unit-size oids, a pool of workers counts
     the good detections in --minibatch round trips and writes the kept oids to
     <out-dir>/units/unit_NNNNN.npy.

A unit file is the done-marker: a rerun skips units whose file exists, and
run.json (the oid list + unit size fingerprint, shared with offline_run_batch)
refuses a resume against a different list. --out is written only when every
unit landed in this run's directory and --max-units was not given.

Typical use on cpu1:

    python scripts/offline_select_oids_rb.py \\
        --oid-file $RUN/oids/squidward_v210_oids.npy \\
        --out $RUN/oids/squidward_v210_oids_ndet8_rb.npy --out-dir $RUN/select_ndet8 \\
        --credentials features/offline/read_credentials.json --workers 32
"""
# --- thread pinning: MUST happen before numpy/BLAS are imported (see offline_run_batch) ---
import os  # env vars first

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):  # every BLAS/OpenMP knob
    os.environ.setdefault(_var, "1")  # one BLAS thread per worker process

import argparse  # CLI
import multiprocessing as mp  # start-method context
import signal  # workers ignore Ctrl-C; the parent handles it
import sys  # path setup
import time  # clocks
from concurrent.futures import ProcessPoolExecutor, as_completed  # the pool
from pathlib import Path  # paths

PIPE = Path(__file__).resolve().parents[2]  # .../pipeline
for _p in (PIPE / "feature_step", PIPE / "lc_classifier", PIPE / "libs" / "idmapper",
           PIPE / "libs" / "apf", PIPE / "libs" / "xmatch_client",
           PIPE / "alerce_classifiers", PIPE / "feature_step" / "scripts"):  # same roots as the siblings
    sys.path.insert(0, str(_p))  # importable without an install

import numpy as np  # oid arrays

import offline_run_batch as R  # units, fingerprint, retry, atomic json
from features.offline import db  # the two readers

DEFAULT_SCHEMA = "multisurvey"  # production; db.SCHEMA's default no longer exists

# Worker-process config, filled by _init_worker (inherited under fork, rebuilt under spawn).
_W: dict = {}  # cfg


def prefilter(oids: np.ndarray, credentials: str, min_detections: int, schema: str) -> np.ndarray:
    """Sorted unique oids of `oids` whose object.n_det (sid 0) is >= min_detections."""
    passing = db.fetch_oids_min_n_det(credentials, min_detections, schema=schema)  # sorted int64, whole table
    return np.intersect1d(np.unique(oids), passing, assume_unique=True).astype(np.int64)  # sorted, unique, int64


def unit_path(out_dir: Path, index: int) -> Path:
    """Where unit `index` writes its kept oids; its existence marks the unit done."""
    return Path(out_dir) / "units" / f"unit_{index:05d}.npy"  # one file per unit


def _init_worker(cfg: dict):
    """Per-process setup: ignore Ctrl-C and keep the run config."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # the parent handles Ctrl-C
    _W["cfg"] = cfg  # the run configuration


def process_unit(unit: tuple) -> dict:
    """Count good detections for one unit in minibatches; write its kept oids atomically."""
    index, oids = unit  # unit number and its oid slice
    cfg = _W["cfg"]  # run configuration
    t0 = time.perf_counter()  # unit clock
    kept = []  # kept oids per minibatch
    for start in range(0, len(oids), cfg["minibatch"]):  # one round trip per minibatch
        chunk = oids[start:start + cfg["minibatch"]]  # this minibatch
        kept.append(R._retry(lambda: db.fetch_oids_with_good_detections(
            cfg["credentials"], chunk, cfg["min_detections"], schema=cfg["schema"]),
            cfg["retries"], f"unit {index} rb count"))  # sorted subset of the chunk
    out = np.concatenate(kept).astype(np.int64) if kept else np.empty(0, dtype=np.int64)  # the unit's kept oids
    path = unit_path(Path(cfg["out_dir"]), index)  # final name
    path.parent.mkdir(parents=True, exist_ok=True)  # units/ on first write
    tmp = path.with_name(path.name + ".tmp")  # unit_NNNNN.npy.tmp
    with open(tmp, "wb") as fh:  # file handle, so np.save adds no extra suffix
        np.save(fh, out)  # write the array
    os.replace(tmp, path)  # atomic: the done-marker appears whole or not at all
    return {"unit": index, "n_in": int(len(oids)), "n_kept": int(len(out)),
            "seconds": round(time.perf_counter() - t0, 2)}  # what the parent prints


def merge_units(out_dir: Path, n_units: int) -> np.ndarray:
    """Concatenate units 0..n_units-1 in order; raise FileNotFoundError naming any missing unit."""
    missing = [i for i in range(n_units) if not unit_path(out_dir, i).exists()]  # units without a file
    if missing:  # an incomplete run must not produce an output
        raise FileNotFoundError(f"missing unit files for units {missing}")  # names them
    parts = [np.load(unit_path(out_dir, i)) for i in range(n_units)]  # each sorted, units contiguous
    return np.concatenate(parts).astype(np.int64) if parts else np.empty(0, dtype=np.int64)  # globally sorted


def run_pool(units: list, cfg: dict, workers: int, start_method: str) -> list:
    """Run `units` on a process pool; a unit that raises comes back as {"unit", "error"}."""
    ctx = mp.get_context(start_method)  # fork or spawn
    results = []  # one dict per unit
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx,
                             initializer=_init_worker, initargs=(cfg,)) as ex:  # workers get cfg
        futures = {ex.submit(process_unit, u): u[0] for u in units}  # all units queued
        for i, fut in enumerate(as_completed(futures), 1):  # as each unit lands
            try:  # the unit's result, or its exception
                m = fut.result()  # raises what process_unit raised
                print(f"[{i}/{len(units)}] unit {m['unit']}: {m['n_in']:,} -> {m['n_kept']:,}, "
                      f"{m['seconds']:.1f}s", flush=True)  # one line per unit
            except Exception as exc:  # a failed unit is data, not a crash
                m = {"unit": futures[fut], "error": repr(exc)}  # stays without a file; rerun retries it
                print(f"[{i}/{len(units)}] unit {m['unit']}: FAILED {m['error']}", flush=True)  # named in the log
            results.append(m)  # collected for main
    return results  # every unit, success or error


def main(argv=None) -> int:
    """Parse the flags, pre-filter by n_det, run the rb count per unit, merge into --out."""
    ap = argparse.ArgumentParser(  # flag names follow the sibling runners
        description="Select oids with >= N ZTF detections after dropping rb < 0.55.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)  # defaults in --help
    ap.add_argument("--oid-file", required=True, help="input oids, a .npy or newline .txt file.")  # the only input
    ap.add_argument("--out", help="output .npy of kept oids (+ .json sidecar).")  # final list
    ap.add_argument("--out-dir", help="run directory: run.json and units/.")  # checkpoint root
    ap.add_argument("--min-detections", type=int, default=8,
                    help="detections an object must keep after the rb cut.")  # N
    ap.add_argument("--credentials", default=R.DEFAULT_CREDENTIALS, help="read-only credentials JSON.")  # read account
    ap.add_argument("--schema", default=DEFAULT_SCHEMA, help="schema holding object and detection.")  # multisurvey
    ap.add_argument("--workers", type=int, default=R.default_workers(), help="worker processes.")  # pool size
    ap.add_argument("--unit-size", type=int, default=50_000, help="oids per unit = checkpoint granularity.")  # unit file per unit
    ap.add_argument("--minibatch", type=int, default=500, help="oids per DB round trip.")  # query size
    ap.add_argument("--max-units", type=int, help="run at most N units this pass (smoke); --out is not written.")  # smoke
    ap.add_argument("--retries", type=int, default=4, help="attempts per DB call before failing the unit.")  # R._retry
    ap.add_argument("--start-method", default=R.default_start_method(), choices=["fork", "spawn"],
                    help="process start method.")  # fork on Linux
    ap.add_argument("--force-resume", action="store_true",
                    help="resume even if run.json does not match (dangerous).")  # fingerprint override
    args = ap.parse_args(argv)  # the run's flags
    if not args.out or not args.out_dir:  # both are needed for a real run
        ap.error("--out and --out-dir are required")  # exits 2

    sys.stdout.reconfigure(line_buffering=True)  # progress through a redirected log
    t_run = time.perf_counter()  # run clock

    raw = R.load_oids(args.oid_file)  # input oids
    print(f"input: {len(raw):,} oids from {args.oid_file}")  # input count
    oids = prefilter(raw, args.credentials, args.min_detections, args.schema)  # n_det >= N, sorted unique
    print(f"after n_det >= {args.min_detections} pre-filter: {len(oids):,}")  # pre-filter count
    if len(oids) == 0:  # nothing to count
        print(f"no oid passed the n_det >= {args.min_detections} pre-filter")  # say so
        return 1  # nothing written

    out_dir = Path(args.out_dir)  # the run directory
    out_dir.mkdir(parents=True, exist_ok=True)  # created on first run
    R.check_or_write_fingerprint(out_dir, R.run_fingerprint(oids, args.unit_size),
                                 args.force_resume)  # refuses a different oid list
    units = R.make_units(oids, args.unit_size)  # contiguous slices
    todo = [u for u in units if not unit_path(out_dir, u[0]).exists()]  # units without a file
    print(f"plan: {len(units):,} units of {args.unit_size:,} ({args.minibatch}-oid minibatches, "
          f"{args.workers} workers); {len(units) - len(todo):,} already done, {len(todo):,} to go")  # the plan
    if args.max_units is not None:  # a smoke or probe
        todo = todo[:args.max_units]  # cap this pass
        print(f"--max-units: running {len(todo)} unit(s) this pass")  # say so

    cfg = {"credentials": args.credentials, "schema": args.schema,  # read side
           "min_detections": args.min_detections, "minibatch": args.minibatch,  # the count
           "retries": args.retries, "out_dir": str(out_dir)}  # resilience, output
    results = []  # this pass's unit results
    if todo:  # anything to run
        db.dispose_engines()  # psycopg2 sockets are not fork-safe
        results = run_pool(todo, cfg, args.workers, args.start_method)  # module global: tests replace it

    failed = [r for r in results if "error" in r]  # units that raised
    if failed or args.max_units is not None:  # partial run: no output
        for r in failed:  # name each one
            print(f"FAILED unit {r['unit']}: {r['error']}")  # named in the log
        done = sum(unit_path(out_dir, u[0]).exists() for u in units)  # unit files present
        print(f"state: {done:,}/{len(units):,} units done; {args.out} not written, rerun to resume")  # where it stands
        return 1  # incomplete

    kept = merge_units(out_dir, len(units))  # sorted, every unit present
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)  # output directory
    np.save(args.out, kept)  # the selected oids
    elapsed = round(time.perf_counter() - t_run, 2)  # whole run
    R._write_json(Path(args.out + ".json"), {
        "n_input": int(len(raw)), "n_after_n_det": int(len(oids)), "n_after_rb": int(len(kept)),  # counts
        "min_detections": args.min_detections, "schema": args.schema,  # the rule
        "unit_size": args.unit_size, "elapsed_s": elapsed})  # the run
    print(f"done: {len(raw):,} input -> {len(oids):,} after n_det -> {len(kept):,} after rb; "
          f"wrote {args.out} ({elapsed:.0f}s)")  # final counts
    return 0  # success


if __name__ == "__main__":  # script entry point
    sys.exit(main())  # the exit code
