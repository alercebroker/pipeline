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

The pre-filtered list is saved once as <out-dir>/prefiltered.npy; every later
pass on the same --out-dir loads it instead of querying object again, so the
unit slices never shift under a resume. --out must end in .npy.

Typical use on cpu1: a one-unit smoke first (exits 1 with "not written, rerun
to resume"), then the full run on the same --out-dir, which resumes from it:

    python scripts/offline_select_oids_rb.py \\
        --oid-file $RUN/oids/squidward_v210_oids.npy \\
        --out $RUN/oids/squidward_v210_oids_ndet8_rb.npy --out-dir $RUN/select_ndet8 \\
        --credentials features/offline/read_credentials.json --workers 32 --max-units 1

    python scripts/offline_select_oids_rb.py \\
        --oid-file $RUN/oids/squidward_v210_oids.npy \\
        --out $RUN/oids/squidward_v210_oids_ndet8_rb.npy --out-dir $RUN/select_ndet8 \\
        --credentials features/offline/read_credentials.json --workers 32
"""
# --- thread pinning: MUST happen before numpy/BLAS are imported (see offline_run_batch) ---
import os  # env vars first

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",  # every BLAS/OpenMP knob,
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):  # continued
    os.environ.setdefault(_var, "1")  # one BLAS thread per worker process

import argparse  # CLI
import hashlib  # digest of the raw input list
import multiprocessing as mp  # start-method context
import signal  # workers ignore Ctrl-C; the parent handles it
import sys  # path setup
import time  # clocks
from concurrent.futures import ProcessPoolExecutor, as_completed  # the pool
from pathlib import Path  # paths

PIPE = Path(__file__).resolve().parents[2]  # .../pipeline
for _p in (PIPE / "feature_step", PIPE / "lc_classifier", PIPE / "libs" / "idmapper",  # same roots as the siblings,
           PIPE / "libs" / "apf", PIPE / "libs" / "xmatch_client",  # continued,
           PIPE / "alerce_classifiers", PIPE / "feature_step" / "scripts"):  # plus scripts/ for offline_run_batch
    sys.path.insert(0, str(_p))  # importable without an install

import numpy as np  # oid arrays

import offline_run_batch as R  # units, fingerprint, retry, atomic json
from features.offline import db  # the two readers

DEFAULT_SCHEMA = "multisurvey"  # production; db.SCHEMA's default no longer exists

# Worker-process config, filled by _init_worker (inherited under fork, rebuilt under spawn).
_W: dict = {}  # cfg


def prefilter(oids: np.ndarray, credentials: str, min_detections: int, schema: str) -> np.ndarray:  # n_det pre-filter
    """Sorted unique oids of `oids` whose object.n_det (sid 0) is >= min_detections."""
    passing = db.fetch_oids_min_n_det(credentials, min_detections, schema=schema)  # sorted int64, whole table
    return np.intersect1d(np.unique(oids), passing, assume_unique=True).astype(np.int64)  # sorted, unique, int64


def unit_path(out_dir: Path, index: int) -> Path:  # a unit's output file
    """Where unit `index` writes its kept oids; its existence marks the unit done."""
    return Path(out_dir) / "units" / f"unit_{index:05d}.npy"  # one file per unit


def _save_npy_atomic(path: Path, arr: np.ndarray) -> None:  # atomic .npy writer
    """Write `arr` to `path` exactly (no suffix added) through a .tmp file and os.replace."""
    tmp = path.with_name(path.name + ".tmp")  # <name>.tmp next to the final file
    with open(tmp, "wb") as fh:  # file handle, so np.save adds no extra suffix
        np.save(fh, arr)  # write the array
    os.replace(tmp, path)  # atomic: the file appears whole or not at all


def _init_worker(cfg: dict):  # per-process setup
    """Per-process setup: ignore Ctrl-C and keep the run config."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # the parent handles Ctrl-C
    _W["cfg"] = cfg  # the run configuration


def process_unit(unit: tuple) -> dict:  # the worker's job
    """Count good detections for one unit in minibatches; write its kept oids atomically."""
    index, oids = unit  # unit number and its oid slice
    cfg = _W["cfg"]  # run configuration
    t0 = time.perf_counter()  # unit clock
    kept = []  # kept oids per minibatch
    for start in range(0, len(oids), cfg["minibatch"]):  # one round trip per minibatch
        chunk = oids[start:start + cfg["minibatch"]]  # this minibatch
        kept.append(R._retry(lambda: db.fetch_oids_with_good_detections(  # rb count with backoff:
            cfg["credentials"], chunk, cfg["min_detections"], schema=cfg["schema"]),  # reader arguments,
            cfg["retries"], f"unit {index} rb count"))  # attempts and label; sorted subset of the chunk
    out = np.concatenate(kept).astype(np.int64) if kept else np.empty(0, dtype=np.int64)  # the unit's kept oids
    path = unit_path(Path(cfg["out_dir"]), index)  # final name
    path.parent.mkdir(parents=True, exist_ok=True)  # units/ on first write
    _save_npy_atomic(path, out)  # the done-marker, last
    return {"unit": index, "n_in": int(len(oids)), "n_kept": int(len(out)),  # counts and
            "seconds": round(time.perf_counter() - t0, 2)}  # wall clock, printed by the parent


def merge_units(out_dir: Path, n_units: int) -> np.ndarray:  # unit files -> one array
    """Concatenate units 0..n_units-1 in order; raise FileNotFoundError naming any missing unit."""
    missing = [i for i in range(n_units) if not unit_path(out_dir, i).exists()]  # units without a file
    if missing:  # an incomplete run must not produce an output
        raise FileNotFoundError(f"missing unit files for units {missing}")  # names them
    parts = [np.load(unit_path(out_dir, i)) for i in range(n_units)]  # each sorted, units contiguous
    return np.concatenate(parts).astype(np.int64) if parts else np.empty(0, dtype=np.int64)  # globally sorted


def run_pool(units: list, cfg: dict, workers: int, start_method: str) -> list:  # the only pool user
    """Run `units` on a process pool; a unit that raises comes back as {"unit", "error"}."""
    ctx = mp.get_context(start_method)  # fork or spawn
    results = []  # one dict per unit
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx,  # pool size and start method,
                             initializer=_init_worker, initargs=(cfg,)) as ex:  # workers get cfg
        futures = {ex.submit(process_unit, u): u[0] for u in units}  # all units queued
        for i, fut in enumerate(as_completed(futures), 1):  # as each unit lands
            try:  # the unit's result, or its exception
                m = fut.result()  # raises what process_unit raised
                print(f"[{i}/{len(units)}] unit {m['unit']}: {m['n_in']:,} -> {m['n_kept']:,}, "  # counts,
                      f"{m['seconds']:.1f}s", flush=True)  # one line per unit
            except Exception as exc:  # a failed unit is data, not a crash
                m = {"unit": futures[fut], "error": repr(exc)}  # stays without a file; rerun retries it
                print(f"[{i}/{len(units)}] unit {m['unit']}: FAILED {m['error']}", flush=True)  # named in the log
            results.append(m)  # collected for main
    return results  # every unit, success or error


def main(argv=None) -> int:  # CLI entry
    """Parse the flags, pre-filter by n_det, run the rb count per unit, merge into --out."""
    ap = argparse.ArgumentParser(  # flag names follow the sibling runners
        description="Select oids with >= N ZTF detections after dropping rb < 0.55.",  # --help header
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)  # defaults in --help
    ap.add_argument("--oid-file", required=True, help="input oids, a .npy or newline .txt file.")  # the only input
    ap.add_argument("--out", required=True,  # final list,
                    help="output .npy of kept oids, written as named (+ .json sidecar).")  # no suffix added
    ap.add_argument("--out-dir", required=True, help="run directory: run.json and units/.")  # checkpoint root
    ap.add_argument("--min-detections", type=int, default=8,  # N,
                    help="detections an object must keep after the rb cut.")  # its help
    ap.add_argument("--credentials", default=R.DEFAULT_CREDENTIALS, help="read-only credentials JSON.")  # read account
    ap.add_argument("--schema", default=DEFAULT_SCHEMA, help="schema holding object and detection.")  # multisurvey
    ap.add_argument("--workers", type=int, default=R.default_workers(), help="worker processes.")  # pool size
    ap.add_argument("--unit-size", type=int, default=50_000, help="oids per unit = checkpoint granularity.")  # unit file per unit
    ap.add_argument("--minibatch", type=int, default=500, help="oids per DB round trip.")  # query size
    ap.add_argument("--max-units", type=int, help="run at most N units this pass (smoke); --out is not written.")  # smoke
    ap.add_argument("--retries", type=int, default=4, help="attempts per DB call before failing the unit.")  # R._retry
    ap.add_argument("--start-method", default=R.default_start_method(), choices=["fork", "spawn"],  # fork on Linux,
                    help="process start method.")  # its help
    ap.add_argument("--force-resume", action="store_true",  # fingerprint override,
                    help="resume even if run.json does not match (dangerous).")  # its help
    args = ap.parse_args(argv)  # the run's flags

    sys.stdout.reconfigure(line_buffering=True)  # progress through a redirected log
    t_run = time.perf_counter()  # run clock

    if not args.out.endswith(".npy"):  # the consumer's loader keys on the suffix
        print("--out must end in .npy (R.load_oids reads anything else as text)")  # say why
        return 1  # before any DB work

    raw = np.unique(R.load_oids(args.oid_file))  # input oids, sorted and deduplicated
    print(f"input: {len(raw):,} unique oids from {args.oid_file}")  # input count
    out_dir = Path(args.out_dir)  # the run directory
    out_dir.mkdir(parents=True, exist_ok=True)  # created on first run
    pre_path = out_dir / "prefiltered.npy"  # the persisted pre-filter: n_det only grows, so never re-query
    if pre_path.exists():  # a resume
        oids = np.load(pre_path).astype(np.int64)  # the list the units were cut from
        print(f"resume: pre-filtered list from {pre_path} ({len(oids):,} oids)")  # no DB query
    else:  # first pass
        oids = prefilter(raw, args.credentials, args.min_detections, args.schema)  # n_det >= N, sorted unique
        print(f"after n_det >= {args.min_detections} pre-filter: {len(oids):,}")  # pre-filter count
        if len(oids) == 0:  # nothing to count
            print(f"no oid passed the n_det >= {args.min_detections} pre-filter")  # say so
            return 1  # nothing written
        _save_npy_atomic(pre_path, oids)  # frozen for every later pass

    fingerprint = R.run_fingerprint(oids, args.unit_size)  # persisted oid list + unit size
    fingerprint["min_detections"] = args.min_detections  # a resume under another N is refused
    fingerprint["schema"] = args.schema  # and under another schema
    fingerprint["input_sha1"] = hashlib.sha1(np.ascontiguousarray(raw).tobytes()).hexdigest()  # and another --oid-file
    fingerprint["n_input"] = int(len(raw))  # its unique size, readable in run.json
    R.check_or_write_fingerprint(out_dir, fingerprint, args.force_resume)  # refuses a different run
    units = R.make_units(oids, args.unit_size)  # contiguous slices
    todo = [u for u in units if not unit_path(out_dir, u[0]).exists()]  # units without a file
    print(f"plan: {len(units):,} units of {args.unit_size:,} ({args.minibatch}-oid minibatches, "  # the plan,
          f"{args.workers} workers); {len(units) - len(todo):,} already done, {len(todo):,} to go")  # and progress
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
    missing = [u[0] for u in units if not unit_path(out_dir, u[0]).exists()]  # units without a file
    done = len(units) - len(missing)  # unit files present
    if failed or args.max_units is not None or missing:  # partial run (incl. units that never landed): no output
        for r in failed:  # name each one
            print(f"FAILED unit {r['unit']}: {r['error']}")  # named in the log
        for i in missing:  # name every unit without a file
            print(f"MISSING unit {i}")  # failed, capped or never landed
        print(f"state: {done:,}/{len(units):,} units done; {args.out} not written, rerun to resume")  # where it stands
        return 1  # incomplete

    kept = merge_units(out_dir, len(units))  # sorted, every unit present
    out = Path(args.out)  # exactly the name given
    out.parent.mkdir(parents=True, exist_ok=True)  # output directory
    _save_npy_atomic(out, kept)  # the selected oids, atomically
    elapsed = round(time.perf_counter() - t_run, 2)  # whole run
    R._write_json(Path(args.out + ".json"), {  # sidecar next to --out:
        "n_input": int(len(raw)), "n_after_n_det": int(len(oids)), "n_after_rb": int(len(kept)),  # counts
        "min_detections": args.min_detections, "schema": args.schema,  # the rule
        "unit_size": args.unit_size, "elapsed_s": elapsed})  # the run
    print(f"done: {len(raw):,} input -> {len(oids):,} after n_det -> {len(kept):,} after rb; "  # final counts,
          f"wrote {args.out} ({elapsed:.0f}s)")  # and where
    return 0  # success


if __name__ == "__main__":  # script entry point
    sys.exit(main())  # the exit code
