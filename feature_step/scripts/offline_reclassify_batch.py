#!/usr/bin/env python
"""Batched, multi-process reclassification: stored features -> BHRF probabilities.

The sibling runner (offline_run_batch.py) reads light curves, computes the 199
features and classifies. This one skips the computation: for every oid it reads
the rows <schema>.feature already holds, rebuilds the model's frame
(features/offline/reclassify.py), classifies the minibatch in one model call and
upserts the 45 probability rows per object under the requested classifier
version. Nothing else is written: no features, no xmatch.

What it shares with offline_run_batch.py, imported from it rather than copied:
the oid file loader, contiguous work units of --unit-size oids, the per-unit
manifest as the done-marker (resume skips a unit whose manifest exists), the
run.json fingerprint that refuses a different oid list, the stale .tmp cleanup,
the atomic writers and the peak-RSS reading. The model is loaded once in the
parent and shared copy-on-write under fork, as there.

Per oid, the manifest separates three ways of producing nothing:
  * n_no_features : no row in <schema>.feature for it (nothing to classify)
  * n_no_lastmjd  : no stored probability row of --lastmjd-classifier to take
                    lastmjd from (probability.lastmjd is NOT NULL)
  * n_errors      : an exception, or fewer rows than the taxonomy holds (45),
                    written in full to errors/unit_*.jsonl

Use a FRESH --out-dir: run.json carries the model version and the schema, so a
directory left by the sibling runner is refused, and --force-resume would take
its manifests as done and skip those units.

Typical use (the squidward 2.2.0 rerun over the oids of the 2.1.0 dump):

    export MODEL_PATH=https://alerce-models.s3.amazonaws.com/squidward/2.2.0/hierarchical_random_forest_model.pkl
    python scripts/offline_reclassify_batch.py \\
        --oid-file $RUN/oids/squidward_v210_oids.npy --out-dir $RUN/reclass_220 \\
        --schema multisurvey --model-version 2.2.0 \\
        --credentials features/offline/credentials.json \\
        --workers 32 --stall-timeout 7200 \\
        --load-db --write-credentials features/offline/credentials.json --no-shards
"""
# --- thread pinning: MUST happen before numpy/BLAS are imported (see offline_run_batch) ---
import os  # env vars first

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):  # every BLAS/OpenMP knob
    os.environ.setdefault(_var, "1")  # one BLAS thread per worker process
os.environ.setdefault("JAX_PLATFORMS", "cpu")  # no accelerator probing in workers

import argparse  # CLI
import multiprocessing as mp  # start-method context
import signal  # workers ignore Ctrl-C; the parent handles it
import sys  # path setup and exit code
import time  # per-phase clocks
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait  # the pool
from concurrent.futures.process import BrokenProcessPool  # a dead worker is fatal, not silent
from pathlib import Path  # paths

PIPE = Path(__file__).resolve().parents[2]  # .../pipeline
for _p in (PIPE / "feature_step", PIPE / "lc_classifier", PIPE / "libs" / "idmapper",
           PIPE / "libs" / "apf", PIPE / "libs" / "xmatch_client",
           PIPE / "alerce_classifiers", PIPE / "feature_step" / "scripts"):  # same roots as the sibling, plus scripts/
    sys.path.insert(0, str(_p))  # importable without an install

import numpy as np  # oid arrays
import pandas as pd  # frames
import sqlalchemy as sa  # write-credentials preflight

import offline_run_batch as R  # the shared unit/manifest/fingerprint machinery
from features.offline import db  # readers
from features.offline import reclassify  # rows -> frame -> per-oid output
from features.offline.classify import load_squidward_model  # model from MODEL_PATH
from features.offline.probability_writer import (  # rows and the upsert
    CLASSIFIER_IDS, build_probability_rows, write_probabilities)

DEFAULT_CREDENTIALS = R.DEFAULT_CREDENTIALS  # read_credentials.json next to the package
DEFAULT_SCHEMA = "multisurvey"  # production; multisurvey_ztf (db.SCHEMA's default) no longer exists
DEFAULT_MODEL_VERSION = "2.2.0"  # the artifact this runner was written for
SID_ZTF = R.SID_ZTF  # ZTF

# Worker-process singletons, built once by _init_worker (inherited under fork).
_W: dict = {}  # cfg, model, feature_list, taxonomy, rows_per_oid
_MODEL = None  # loaded in the parent under fork, shared copy-on-write


def fetch_minibatch(oids: list, cfg: dict) -> tuple:
    """Two batched reads for `oids`: (feature rows, {oid: lastmjd}).

    Feature rows are already filtered to each oid's newest updated_date. Both
    reads retry with backoff like the sibling runner's; a persistent failure
    aborts the unit, which stays unmarked and is redone on resume.
    """
    cred, retries, schema = cfg["credentials"], cfg["retries"], cfg["schema"]  # from main()
    rows = R._retry(lambda: db.fetch_feature_rows(cred, oids, schema=schema),
                    retries, "feature rows")  # ~190 rows per oid
    lastmjd = R._retry(lambda: db.fetch_probability_lastmjd(
        cred, oids, classifier_id=cfg["lastmjd_classifier"], schema=schema),
        retries, "lastmjd")  # one value per oid that has a stored row
    return reclassify.current_rows(rows), lastmjd  # stale rows dropped here, once


def classify_minibatch(rows: pd.DataFrame):
    """Feature rows -> (batched OutputDTO, the oids that had a feature frame)."""
    frame = reclassify.features_frame(rows, _W["feature_list"])  # one row per oid, model columns
    return reclassify.classify_frame(frame, _W["model"]), list(frame.index)  # one model call


def _init_worker(cfg: dict):
    """Per-process setup: model, taxonomy maps, the model's feature list and the row count per oid."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # the parent handles Ctrl-C
    if not cfg["warnings"]:  # same pandas/numpy noise filter as the sibling
        R._silence_library_warnings()  # per-object warnings would drown the log
    _W["cfg"] = cfg  # the run configuration
    _W["model"] = _MODEL if _MODEL is not None else load_squidward_model(
        expected_version=cfg["model_version"])[0]  # inherited under fork, loaded under spawn
    _W["feature_list"] = list(_W["model"].model.feature_list)  # the 199 column names, from the pickle
    _W["taxonomy"] = db.fetch_taxonomy_maps(cfg["credentials"], CLASSIFIER_IDS,
                                            schema=cfg["schema"])  # class_name -> class_id per classifier
    _W["rows_per_oid"] = sum(len(_W["taxonomy"][c]) for c in CLASSIFIER_IDS)  # 21+3+6+6+9 = 45


def process_unit(unit) -> dict:
    """Run one work unit: read, classify, upsert, manifest. Returns the manifest dict."""
    index, oids = unit  # unit number and its oid slice
    cfg = _W["cfg"]  # run configuration
    out_dir = Path(cfg["out_dir"])  # manifests, errors and optional shards live here
    t0 = time.perf_counter()  # unit clock

    oids = [int(o) for o in oids]  # plain ints for SQL and json
    prob_rows = []  # every probability row of the unit
    n_ok = n_errors = n_no_features = n_no_lastmjd = 0  # the four outcomes
    t_fetch = t_compute = t_write = 0.0  # where the unit's time goes
    failed = []  # every failed oid, for errors/unit_*.jsonl
    expected = _W["rows_per_oid"]  # rows a fully classified oid must yield

    for start in range(0, len(oids), cfg["minibatch"]):  # one round trip's worth at a time
        mb = oids[start:start + cfg["minibatch"]]  # this minibatch
        t_mb = time.perf_counter()  # fetch clock
        rows, lastmjd = fetch_minibatch(mb, cfg)  # two queries for the minibatch
        t_fetch += time.perf_counter() - t_mb  # add the fetch
        t_mb = time.perf_counter()  # compute clock
        dto, with_features = classify_minibatch(rows)  # one model call
        present = set(with_features)  # oids that had a feature frame
        for oid in mb:  # per-oid bookkeeping; the model already ran
            if oid not in present:  # no stored features: nothing to classify
                n_no_features += 1  # not a failure, not retried
                continue  # next oid
            if oid not in lastmjd:  # nothing to stamp lastmjd with
                n_no_lastmjd += 1  # not a failure, not retried
                continue  # next oid
            try:  # one bad object does not kill the unit
                p_rows = build_probability_rows(
                    reclassify.split_output(dto, oid), oid, lastmjd[oid], _W["taxonomy"],
                    version=cfg["model_version"], sid=SID_ZTF)  # 45 rows under the run's version
                if len(p_rows) != expected:  # a branch frame without the oid, or an empty prediction
                    raise ValueError(f"got {len(p_rows)} probability rows, expected {expected}")  # refuse a partial set
            except Exception as exc:  # recorded by oid; the unit goes on
                n_errors += 1  # worth a retry
                failed.append({"oid": oid, "error": f"{type(exc).__name__}: {exc}"})  # the full list
                continue  # next oid
            prob_rows.extend(p_rows)  # collected for one write per unit
            n_ok += 1  # exactly `expected` rows landed
        t_compute += time.perf_counter() - t_mb  # add the compute
        print(f"  mb unit {index} {start // cfg['minibatch'] + 1:>2}/"
              f"{-(-len(oids) // cfg['minibatch'])} fetch {t_fetch:6.0f}s "
              f"compute {t_compute:6.0f}s ok {n_ok}", flush=True)  # grep -E '^  mb ' finds these

    if not cfg.get("no_shards"):  # the shards are the output unless the DB holds the rows
        R._write_shard(out_dir / "probabilities" / f"unit_{index:07d}.parquet",
                       pd.DataFrame(prob_rows))  # atomic parquet

    n_db_prob = 0  # rows the DB accepted
    t_w = time.perf_counter()  # write clock
    if cfg.get("load_db") and prob_rows:  # one upsert per unit, before the manifest
        n_db_prob = write_probabilities(prob_rows, cfg["write_credentials"],
                                        schema=cfg["schema"], execute=True)["written"]  # commits or raises
    t_write = time.perf_counter() - t_w  # the write

    R._write_jsonl(out_dir / "errors" / f"unit_{index:07d}.jsonl", failed)  # before the manifest

    manifest = {  # what the unit leaves behind; the done-marker
        "unit": index, "oid_lo": oids[0], "oid_hi": oids[-1], "n_oids": len(oids),  # which oids
        "n_ok": n_ok, "n_errors": n_errors,  # classified / failed
        "n_skipped": n_errors + n_no_features + n_no_lastmjd,  # total, for older readers
        "n_no_features": n_no_features, "n_no_lastmjd": n_no_lastmjd,  # the two silent skips
        "prob_rows": len(prob_rows), "db_prob_rows": n_db_prob,  # must match when --load-db
        "elapsed_s": round(time.perf_counter() - t0, 2),  # wall clock of the unit
        "t_fetch_s": round(t_fetch, 2), "t_compute_s": round(t_compute, 2),  # per phase
        "t_write_s": round(t_write, 2),  # the upsert
        "peak_rss_mb": R.worker_peak_rss_mb(),  # memory of this worker so far
        "errors": failed[:20],  # sample; errors/unit_*.jsonl has all of them
    }
    R._write_json(out_dir / "manifests" / f"unit_{index:07d}.json", manifest)  # the done-marker, last
    return manifest  # aggregated by the parent


def main():
    """Parse the flags, plan the units, preflight the DB and the model, run the pool."""
    ap = argparse.ArgumentParser(  # same flag names as the sibling where they mean the same
        description="Batched multi-process BHRF reclassification from stored features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)  # defaults in --help
    ap.add_argument("--oid-file", required=True,
                    help="oids to reclassify, a .npy or newline .txt file.")  # the only input
    ap.add_argument("--out-dir", help="output root for manifests (+ shards).")  # fresh per run
    ap.add_argument("--model-version", default=DEFAULT_MODEL_VERSION,
                    help="version the model at MODEL_PATH must report and the rows are stamped with.")  # 2.2.0 -> 220
    ap.add_argument("--lastmjd-classifier", type=int, default=5,
                    help="classifier_id whose stored probability row lends lastmjd (5 = flat).")  # where lastmjd comes from
    ap.add_argument("--workers", type=int, default=R.default_workers(),
                    help="worker processes (physical cores - 2 here).")  # pool size
    ap.add_argument("--unit-size", type=int, default=5000,
                    help="oids per work unit = checkpoint granularity.")  # manifest per unit
    ap.add_argument("--minibatch", type=int, default=500,
                    help="oids per batched DB round trip and per model call.")  # ~95k feature rows
    ap.add_argument("--max-units", type=int, help="stop after N units (smoke / probe).")  # smoke
    ap.add_argument("--load-db", action="store_true", dest="load_db",
                    help="upsert each finished unit into <schema>.probability. "
                         "Requires --write-credentials.")  # the real run
    ap.add_argument("--no-shards", action="store_true", dest="no_shards",
                    help="skip the parquet shards; requires --load-db.")  # DB holds the rows
    ap.add_argument("--write-credentials", dest="write_credentials",
                    help="credentials JSON with INSERT rights, required by --load-db.")  # write account
    ap.add_argument("--retries", type=int, default=4,
                    help="attempts per DB call before failing the unit.")  # backoff in R._retry
    ap.add_argument("--credentials", default=DEFAULT_CREDENTIALS)  # read account
    ap.add_argument("--schema", default=DEFAULT_SCHEMA)  # multisurvey
    ap.add_argument("--start-method", default=R.default_start_method(),
                    choices=["fork", "spawn"],
                    help="fork shares the loaded model copy-on-write (Linux).")  # spawn on macOS
    ap.add_argument("--stall-timeout", type=float, default=1800.0,
                    help="abort if no unit completes for this many seconds.")  # watchdog
    ap.add_argument("--force-resume", action="store_true",
                    help="resume even if run.json does not match (dangerous).")  # see module docstring
    ap.add_argument("--warnings", action="store_true",
                    help="keep pandas/numpy deprecation warnings.")  # off by default
    ap.add_argument("--plan-only", action="store_true",
                    help="load the oids, report the plan, and exit.")  # no DB, no model
    args = ap.parse_args()  # the run's flags

    if args.load_db and not args.write_credentials:  # validate before any work
        ap.error("--load-db requires --write-credentials")  # exits 2
    if R.shards_would_be_lost(args.no_shards, args.load_db):  # no output at all otherwise
        ap.error("--no-shards without --load-db would discard the run's only output")  # exits 2

    sys.stdout.reconfigure(line_buffering=True)  # progress through a redirected log

    oids = np.sort(R.load_oids(args.oid_file))  # sorted: contiguous units, stable fingerprint
    print(f"loaded {len(oids):,} oids from {args.oid_file}")  # the input size
    units = R.make_units(oids, args.unit_size)  # contiguous slices
    print(f"host: {R.physical_cores()} physical cores / {os.cpu_count()} hw threads, "
          f"start-method={args.start_method}")  # where it runs
    print(f"plan: {len(oids):,} oids -> {len(units):,} units of {args.unit_size} "
          f"({args.minibatch}-oid minibatches, {args.workers} workers), "
          f"model {args.model_version} -> {args.schema}.probability")  # what it will do
    if args.plan_only:  # nothing else was asked
        return 0  # success
    if not args.out_dir:  # every real run needs a directory
        ap.error("--out-dir is required unless --plan-only")  # exits 2

    out_dir = Path(args.out_dir)  # the run directory
    out_dir.mkdir(parents=True, exist_ok=True)  # created on first run
    fingerprint = R.run_fingerprint(oids, args.unit_size)  # oid list + unit size
    fingerprint["model_version"] = args.model_version  # a resume under another version is refused too
    fingerprint["schema"] = args.schema  # and under another schema
    R.check_or_write_fingerprint(out_dir, fingerprint, args.force_resume)  # writes run.json the first time
    stale = R.clean_stale_tmp(out_dir)  # a hard kill leaves .tmp files
    if stale:  # say so
        print(f"cleaned {stale} stale .tmp file(s) from an interrupted run")  # informational
    done = {int(p.stem.split("_")[1])
            for p in (out_dir / "manifests").glob("unit_*.json")}  # a manifest means done
    todo = [u for u in units if u[0] not in done]  # the rest
    if done:  # a resume
        print(f"resume: {len(done):,} units already done, {len(todo):,} to go")  # progress so far
    if args.max_units:  # a smoke or probe
        todo = todo[:args.max_units]  # cap this pass
        print(f"--max-units: running {len(todo)} unit(s) this pass")  # say so
    if not todo:  # everything already has a manifest
        print("nothing to do.")  # done
        return 0  # success

    cfg = {  # what every worker needs; picklable
        "credentials": args.credentials, "schema": args.schema,  # read side
        "load_db": args.load_db, "write_credentials": args.write_credentials,  # write side
        "no_shards": args.no_shards, "out_dir": str(out_dir),  # outputs
        "minibatch": args.minibatch, "retries": args.retries,  # batching
        "warnings": args.warnings, "model_version": args.model_version,  # model
        "lastmjd_classifier": args.lastmjd_classifier,  # lastmjd source
    }

    # Preflights in the parent, so a wrong flag fails before N workers fork.
    taxonomy = db.fetch_taxonomy_maps(args.credentials, CLASSIFIER_IDS, schema=args.schema)  # read account
    missing = [c for c in CLASSIFIER_IDS if c not in taxonomy]  # a classifier with no classes
    if missing:  # nothing could be stamped for it
        raise SystemExit(f"{args.schema}.taxonomy has no rows for classifier_id {missing}; "
                         "seed the taxonomy before writing probabilities")  # exits 1
    print(f"taxonomy from {args.schema}.taxonomy: "
          + ", ".join(f"{c}:{len(taxonomy[c])}" for c in CLASSIFIER_IDS))  # expect 21, 3, 6, 6, 9
    if args.load_db:  # the write account must connect before anything is computed
        with db._make_engine(args.write_credentials).connect() as _conn:  # pooled, disposed below
            _conn.execute(sa.text("SELECT 1"))  # round trip
        print(f"load-db: ON -> {args.schema}.probability")  # the only table written

    if not args.warnings:  # the parent loads the model too
        R._silence_library_warnings()  # same filter as the workers

    global _MODEL  # shared with the workers under fork
    if args.start_method == "fork":  # only fork can share the parent's copy
        print("loading BHRF model in the parent (shared copy-on-write by fork)...")  # 1.7 GB pickle
        _MODEL, mname, mversion = load_squidward_model(expected_version=args.model_version)  # refuses another version
        print(f"  model: {mname} version={mversion}")  # what was loaded

    db.dispose_engines()  # psycopg2 sockets are not fork-safe

    ctx = mp.get_context(args.start_method)  # fork or spawn
    n_units = len(todo)  # this pass
    t_run = time.perf_counter()  # run clock
    agg = {"n_ok": 0, "n_skipped": 0, "n_errors": 0, "n_no_features": 0,
           "n_no_lastmjd": 0, "prob_rows": 0, "db_prob_rows": 0}  # run totals, summed from manifests
    n_failed = 0  # units that raised

    try:  # the pool; BrokenProcessPool and Ctrl-C handled below
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx,
                                 initializer=_init_worker, initargs=(cfg,)) as ex:  # not mp.Pool: a dead worker raises
            futures = {ex.submit(process_unit, u): u[0] for u in todo}  # all units queued
            pending, i = set(futures), 0  # in flight, finished count
            last_progress = time.monotonic()  # for the stall watchdog
            while pending:  # until every unit landed or the watchdog fires
                finished, pending = wait(pending, timeout=args.stall_timeout,
                                         return_when=FIRST_COMPLETED)  # wake per unit
                if not finished:  # nothing landed within the timeout
                    stalled = time.monotonic() - last_progress  # silence so far
                    print(f"WARNING: no unit completed in {stalled / 60:.1f} min "
                          f"({len(pending)} still running)", flush=True)  # say so
                    if stalled > args.stall_timeout:  # too long: abort, units are checkpointed
                        for fut in pending:  # drop the queue
                            fut.cancel()  # not-yet-started units
                        raise SystemExit(
                            f"\nABORTING: stalled for {stalled / 60:.1f} min with no "
                            f"progress. Finished units are checkpointed; rerun to resume.")  # exits 1
                    continue  # wait again
                last_progress = time.monotonic()  # something landed
                for fut in finished:  # usually one
                    i += 1  # finished count
                    try:  # the unit's manifest, or its exception
                        man = fut.result()  # raises what process_unit raised
                    except Exception as exc:  # one unit failing is survivable
                        if isinstance(exc, BrokenProcessPool):  # only a dead worker is fatal
                            raise  # handled below
                        n_failed += 1  # the unit stays unmarked; a rerun retries it
                        print(f"[{i}/{n_units}] unit {futures[fut]:>7} FAILED: "
                              f"{type(exc).__name__}: {exc} -- left unmarked, "
                              f"rerun to retry", flush=True)  # named in the log
                        continue  # next future
                    for k in agg:  # run totals
                        agg[k] += man[k]  # every key exists in the manifest
                    elapsed = time.perf_counter() - t_run  # so far
                    rate = agg["n_ok"] / elapsed if elapsed else 0.0  # oids per second
                    eta = (n_units - i) * (elapsed / i) if i else 0.0  # linear projection
                    print(f"[{i}/{n_units}] unit {man['unit']:>7} "
                          f"ok={man['n_ok']:>5} skip={man['n_skipped']:>4} "
                          f"err={man['n_errors']:>3} {man['elapsed_s']:>7.1f}s "
                          f"(fetch {man['t_fetch_s']:.0f} compute {man['t_compute_s']:.0f} "
                          f"write {man['t_write_s']:.0f}) | "
                          f"{rate:6.1f} oid/s  ETA {eta / 3600:5.2f}h", flush=True)  # one line per unit
    except BrokenProcessPool as exc:  # a worker died without raising: native crash or OOM
        raise SystemExit(
            f"\nABORTING: a worker process died abruptly ({exc}).\n"
            f"  start method: {args.start_method}\n"
            "  Try --start-method spawn, or fewer --workers if memory-bound.\n"
            "  Finished units are checkpointed; rerun to resume.")  # exits 1
    except KeyboardInterrupt:  # Ctrl-C in the parent
        print("\ninterrupted -- finished units are checkpointed; "
              "rerun the same command to resume.")  # the pool drains in-flight units first
        return 130  # the conventional code

    elapsed = time.perf_counter() - t_run  # whole pass
    print("\n" + "=" * 70)  # summary block
    print(f"  units          : {n_units:,}")  # this pass
    print(f"  classified     : {agg['n_ok']:,}")  # oids with 45 rows
    print(f"  skipped        : {agg['n_skipped']:,}  "
          f"(no features: {agg['n_no_features']:,}, "
          f"no lastmjd: {agg['n_no_lastmjd']:,}, errors: {agg['n_errors']:,})")  # the three reasons
    if agg["n_errors"]:  # only errors are worth a retry
        print(f"  -> retry them  : cat {out_dir}/errors/*.jsonl | jq -r .oid > retry.txt")  # how
    if n_failed:  # units left unmarked
        print(f"  FAILED units   : {n_failed:,}  <- left unmarked; rerun to retry")  # how
    print(f"  probability rows: {agg['prob_rows']:,}")  # built
    if args.load_db:  # written
        print(f"  upserted to DB : {agg['db_prob_rows']:,} probability")  # must equal prob_rows
    print(f"  elapsed        : {elapsed / 3600:.2f} h "
          f"({agg['n_ok'] / elapsed if elapsed else 0:.1f} oid/s)")  # throughput
    print(f"  output         : {out_dir}")  # where the manifests are
    print("=" * 70)  # end of block
    return R.exit_code(n_failed, agg["n_errors"])  # non-zero only when units were lost


if __name__ == "__main__":  # script entry point
    sys.exit(main())  # the exit code
