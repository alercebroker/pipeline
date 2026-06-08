#!/usr/bin/env python
"""Benchmark per-oid ZTF feature computation.

Times `lc_features.compute_features` (preprocess + extract) over a sample of real
multisurvey oids, with the preprocessor/extractor built ONCE (as in production)
and a warm-up to exclude JIT/numba/jax first-call compilation. DB-fetch and
message-build are timed separately so the "feature computation" number is clean.

NOTE: the multisurvey DB is currently a recent ~3-month slice, so light curves
are short (low n_det). Extractor cost grows with light-curve length, so these
numbers are a LOWER bound vs full-history light curves.

Run from the pipeline root or feature_step/:
    conda run --no-capture-output -n training_py310 python \
        feature_step/scripts/offline_benchmark_features.py --n 30

Default --credentials points at the training-repo credentials file; override as needed.
"""
import argparse
import statistics
import sys
import time
from pathlib import Path

PIPE = Path(__file__).resolve().parents[2]   # .../pipeline
for p in (PIPE / "feature_step", PIPE / "lc_classifier", PIPE / "libs" / "idmapper"):
    sys.path.insert(0, str(p))

import pandas as pd
from sqlalchemy import text

from features.offline import db, lc_features
from features.offline.message import build_message

DEFAULT_CREDENTIALS = "/home/fandrades/desktop/repos/training/features_ztf/data/credentials.json"


def _select_oids(credentials: str, n: int, min_det: int) -> list:
    engine = db._make_engine(credentials)
    with engine.connect() as conn:
        rows = pd.read_sql_query(
            text(f"""
                SELECT oid, n_det FROM {db.SCHEMA}.object
                WHERE sid = :sid AND n_det >= :min_det
                ORDER BY oid LIMIT :n
            """),
            conn, params={"sid": db.SID, "min_det": min_det, "n": n},
        )
    return list(zip(rows["oid"].tolist(), rows["n_det"].tolist()))


def _fetch_and_build(credentials: str, oid: int):
    oids = [oid]
    dets = db.fetch_detections(credentials, oids)
    forced = db.fetch_forced_photometry(credentials, oids)
    ps1 = db.fetch_ps1(credentials, oids)
    allwise = db.fetch_allwise(credentials, oids)
    refs = db.fetch_references(credentials, oids)
    msg = build_message(oid, dets, forced, ps1)
    n_epochs = len(dets) + len(forced)
    return msg, refs, allwise, n_epochs


def main():
    ap = argparse.ArgumentParser(
        description="Benchmark offline ZTF feature computation over a sample of oids."
    )
    ap.add_argument("--n", type=int, default=30, help="oids to time")
    ap.add_argument("--warmup", type=int, default=2, help="warm-up oids (excluded)")
    ap.add_argument("--min-det", type=int, default=20)
    ap.add_argument("--credentials", default=DEFAULT_CREDENTIALS,
                    help="Path to DB credentials JSON (override for non-default envs).")
    args = ap.parse_args()

    cand = _select_oids(args.credentials, args.n + args.warmup, args.min_det)
    print(f"selected {len(cand)} oids (n_det >= {args.min_det}); "
          f"{args.warmup} warm-up + up to {args.n} timed")

    from lc_classifier.features.composites.ztf import ZTFFeatureExtractor
    from lc_classifier.features.preprocess.ztf import ZTFLightcurvePreprocessor

    t0 = time.perf_counter()
    preproc = ZTFLightcurvePreprocessor(drop_bogus=True)
    extractor = ZTFFeatureExtractor()
    init_s = time.perf_counter() - t0
    print(f"extractor+preprocessor construction: {init_s:.2f}s (one-time)")

    def run(oid):
        msg, refs, allwise, n_epochs = _fetch_and_build(args.credentials, oid)
        t = time.perf_counter()
        feats = lc_features.compute_features(msg, refs, allwise,
                                             preprocessor=preproc, extractor=extractor)
        return time.perf_counter() - t, n_epochs, feats

    # warm-up (JIT/numba/jax compile) — not counted
    cold_s = None
    for i, (oid, _) in enumerate(cand[:args.warmup]):
        c, _, _ = run(oid)
        if i == 0:
            cold_s = c
    if cold_s is not None:
        print(f"cold-start (first call, incl. compilation): {cold_s:.2f}s\n")

    rows, computes, fetches = [], [], []
    for oid, n_det in cand[args.warmup:]:
        tf = time.perf_counter()
        msg, refs, allwise, n_epochs = _fetch_and_build(args.credentials, oid)
        fetch_s = time.perf_counter() - tf
        tc = time.perf_counter()
        feats = lc_features.compute_features(msg, refs, allwise,
                                             preprocessor=preproc, extractor=extractor)
        compute_s = time.perf_counter() - tc
        if feats is None:
            continue
        computes.append(compute_s); fetches.append(fetch_s)
        rows.append((oid, n_epochs, fetch_s, compute_s))

    print(f"{'oid':>20} {'epochs':>7} {'fetch_s':>9} {'compute_s':>10}")
    for oid, ne, fs, cs in rows:
        print(f"{oid:>20} {ne:>7} {fs:>9.3f} {cs:>10.3f}")

    def stats(xs):
        return (statistics.mean(xs), statistics.median(xs), min(xs), max(xs),
                statistics.pstdev(xs) if len(xs) > 1 else 0.0)

    print(f"\ntimed oids: {len(computes)}")
    if computes:
        m, md, lo, hi, sd = stats(computes)
        print(f"compute_features  mean={m*1000:.1f}ms  median={md*1000:.1f}ms  "
              f"min={lo*1000:.1f}ms  max={hi*1000:.1f}ms  std={sd*1000:.1f}ms")
        m, md, lo, hi, sd = stats(fetches)
        print(f"db fetch+build    mean={m*1000:.1f}ms  median={md*1000:.1f}ms  "
              f"min={lo*1000:.1f}ms  max={hi*1000:.1f}ms")
        epc = [r[1] for r in rows]
        print(f"epochs/oid        mean={statistics.mean(epc):.1f}  "
              f"min={min(epc)}  max={max(epc)}")
        print("\nNOTE: current multisurvey LCs are a ~3-month slice (short). "
              "compute_features scales with light-curve length; full-history LCs "
              "will be slower.")


if __name__ == "__main__":
    main()
