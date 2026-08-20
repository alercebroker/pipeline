#!/usr/bin/env python
"""Benchmark the FULL offline path per oid: DB -> features -> BHRF probabilities.

`offline_benchmark_features.py` stops at `compute_features`. This one carries an
oid all the way to an OutputDTO and times each stage separately, so you can see
where the per-oid budget actually goes:

    db_fetch   detections + forced + ps1 + references, and build_message
    allwise    live Xwave crossmatch (--xmatch-url), or the DB xmatch read
    features   preprocess + ZTFFeatureExtractor  (compute_astro_object)
    parse      parse_output -> wide dict -> features-only InputDTO
    predict    model.can_predict + model.predict  (BHRF 2.1.0)

Cost scales with light-curve length, and `multisurvey_ztf` light curves are long
(n_det: median 123, p90 549, max ~3900). A flat average over an arbitrary sample
is therefore misleading, so oids are sampled in **n_det strata** and the report
is per stratum, then reweighted by how much of the catalog each stratum holds.

⚠ Model gotcha (see OFFLINE_VS_LEGACY_VALIDATION.md §3): pass a LOCAL --model-path.
A URL MODEL_PATH is downloaded into /tmp/SquidwardFeaturesClassifier/ and whatever
already sits there is reused — that cache holds a stale SNIbc pickle. This script
asserts SESN is in the loaded classes and SNIbc is not, and refuses to run otherwise.

Run from the pipeline root:

    MODEL_PATH=~/Desktop/alerce_models/squidward/2.1.0/hierarchical_random_forest_model.pkl \
        python feature_step/scripts/offline_benchmark_end_to_end.py --per-stratum 5
"""
import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

PIPE = Path(__file__).resolve().parents[2]  # .../pipeline
for p in (PIPE / "feature_step", PIPE / "lc_classifier", PIPE / "alerce_classifiers",
          PIPE / "libs" / "idmapper"):
    sys.path.insert(0, str(p))

import numpy as np
import pandas as pd
from sqlalchemy import text

from features.offline import db, xmatch
from features.offline.classify import (
    features_message_to_dto, load_squidward_model,
)
from features.offline.lc_features import compute_astro_object
from features.offline.message import build_message
from features.utils.parsers import parse_output

DEFAULT_CREDENTIALS = str(PIPE / "feature_step" / "features" / "offline" / "credentials.json")
# n_det strata: (label, lo, hi). Bounds come from pg_stats on multisurvey_ztf.object.
STRATA = [
    ("p0-p25   (short)", 0, 59),
    ("p25-p50", 59, 123),
    ("p50-p75", 123, 274),
    ("p75-p90", 274, 549),
    ("p90-p99", 549, 1127),
    ("p99+     (long)", 1127, 10 ** 6),
]
# Fraction of the catalog in each stratum, by construction of the percentiles.
STRATUM_WEIGHT = [0.25, 0.25, 0.25, 0.15, 0.09, 0.01]
STAGES = ["db_fetch", "allwise", "features", "parse", "predict"]


def sample_oids(credentials: str, per_stratum: int, seed: int) -> list:
    """Pick `per_stratum` oids inside each n_det band, using ix_object_n_det."""
    engine = db._make_engine(credentials)
    out = []
    with engine.connect() as conn:
        for label, lo, hi in STRATA:
            rows = pd.read_sql_query(
                text(f"""
                    SELECT oid, n_det, n_forced FROM {db.SCHEMA}.object
                    WHERE sid = :sid AND n_det > :lo AND n_det <= :hi
                    ORDER BY oid OFFSET :off LIMIT :n
                """),
                conn,
                params={"sid": db.SID, "lo": lo, "hi": hi,
                        "off": seed % 1000, "n": per_stratum},
            )
            for r in rows.itertuples(index=False):
                out.append((label, int(r.oid), int(r.n_det), int(r.n_forced)))
    return out


def time_one(oid: int, credentials: str, model, preproc, extractor,
             min_det: int, xmatch_url) -> dict:
    """Run one oid through every stage, returning per-stage seconds."""
    t = {}
    oids = [oid]

    t0 = time.perf_counter()
    dets = db.fetch_detections(credentials, oids)
    forced = db.fetch_forced_photometry(credentials, oids)
    ps1 = db.fetch_ps1(credentials, oids)
    refs = db.fetch_references(credentials, oids)
    message = build_message(oid, dets, forced, ps1)
    t["db_fetch"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    if xmatch_url:
        matches = xmatch.compute_matches([oid], [message["meanra"]],
                                         [message["meandec"]], base_url=xmatch_url)
        allwise = xmatch.matches_to_allwise_df(matches)
    else:
        allwise = db.fetch_allwise(credentials, oids)
    t["allwise"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    ao = compute_astro_object(message, refs, allwise, min_det,
                              preprocessor=preproc, extractor=extractor)
    t["features"] = time.perf_counter() - t0
    if ao is None:
        return None

    t0 = time.perf_counter()
    candids = {message["oid"]: message.get("measurement_id", [])}
    out_message = parse_output([ao], [message], candids)[0]
    dto = features_message_to_dto(out_message)
    t["parse"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    can, _ = model.can_predict(dto)
    dto_out = model.predict(dto) if can else None
    t["predict"] = time.perf_counter() - t0

    t["epochs"] = len(dets) + len(forced)
    t["n_features"] = len(out_message.get("features") or {})
    t["predicted"] = dto_out is not None
    return t


def pct(xs, q):
    return float(np.percentile(xs, q)) if xs else float("nan")


def fmt_ms(x):
    return f"{x * 1000:,.0f}ms" if x < 1 else f"{x:,.2f}s"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--credentials", default=DEFAULT_CREDENTIALS)
    ap.add_argument("--per-stratum", type=int, default=5, help="oids timed per n_det band")
    ap.add_argument("--warmup", type=int, default=2, help="oids run first and discarded")
    ap.add_argument("--min-det", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0, help="shifts the OFFSET of each sample")
    ap.add_argument("--model-path", default=os.getenv("MODEL_PATH"),
                    help="LOCAL BHRF pickle (a URL hits the stale /tmp cache)")
    ap.add_argument("--xmatch-url", default=os.getenv("XMATCH_URL"),
                    help="Xwave base url for a live AllWISE crossmatch; "
                         "omit to read the (empty) DB xmatch tables")
    ap.add_argument("--json-out", default=None, help="write raw per-oid timings here")
    args = ap.parse_args()

    if not args.model_path:
        sys.exit("error: --model-path (or MODEL_PATH) is required")
    model_path = os.path.expanduser(args.model_path)
    if model_path.startswith(("http://", "https://")):
        sys.exit("error: pass a LOCAL model path — a URL reuses the stale "
                 "SNIbc pickle in /tmp/SquidwardFeaturesClassifier/")
    if not Path(model_path).exists():
        sys.exit(f"error: model not found at {model_path}")
    os.environ["MODEL_PATH"] = model_path

    print("building preprocessor + extractor ...", flush=True)
    from lc_classifier.features.composites.ztf import ZTFFeatureExtractor
    from lc_classifier.features.preprocess.ztf import ZTFLightcurvePreprocessor
    t0 = time.perf_counter()
    preproc = ZTFLightcurvePreprocessor(drop_bogus=True)
    extractor = ZTFFeatureExtractor()
    init_extract_s = time.perf_counter() - t0

    print(f"loading BHRF model from {model_path} ...", flush=True)
    t0 = time.perf_counter()
    model, clf_name, clf_version = load_squidward_model()
    init_model_s = time.perf_counter() - t0

    # list_of_classes lives on the wrapped HierarchicalRandomForestClassifier,
    # not on the SquidwardFeaturesClassifier facade.
    inner = getattr(model, "model", None)
    raw_classes = getattr(inner, "list_of_classes", None)
    if raw_classes is None:  # ndarray — don't use `or`, its truth value is ambiguous
        raw_classes = getattr(model, "list_of_classes", [])
    classes = [str(c) for c in raw_classes]
    if "SESN" not in classes or "SNIbc" in classes:
        sys.exit(f"error: wrong model loaded — expected SESN and no SNIbc, got {classes}")
    print(f"  {clf_name} {clf_version}, {len(classes)} classes, SESN check OK")
    print(f"  one-time: extractor {init_extract_s:.1f}s, model load {init_model_s:.1f}s\n")

    sample = sample_oids(args.credentials, args.per_stratum + args.warmup, args.seed)
    print(f"sampled {len(sample)} oids across {len(STRATA)} n_det strata "
          f"({args.warmup} warm-up per stratum, discarded)\n")

    # Warm-up: numba/jax compile on first call, don't let it pollute the numbers.
    warm = [s for s in sample if s[0] == STRATA[0][0]][: args.warmup]
    for _, oid, _, _ in warm:
        try:
            time_one(oid, args.credentials, model, preproc, extractor,
                     args.min_det, args.xmatch_url)
        except Exception as exc:
            print(f"  warm-up oid {oid} failed: {type(exc).__name__}: {exc}")

    by_stratum, records, skipped = {}, [], 0
    seen_per_stratum = {}
    print(f"{'stratum':18s} {'oid':>19s} {'n_det':>6s} {'epochs':>7s} "
          + ' '.join(f'{s:>9s}' for s in STAGES) + f" {'total':>9s}")
    for label, oid, n_det, n_forced in sample:
        k = seen_per_stratum.get(label, 0)
        if k >= args.per_stratum:
            continue
        try:
            r = time_one(oid, args.credentials, model, preproc, extractor,
                         args.min_det, args.xmatch_url)
        except Exception as exc:
            print(f"{label:18s} {oid:>19} FAILED {type(exc).__name__}: {exc}")
            skipped += 1
            continue
        if r is None:
            skipped += 1
            continue
        seen_per_stratum[label] = k + 1
        total = sum(r[s] for s in STAGES)
        r.update(stratum=label, oid=oid, n_det=n_det, n_forced=n_forced, total=total)
        records.append(r)
        by_stratum.setdefault(label, []).append(r)
        print(f"{label:18s} {oid:>19} {n_det:>6} {r['epochs']:>7} "
              + ' '.join(f"{r[s]*1000:>8.0f}m" for s in STAGES)
              + f" {total:>8.2f}s", flush=True)

    if not records:
        sys.exit("\nno oid completed — nothing to report")

    print(f"\n{'='*104}\nPER-STAGE, BY STRATUM (median seconds)\n{'='*104}")
    print(f"{'stratum':18s} {'n':>3s} {'med n_det':>9s} "
          + ' '.join(f'{s:>10s}' for s in STAGES) + f" {'TOTAL':>10s}")
    for label, _, _ in STRATA:
        rs = by_stratum.get(label)
        if not rs:
            continue
        meds = [statistics.median([r[s] for r in rs]) for s in STAGES]
        print(f"{label:18s} {len(rs):>3d} "
              f"{statistics.median([r['n_det'] for r in rs]):>9.0f} "
              + ' '.join(f'{m:>10.3f}' for m in meds)
              + f" {statistics.median([r['total'] for r in rs]):>10.3f}")

    totals = [r["total"] for r in records]
    print(f"\n{'='*104}\nOVERALL ({len(records)} oids timed, {skipped} skipped)\n{'='*104}")
    for s in STAGES:
        xs = [r[s] for r in records]
        share = sum(xs) / sum(totals) * 100
        print(f"  {s:10s} mean={fmt_ms(statistics.mean(xs)):>9s} "
              f"median={fmt_ms(statistics.median(xs)):>9s} "
              f"p90={fmt_ms(pct(xs, 90)):>9s} max={fmt_ms(max(xs)):>9s}  {share:5.1f}% del total")
    print(f"  {'TOTAL':10s} mean={fmt_ms(statistics.mean(totals)):>9s} "
          f"median={fmt_ms(statistics.median(totals)):>9s} "
          f"p90={fmt_ms(pct(totals, 90)):>9s} max={fmt_ms(max(totals)):>9s}")

    # Catalog-level projection: reweight the strata medians by catalog share,
    # instead of averaging a sample that over-represents the long tail.
    weighted = 0.0
    covered = 0.0
    for (label, _, _), w in zip(STRATA, STRATUM_WEIGHT):
        rs = by_stratum.get(label)
        if not rs:
            continue
        weighted += w * statistics.median([r["total"] for r in rs])
        covered += w
    if covered:
        weighted /= covered  # renormalize over the strata we actually measured
    n_catalog = 130_451_381
    print(f"\n{'='*104}\nPROJECTION OVER THE FULL CATALOG\n{'='*104}")
    print(f"  weighted mean per oid (strata reweighted by catalog share): {weighted:.3f}s")
    print(f"  catalog: {n_catalog:,} oids -> {weighted * n_catalog / 3600:,.0f} core-hours "
          f"({weighted * n_catalog / 86400:,.0f} core-days)")
    for w in (8, 32, 128, 512):
        print(f"    with {w:>4d} workers: {weighted * n_catalog / 3600 / w:>10,.1f} h "
              f"({weighted * n_catalog / 86400 / w:>7,.1f} d)")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump({"records": records, "weighted_s_per_oid": weighted,
                       "init": {"extractor_s": init_extract_s, "model_s": init_model_s}},
                      f, indent=2, default=str)
        print(f"\n  raw timings -> {args.json_out}")


if __name__ == "__main__":
    main()
