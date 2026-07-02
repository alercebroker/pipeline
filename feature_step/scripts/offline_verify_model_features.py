#!/usr/bin/env python
"""Verify the offline pipeline emits all 199 features the deployed BHRF model needs.

Primary check (default, no model download): for each sample oid, compute features
offline and diff the resulting predict-input column set against the pinned
MODEL_FEATURE_LIST. A missing name would KeyError inside model.predict.

Confirmation (--smoke, requires MODEL_PATH + training_py310): load the real model
once, run predict on the sample, assert no KeyError and model.feature_list matches
the pinned constant.

    conda run --no-capture-output -n training_py310 python \
        feature_step/scripts/offline_verify_model_features.py

    MODEL_PATH=https://alerce-models.s3.amazonaws.com/squidward/2.1.0/hierarchical_random_forest_model.pkl \
        conda run --no-capture-output -n training_py310 python \
        feature_step/scripts/offline_verify_model_features.py --smoke
"""
import sys
from pathlib import Path

PIPE = Path(__file__).resolve().parents[2]   # .../pipeline
for p in (PIPE / "feature_step", PIPE / "lc_classifier",
          PIPE / "libs" / "idmapper", PIPE / "libs" / "apf",
          PIPE / "alerce_classifiers"):
    sys.path.insert(0, str(p))

import argparse

from features.offline import db
from features.offline.message import build_message
from features.offline.lc_features import compute_astro_object
from features.offline.model_features import predict_input_columns, diff_feature_coverage
from features.offline.model_feature_list import MODEL_FEATURE_LIST, MODEL_VERSION

DEFAULT_CREDENTIALS = str(PIPE / "feature_step" / "features" / "offline" / "credentials.json")

# Diverse handful discovered via scripts discovery (Task 3), pinned for reproducibility.
SAMPLE_OIDS = [
    36028941595456515,  # forced n_det=2043 n_fp=341
    36028933560567118,  # dense n_det=1543 n_fp=208
    36028941602080885,  # dense n_det=1466 n_fp=190
    36028981753134781,  # sparse n_det=5 n_fp=12
    36028957686637252,  # sparse n_det=5 n_fp=0
    36028941618070191,  # sparse n_det=5 n_fp=0
    36028933561429258,  # forced n_det=1205 n_fp=342
    36028941604445605,  # forced n_det=929 n_fp=272
    36028941624528297,  # lut oid (feature_lut.py provenance)
]


def _astro_object_for(oid: int, credentials: str, min_det: int):
    dets = db.fetch_detections(credentials, [oid])
    forced = db.fetch_forced_photometry(credentials, [oid])
    ps1 = db.fetch_ps1(credentials, [oid])
    allwise = db.fetch_allwise(credentials, [oid])
    refs = db.fetch_references(credentials, [oid])
    message = build_message(oid, dets, forced, ps1)
    ao = compute_astro_object(message, refs, allwise, min_det)
    return ao, message


def run_name_diff(oids, credentials, min_det) -> int:
    """Name-diff each oid; return process exit code (0 = all covered)."""
    agg_missing = set(MODEL_FEATURE_LIST)   # intersect down to names missing for ALL
    any_missing = set()                     # union of names missing for ANY oid
    checked = 0
    print(f"expected: {len(MODEL_FEATURE_LIST)} features (BHRF {MODEL_VERSION})\n")
    for oid in oids:
        ao, message = _astro_object_for(oid, credentials, min_det)
        if ao is None:
            print(f"  oid {oid}: SKIP (too few real detections)")
            continue
        cols = predict_input_columns(ao, message)
        diff = diff_feature_coverage(cols, MODEL_FEATURE_LIST)
        checked += 1
        agg_missing &= set(diff["missing"])
        any_missing |= set(diff["missing"])
        status = "OK" if diff["n_missing"] == 0 else f"MISSING {diff['n_missing']}"
        print(f"  oid {oid}: {status}"
              + (f" -> {diff['missing']}" if diff["missing"] else "")
              + (f"  (+{len(diff['extra'])} extra)" if diff["extra"] else ""))

    print(f"\nchecked {checked}/{len(oids)} oids")
    if checked == 0:
        print("FAIL: no oids produced features")
        return 1
    if any_missing:
        print(f"FAIL: {len(any_missing)} name(s) missing for at least one oid: "
              f"{sorted(any_missing)}")
        if agg_missing:
            print(f"  of which missing for ALL oids: {sorted(agg_missing)}")
        return 1
    print("PASS: all 199 features covered for every checked oid")
    return 0


def run_smoke(oids, credentials, min_det) -> int:
    raise SystemExit("run_smoke not implemented yet (Task 5)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--oid", type=int, action="append",
                    help="Override SAMPLE_OIDS (repeatable).")
    ap.add_argument("--credentials", default=DEFAULT_CREDENTIALS)
    ap.add_argument("--min-det", type=int, default=1)
    ap.add_argument("--smoke", action="store_true",
                    help="Also load the real model (MODEL_PATH) and run predict.")
    args = ap.parse_args()

    oids = args.oid if args.oid else SAMPLE_OIDS
    code = run_name_diff(oids, args.credentials, args.min_det)

    if args.smoke:
        code |= run_smoke(oids, args.credentials, args.min_det)   # replaced in Task 5

    sys.exit(code)


if __name__ == "__main__":
    main()
