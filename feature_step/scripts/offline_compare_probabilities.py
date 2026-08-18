#!/usr/bin/env python
"""Compare offline BHRF probabilities vs. stored legacy probabilities (alerce.probability).

Mirrors offline_compare_vs_alerce.py, for probabilities: run the offline pipeline
on one oid to get BHRF probabilities across all 5 heads, read the stored
probabilities for the same oid + version from alerce.probability, and diff.

Only sound when multisurvey_ztf holds the object's COMPLETE light curve (i.e. the
object finished before the ~Aug-2025 backfill edge). For still-active objects the
backfill is missing recent detections and a mismatch is expected, not a bug.

Requires MODEL_PATH (the BHRF pickle), same as offline_classify.py:

    MODEL_PATH=https://alerce-models.s3.amazonaws.com/squidward/2.1.0/hierarchical_random_forest_model.pkl \
        conda run --no-capture-output -n training_py310 python \
        feature_step/scripts/offline_compare_probabilities.py --oid ZTF18abkhtlr
"""
import argparse
import os
import sys
from pathlib import Path

PIPE = Path(__file__).resolve().parents[2]   # .../pipeline
for p in (PIPE / "feature_step", PIPE / "lc_classifier",
          PIPE / "libs" / "idmapper", PIPE / "libs" / "apf",
          PIPE / "libs" / "xmatch_client", PIPE / "alerce_classifiers"):
    sys.path.insert(0, str(p))

from idmapper.mapper import catalog_oid_to_masterid, decode_masterid

from features.offline import db, xmatch
from features.offline.classify import classify_oid, load_squidward_model
from features.offline.probability_compare import (
    BHRF_CLASSIFIER_NAMES,
    compare_probability_frames,
    offline_dto_to_series,
)

DEFAULT_CREDENTIALS = str(PIPE / "feature_step" / "features" / "offline" / "credentials.json")
_ZTF_PREFIX = "ZTF"


def _resolve_oid(raw: str) -> tuple[int, str]:
    """Return (bigint_oid, ztf_string_oid) from either form of oid string."""
    raw = raw.strip()
    if raw.startswith(_ZTF_PREFIX) and len(raw) > 3:
        return catalog_oid_to_masterid("ZTF", raw, True), raw
    try:
        bigint = int(raw)
    except ValueError:
        print(f"ERROR: cannot parse oid {raw!r} as a ZTF string oid or a bigint integer.")
        sys.exit(1)
    survey, ztf_str = decode_masterid(bigint)
    if survey != "ZTF":
        print(f"ERROR: bigint {bigint} decoded to survey {survey!r}, not ZTF.")
        sys.exit(1)
    return bigint, ztf_str


def main():
    ap = argparse.ArgumentParser(
        description="Compare offline BHRF probabilities vs stored alerce.probability."
    )
    ap.add_argument("--oid", required=True,
                    help="ZTF string oid (e.g. ZTF18abkhtlr) or multisurvey bigint.")
    ap.add_argument("--version", default="2.1.0",
                    help="stored classifier_version to compare against (default 2.1.0).")
    ap.add_argument("--rtol", type=float, default=1e-3, help="relative tolerance.")
    ap.add_argument("--atol", type=float, default=1e-4, help="absolute tolerance.")
    ap.add_argument("--credentials", default=DEFAULT_CREDENTIALS,
                    help="Path to DB credentials JSON.")
    ap.add_argument("--min-det", type=int, default=1,
                    help="Minimum real detections required to classify.")
    ap.add_argument("--xmatch-url", default=os.getenv("XMATCH_URL", xmatch.DEFAULT_XMATCH_URL),
                    dest="xmatch_url",
                    help="Xwave crossmatch service URL (default: %(default)s). Computes the "
                         "AllWISE crossmatch live like the step; pass '' for the DB-read fallback.")
    args = ap.parse_args()

    bigint_oid, ztf_oid = _resolve_oid(args.oid)
    print("=" * 70)
    print(f"  bigint oid : {bigint_oid}")
    print(f"  ZTF oid    : {ztf_oid}")
    print(f"  version    : {args.version}")
    print("=" * 70)

    model, name, model_version = load_squidward_model()
    print(f"\nmodel: {name} version={model_version}")
    if model_version != args.version:
        print(f"  WARNING: model version {model_version!r} != compare version "
              f"{args.version!r}; probabilities are not expected to match.")

    print(f"\nComputing offline probabilities... "
          f"(xmatch: {'live Xwave @ ' + args.xmatch_url if args.xmatch_url else 'DB read'})")
    result = classify_oid(bigint_oid, args.credentials, model, min_detections=args.min_det,
                          xmatch_url=args.xmatch_url or None)
    if result is None or result.probabilities is None or len(result.probabilities) == 0:
        print("ERROR: offline classify produced no probabilities "
              "(too few detections or can't predict).")
        sys.exit(1)
    offline_by_name = offline_dto_to_series(result)
    print(f"  offline heads: {sorted(offline_by_name)}")

    print(f"\nFetching stored probabilities from alerce.probability...")
    stored = db.fetch_stored_probabilities(
        args.credentials, ztf_oid, BHRF_CLASSIFIER_NAMES, args.version)
    print(f"  stored rows: {len(stored)} across {stored['classifier_name'].nunique()} heads")
    if stored.empty:
        print(f"\nERROR: no stored probabilities for oid {ztf_oid!r} version {args.version!r}.")
        sys.exit(1)

    merged, summary = compare_probability_frames(
        offline_by_name, stored, rtol=args.rtol, atol=args.atol)

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  n_compared   : {summary['n_compared']}")
    print(f"  match        : {summary['match']}")
    print(f"  differ       : {summary['differ']}")
    print(f"  only_offline : {summary['only_offline']}")
    print(f"  only_stored  : {summary['only_stored']}")
    print(f"  rank-1 agree : {summary['rank1_agree']}/{summary['rank1_total']}")
    print("\n  rank-1 per head (offline vs stored):")
    for cname, r in summary["rank1"].items():
        flag = "OK " if r["agree"] else "XX "
        short = cname.replace("lc_classifier_BHRF_forced_phot_", "").replace(
            "lc_classifier_BHRF_forced_phot", "flat")
        print(f"    {flag}{short:11s} offline={r['offline']!s:10} stored={r['stored']!s:10}")

    differ_rows = merged[merged["status"] == "differ"].copy()
    if not differ_rows.empty:
        differ_rows = differ_rows.sort_values("abs_diff", ascending=False)
        print(f"\n  Top {min(20, len(differ_rows))} differing classes (by abs_diff):")
        print(differ_rows[["classifier_name", "class_name", "prob_offline",
                           "prob_stored", "abs_diff", "rel_diff"]]
              .head(20).to_string(index=False))

    print("\n" + "=" * 70)
    if summary["passed"]:
        print("RESULT: PASS — offline probabilities reproduce the stored BHRF prediction.")
        sys.exit(0)
    print("RESULT: FAIL — see differ/only_* rows above. If the object is still active\n"
          "(multisurvey missing its recent tail), this is expected, not a bug.")
    sys.exit(1)


if __name__ == "__main__":
    main()
