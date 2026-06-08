#!/usr/bin/env python
"""Compare our pipeline features against legacy ALeRCE features in alerce.feature.

Run from the pipeline root or feature_step/:
    conda run --no-capture-output -n training_py310 python \
        feature_step/scripts/offline_compare_vs_alerce.py --oid 36028933559755080

Accepts either a multisurvey bigint oid (e.g. 36028933559755080) or a ZTF string
oid (e.g. ZTF17aaabauy). Resolves both forms automatically via idmapper.

NOTE: until the multisurvey DB is backfilled to full light curves, large differences
(i.e. 'differ' status) are EXPECTED — the input light curves differ (multisurvey
currently holds only a recent ~3-month slice, while alerce holds the full history).
The tooling is correct; exit code 1 today simply reflects the truncated LC, not a
bug. Re-run once multisurvey is backfilled to confirm full equality.

Default --credentials points at the training-repo credentials file; override as needed.
"""
import argparse
import sys
from pathlib import Path

PIPE = Path(__file__).resolve().parents[2]   # .../pipeline
for p in (PIPE / "feature_step", PIPE / "lc_classifier", PIPE / "libs" / "idmapper"):
    sys.path.insert(0, str(p))

import pandas as pd
from sqlalchemy import text

from features.offline import db, feature_compare, lc_features
from features.offline.feature_compare import compare_feature_frames
from features.offline.message import build_message
from idmapper.mapper import decode_masterid, catalog_oid_to_masterid

DEFAULT_CREDENTIALS = "/home/fandrades/desktop/repos/training/features_ztf/data/credentials.json"

# ZTF oid strings start with "ZTF" followed by digits/letters.
_ZTF_PREFIX = "ZTF"


def _is_ztf_string(raw: str) -> bool:
    return raw.startswith(_ZTF_PREFIX) and len(raw) > 3


def _resolve_oid(raw: str) -> tuple[int, str]:
    """Return (bigint_oid, ztf_string_oid) from either form of oid string."""
    raw = raw.strip()
    if _is_ztf_string(raw):
        bigint = catalog_oid_to_masterid("ZTF", raw, True)
        return bigint, raw
    # Try to parse as a bigint
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


def _fetch_alerce_lc_span(credentials_json: str, ztf_oid: str) -> dict:
    """Fetch firstmjd, lastmjd, n_det from alerce.object for context display."""
    engine = db._make_engine(credentials_json)
    query = text("""
        SELECT firstmjd, lastmjd, ndet
        FROM alerce.object
        WHERE oid = :oid
    """)
    try:
        with engine.connect() as conn:
            df = pd.read_sql_query(query, conn, params={"oid": ztf_oid})
        if df.empty:
            return {}
        row = df.iloc[0]
        return {
            "firstmjd": row.get("firstmjd"),
            "lastmjd": row.get("lastmjd"),
            "ndet": row.get("ndet"),
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def main():
    parser = argparse.ArgumentParser(
        description="Compare pipeline features vs ALeRCE stored features."
    )
    parser.add_argument(
        "--oid",
        required=True,
        help="Multisurvey bigint oid (e.g. 36028933559755080) or ZTF string (e.g. ZTF17aaabauy).",
    )
    parser.add_argument(
        "--version",
        default=None,
        help="alerce.feature version string to compare against. Default: latest found.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-3,
        help="Relative tolerance for feature matching (default: 1e-3).",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-6,
        help="Absolute tolerance for feature matching (default: 1e-6).",
    )
    parser.add_argument(
        "--credentials",
        default=DEFAULT_CREDENTIALS,
        help="Path to DB credentials JSON (override for non-default envs).",
    )
    args = parser.parse_args()

    bigint_oid, ztf_oid = _resolve_oid(args.oid)
    credentials = args.credentials

    print("=" * 70)
    print(f"  bigint oid : {bigint_oid}")
    print(f"  ZTF oid    : {ztf_oid}")
    print("=" * 70)

    # --- Fetch our pipeline data ---
    oids = [bigint_oid]
    print("\nFetching multisurvey data...")
    try:
        dets = db.fetch_detections(credentials, oids)
        forced = db.fetch_forced_photometry(credentials, oids)
        ps1 = db.fetch_ps1(credentials, oids)
        allwise = db.fetch_allwise(credentials, oids)
        refs = db.fetch_references(credentials, oids)
    except Exception as exc:
        print(f"ERROR: failed to fetch multisurvey data: {exc}")
        sys.exit(1)

    if dets.empty:
        print(f"\nERROR: no detections found in multisurvey for bigint oid {bigint_oid}.")
        print("Check that the oid exists in the DB or that the DB is accessible.")
        sys.exit(1)

    our_mjd_min = dets["mjd"].min()
    our_mjd_max = dets["mjd"].max()
    our_n_det = len(dets)

    print(f"\n  Our LC (multisurvey):  n_det={our_n_det},  mjd={our_mjd_min:.2f}..{our_mjd_max:.2f}")

    # Alerce LC span from alerce.object
    alerce_span = _fetch_alerce_lc_span(credentials, ztf_oid)
    if alerce_span and "error" not in alerce_span:
        print(
            f"  Alerce LC (object):    ndet={alerce_span.get('ndet')}, "
            f"mjd={alerce_span.get('firstmjd')}..{alerce_span.get('lastmjd')}"
        )
    elif "error" in alerce_span:
        print(f"  Alerce LC span: unavailable ({alerce_span['error']})")
    else:
        print("  Alerce LC span: oid not found in alerce.object")

    # --- Build message and compute our features ---
    print("\nComputing our features...")
    try:
        message = build_message(bigint_oid, dets, forced, ps1)
        our_features = lc_features.compute_features(message, refs, allwise)
    except Exception as exc:
        print(f"ERROR: failed to compute features: {exc}")
        sys.exit(1)

    if our_features is None or our_features.empty:
        print("ERROR: compute_features returned an empty/None frame.")
        sys.exit(1)

    print(f"  Our features: {len(our_features)} rows, {our_features['value'].notna().sum()} non-NaN values")

    # --- Resolve alerce version ---
    print(f"\nFetching alerce.feature for oid {ztf_oid!r}...")
    versions = db.list_alerce_feature_versions(credentials, ztf_oid)
    if not versions:
        print(f"\nERROR: no features found in alerce.feature for oid {ztf_oid!r}.")
        print("The oid may not be in the ALeRCE DB or may have a different string form.")
        sys.exit(1)

    chosen_version = args.version
    if chosen_version is None:
        chosen_version = feature_compare.latest_feature_version(versions)
        print(f"  Available versions: {versions}")
        print(f"  Defaulting to: {chosen_version!r}")
        print(
            "  NOTE: Default picks the highest modern version; for a true equality "
            "check pass --version matching our lc_classifier version."
        )
    else:
        if chosen_version not in versions:
            print(
                f"  WARNING: requested version {chosen_version!r} not found. "
                f"Available: {versions}"
            )

    alerce_features = db.fetch_alerce_features(credentials, ztf_oid, version=chosen_version)
    print(f"  Alerce features: {len(alerce_features)} rows (version={chosen_version!r})")

    if alerce_features.empty:
        print("\nERROR: alerce.feature returned 0 rows for this oid+version.")
        sys.exit(1)

    # --- Compare ---
    print(f"\nComparing features (rtol={args.rtol}, atol={args.atol})...")
    merged, summary = compare_feature_frames(
        our_features, alerce_features, rtol=args.rtol, atol=args.atol
    )

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  n_compared  : {summary['n_compared']}")
    print(f"  match       : {summary['match']}")
    print(f"  differ      : {summary['differ']}")
    print(f"  only_ours   : {summary['only_ours']}")
    print(f"  only_theirs : {summary['only_theirs']}")

    if summary["only_ours_names"]:
        print(f"\n  Features only in our output ({len(summary['only_ours_names'])} unique names shown):")
        for name in summary["only_ours_names"]:
            print(f"    {name}")

    if summary["only_theirs_names"]:
        print(f"\n  Features only in alerce ({len(summary['only_theirs_names'])} unique names shown):")
        for name in summary["only_theirs_names"]:
            print(f"    {name}")

    # Top differ rows by rel_diff
    differ_rows = merged[merged["status"] == "differ"].copy()
    if not differ_rows.empty:
        differ_rows = differ_rows.sort_values("rel_diff", ascending=False)
        print(f"\n  Top {min(20, len(differ_rows))} differing features (by rel_diff):")
        print(
            differ_rows[["name", "fid_int", "value_ours", "value_theirs", "abs_diff", "rel_diff"]]
            .head(20)
            .to_string(index=False)
        )

    print("\n" + "=" * 70)
    if summary["differ"] > 0 or summary["n_compared"] == 0:
        print(
            "  NOTE: 'differ' or n_compared=0 is EXPECTED until multisurvey is backfilled\n"
            "  to full light curves. Our LC is a recent ~3-month slice; alerce stores the\n"
            "  full multi-year history. Re-run once backfill is complete."
        )
    print("=" * 70)

    if summary["n_compared"] > 0 and summary["differ"] == 0:
        print("\nRESULT: PASS — all compared features match within tolerance.")
        sys.exit(0)
    else:
        print("\nRESULT: FAIL (exit 1) — differ>0 or n_compared=0 (see NOTE above).")
        sys.exit(1)


if __name__ == "__main__":
    main()
