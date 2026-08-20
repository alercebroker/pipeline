#!/usr/bin/env python
"""Live-DB check: DB -> correction-ztf message -> features for one real oid, end-to-end.

Run from the pipeline root or feature_step/:
    conda run --no-capture-output -n training_py310 python \
        feature_step/scripts/offline_compute_features.py --oid 36028941624528297

Default --credentials points at the training-repo credentials file; override as needed.
"""
import sys
from pathlib import Path

PIPE = Path(__file__).resolve().parents[2]   # .../pipeline
for p in (PIPE / "feature_step", PIPE / "lc_classifier", PIPE / "libs" / "idmapper"):
    sys.path.insert(0, str(p))

import argparse
import os

from importlib.metadata import PackageNotFoundError, version as _pkg_version

from features.offline import db, lc_features, feature_writer, xmatch
from features.offline.feature_lut import default_version_name
from features.offline.message import build_message

DEFAULT_CREDENTIALS = str(PIPE / "feature_step" / "features" / "offline" / "credentials.json")


def main():
    ap = argparse.ArgumentParser(
        description="Offline DB->message->features smoke check for one ZTF oid."
    )
    ap.add_argument("--oid", type=int, required=True,
                    help="Multisurvey bigint oid (e.g. 36028941624528297).")
    ap.add_argument("--credentials", default=DEFAULT_CREDENTIALS,
                    help="Path to DB credentials JSON (override for non-default envs).")
    ap.add_argument("--feature-version", default=None, dest="feature_version",
                    help="Feature-step version string (e.g. 27.5.7a31). "
                         "Defaults to importlib.metadata version('feature-step'); "
                         "supply this if the package is not installed in the active env.")
    ap.add_argument("--xmatch-url",
                    default=os.getenv("XMATCH_URL", xmatch.DEFAULT_XMATCH_URL),
                    dest="xmatch_url",
                    help="Xwave crossmatch service URL (default: %(default)s; or set "
                         "XMATCH_URL). Computes the AllWISE crossmatch live, like the "
                         "deployed step. Pass '' to force the DB read instead: "
                         "<schema>.allwise is empty for ZTF, so that yields NaN for every "
                         "WISE colour and biases the classification toward Stochastic.")
    ap.add_argument("--save", action="store_true",
                    help="Persist the DB-ready features into <schema>.feature.")
    ap.add_argument("--execute", action="store_true",
                    help="With --save, actually write (otherwise dry-run). "
                         "Requires --write-credentials.")
    ap.add_argument("--write-credentials", default=None, dest="write_credentials",
                    help="Credentials JSON with INSERT privileges; required when --execute "
                         "(the default credentials are read-only).")
    args = ap.parse_args()

    if args.execute and not args.save:
        ap.error("--execute only applies together with --save")
    if args.save and args.execute and not args.write_credentials:
        ap.error("--execute requires --write-credentials (the default credentials are read-only)")

    oid = args.oid
    credentials = args.credentials
    print(f"oid: {oid}")

    oids = [oid]
    dets = db.fetch_detections(credentials, oids)
    forced = db.fetch_forced_photometry(credentials, oids)
    ps1 = db.fetch_ps1(credentials, oids)
    refs = db.fetch_references(credentials, oids)

    # The cone centre lives in the message, so the crossmatch comes after it.
    message = build_message(oid, dets, forced, ps1)
    allwise, _matches = xmatch.allwise_for_oid(
        oid, message.get("meanra"), message.get("meandec"), credentials,
        xmatch_url=args.xmatch_url or None)
    print(f"detections={dets.shape} forced={forced.shape} ps1={ps1.shape} "
          f"allwise={allwise.shape} references={refs.shape}")
    print(f"xmatch: {'live Xwave @ ' + args.xmatch_url if args.xmatch_url else 'DB read (empty for ZTF)'}")
    print(f"message: {len(message['detections'])} detections")

    # Both LUT ids come from the DB, not the local fixture: <schema>.feature has
    # no FK to the LUTs, so a drifted fixture writes rows whose feature_id /
    # version resolve to something else and nothing rejects them (FLOW.md §3d).
    fver = args.feature_version
    if fver is None:
        try:
            fver = _pkg_version("feature-step")
        except PackageNotFoundError:
            fver = default_version_name()   # running from source, not installed
    feature_lut = db.fetch_feature_name_lut(credentials)
    version_id = db.fetch_feature_version_id(credentials, fver)
    print(f"feature LUT from DB: {len(feature_lut)} names; "
          f"version {fver} -> id {version_id}")

    features = lc_features.compute_db_features(message, refs, allwise,
                                               feature_name_lut=feature_lut,
                                               version_id=version_id)
    if features is None or len(features) == 0:
        print("\nFAIL: empty DB-ready features frame")
        sys.exit(1)
    print(f"\nDB-ready features: {features.shape}; columns={list(features.columns)}")
    print(features.head(20).to_string())
    print("\nOK: DB-ready feature rows produced.")

    if args.save:
        write_creds = args.write_credentials or credentials
        result = feature_writer.write_features(features, write_creds, execute=args.execute)
        if result["executed"]:
            print(f"\nSAVED: {result['written']} rows upserted into feature.")
        else:
            print(f"\nDRY RUN: would write {result['would_write']} rows "
                  f"(pass --execute with --write-credentials to write).")


if __name__ == "__main__":
    main()
