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

from features.offline import db, lc_features
from features.offline.message import build_message

DEFAULT_CREDENTIALS = "/home/fandrades/desktop/repos/training/features_ztf/data/credentials.json"


def main():
    ap = argparse.ArgumentParser(
        description="Offline DB->message->features smoke check for one ZTF oid."
    )
    ap.add_argument("--oid", type=int, required=True,
                    help="Multisurvey bigint oid (e.g. 36028941624528297).")
    ap.add_argument("--credentials", default=DEFAULT_CREDENTIALS,
                    help="Path to DB credentials JSON (override for non-default envs).")
    args = ap.parse_args()

    oid = args.oid
    credentials = args.credentials
    print(f"oid: {oid}")

    oids = [oid]
    dets = db.fetch_detections(credentials, oids)
    forced = db.fetch_forced_photometry(credentials, oids)
    ps1 = db.fetch_ps1(credentials, oids)
    allwise = db.fetch_allwise(credentials, oids)
    refs = db.fetch_references(credentials, oids)
    print(f"detections={dets.shape} forced={forced.shape} ps1={ps1.shape} "
          f"allwise={allwise.shape} references={refs.shape}")

    message = build_message(oid, dets, forced, ps1)
    print(f"message: {len(message['detections'])} detections")

    features = lc_features.compute_features(message, refs, allwise)
    if features is None or len(features) == 0 or features["value"].notna().sum() == 0:
        print("\nFAIL: empty / all-NaN features frame")
        sys.exit(1)
    print(f"\nfeatures: {features.shape}; "
          f"non-null values: {features['value'].notna().sum()}/{len(features)}")
    print(features.head(20).to_string())
    print("\nOK: populated features frame produced.")


if __name__ == "__main__":
    main()
