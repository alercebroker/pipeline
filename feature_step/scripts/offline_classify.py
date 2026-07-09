#!/usr/bin/env python
"""Live-DB check: DB -> message -> features -> BHRF probabilities for one ZTF oid.

Requires MODEL_PATH (and optionally MAPPER_CLASS) env vars, same as the deployed step:

    MODEL_PATH=https://alerce-models.s3.amazonaws.com/squidward/2.1.0/hierarchical_random_forest_model.pkl \
        conda run --no-capture-output -n training_py310 python \
        feature_step/scripts/offline_classify.py --oid 36028941624528297
"""
import sys
from pathlib import Path

PIPE = Path(__file__).resolve().parents[2]   # .../pipeline
for p in (PIPE / "feature_step", PIPE / "lc_classifier",
          PIPE / "libs" / "idmapper", PIPE / "libs" / "apf",
          PIPE / "libs" / "xmatch_client", PIPE / "alerce_classifiers"):
    sys.path.insert(0, str(p))

import argparse
import os

from features.offline.classify import load_squidward_model, classify_oid_for_save
from features.offline import db, probability_writer, xmatch

DEFAULT_CREDENTIALS = str(PIPE / "feature_step" / "features" / "offline" / "credentials.json")


def main():
    ap = argparse.ArgumentParser(
        description="Offline DB->features->BHRF probabilities for one ZTF oid."
    )
    ap.add_argument("--oid", type=int, required=True,
                    help="Multisurvey bigint oid (e.g. 36028941624528297).")
    ap.add_argument("--credentials", default=DEFAULT_CREDENTIALS,
                    help="Path to DB credentials JSON (read; also used to read taxonomy).")
    ap.add_argument("--min-det", type=int, default=1,
                    help="Minimum real detections required to classify.")
    ap.add_argument("--save", action="store_true",
                    help="Persist probabilities into <schema>.probability "
                         "(dry-run unless --execute).")
    ap.add_argument("--execute", action="store_true",
                    help="With --save, actually write. Requires --write-credentials.")
    ap.add_argument("--write-credentials", default=None, dest="write_credentials",
                    help="Write-capable DB credentials JSON (default credentials are read-only).")
    ap.add_argument("--xmatch-url", default=os.getenv("XMATCH_URL", xmatch.DEFAULT_XMATCH_URL),
                    dest="xmatch_url",
                    help="Xwave crossmatch service URL (default: %(default)s; or set XMATCH_URL). "
                         "The AllWISE crossmatch is computed live like the step, instead of "
                         "reading the (empty) multisurvey_ztf.xmatch table. Pass '' to force the "
                         "DB-read fallback.")
    ap.add_argument("--persist-xmatch", action="store_true",
                    help="Save the computed crossmatch directly to the DB (PLACEHOLDER — "
                         "dry-run only; requires --xmatch-url).")
    args = ap.parse_args()

    if args.save and args.execute and not args.write_credentials:
        ap.error("--execute requires --write-credentials (the default credentials are read-only)")
    if args.persist_xmatch and not args.xmatch_url:
        ap.error("--persist-xmatch requires --xmatch-url (nothing is computed to persist otherwise)")

    model, name, version = load_squidward_model()
    print(f"model: {name} version={version}")
    print(f"oid: {args.oid}")
    print(f"xmatch: {'live Xwave @ ' + args.xmatch_url if args.xmatch_url else 'DB read (multisurvey_ztf.xmatch)'}")

    result, lastmjd, matches = classify_oid_for_save(args.oid, args.credentials, model,
                                                     min_detections=args.min_det,
                                                     xmatch_url=args.xmatch_url)
    if result is None or result.probabilities is None or len(result.probabilities) == 0:
        print("\nFAIL: no probabilities (too few detections or can't predict)")
        sys.exit(1)

    if args.xmatch_url:
        aw = xmatch.matches_to_allwise_df(matches)
        print(f"\ncrossmatch: {len(matches)} match(es); AllWISE rows:\n{aw.to_string(index=False)}")

    print(f"\nprobabilities:\n{result.probabilities.to_string()}")
    top = result.hierarchical.get("top")
    if top is not None and len(top):
        print(f"\ntop:\n{top.to_string()}")
    print("\nOK: probabilities produced.")

    if args.persist_xmatch:
        rows = xmatch.persist_matches(matches, write_credentials=args.write_credentials,
                                      execute=False)
        print(f"\npersist-xmatch (PLACEHOLDER, dry-run): {len(rows)} row(s) that would be "
              f"written to {db.SCHEMA}.xmatch:")
        for r in rows:
            print(f"  {r}")

    if args.save:
        # class_id authority is the DB taxonomy — read it (read-only creds).
        taxonomy_maps = db.fetch_taxonomy_maps(args.credentials,
                                               probability_writer.CLASSIFIER_IDS)
        rows = probability_writer.build_probability_rows(result, args.oid, lastmjd,
                                                         taxonomy_maps, version=version)
        write_creds = args.write_credentials or args.credentials
        outcome = probability_writer.write_probabilities(
            rows, write_creds, execute=args.execute)
        print(f"\nsave: {outcome} (lastmjd={lastmjd})")
        if not args.execute:
            print("(dry-run — pass --execute with --write-credentials to write)")


if __name__ == "__main__":
    main()
