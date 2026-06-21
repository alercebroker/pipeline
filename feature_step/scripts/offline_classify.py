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
          PIPE / "alerce_classifiers"):
    sys.path.insert(0, str(p))

import argparse

from features.offline.classify import load_squidward_model, classify_oid

DEFAULT_CREDENTIALS = "/home/fandrades/desktop/repos/training/features_ztf/data/credentials.json"


def main():
    ap = argparse.ArgumentParser(
        description="Offline DB->features->BHRF probabilities for one ZTF oid."
    )
    ap.add_argument("--oid", type=int, required=True,
                    help="Multisurvey bigint oid (e.g. 36028941624528297).")
    ap.add_argument("--credentials", default=DEFAULT_CREDENTIALS,
                    help="Path to DB credentials JSON.")
    ap.add_argument("--min-det", type=int, default=1,
                    help="Minimum real detections required to classify.")
    args = ap.parse_args()

    model, name, version = load_squidward_model()
    print(f"model: {name} version={version}")
    print(f"oid: {args.oid}")

    result = classify_oid(args.oid, args.credentials, model, min_detections=args.min_det)
    if result is None or result.probabilities is None or len(result.probabilities) == 0:
        print("\nFAIL: no probabilities (too few detections or can't predict)")
        sys.exit(1)

    print(f"\nprobabilities:\n{result.probabilities.to_string()}")
    top = result.hierarchical.get("top")
    if top is not None and len(top):
        print(f"\ntop:\n{top.to_string()}")
    print("\nOK: probabilities produced.")


if __name__ == "__main__":
    main()
