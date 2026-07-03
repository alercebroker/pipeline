#!/usr/bin/env python
"""Cross-check the classifier_taxonomy_lut fixture against the deployed BHRF pickle.

Loads the deployed Squidward 2.1.0 model at MODEL_PATH and asserts the fixture's
class names + order exactly match the model: TAXONOMY_LUT[5] == the flat
list_of_classes (21 leaves), and each branch (TAXONOMY_LUT[6..9]) == that branch
RF's classes_. This guards the DB write path, which maps class_name -> class_id by
exact string match (e.g. the SESN vs SNIbc label risk). Exit 0 on match.

    MODEL_PATH=https://alerce-models.s3.amazonaws.com/squidward/2.1.0/hierarchical_random_forest_model.pkl \
        conda run --no-capture-output -n training_py310 python \
        feature_step/scripts/offline_verify_taxonomy.py

Requires an env where imblearn/alerce_classifiers import cleanly (training_py310).

Attribute paths (verified against alerce_classifiers source):
  * SquidwardFeaturesClassifier.model -> HierarchicalRandomForestClassifier
      (alerce_classifiers/squidward/model.py:20-21)
  * .model.list_of_classes -> flat 21 class names
      (hierarchical_random_forest.py:193, loaded_data["list_of_classes"])
  * .model.dict_of_rf[key].classes_ -> per-branch class names
      (dict_of_rf keys: top / Stochastic / Periodic / Transient, line 197;
       classes_ used as prob columns at line 59)
"""
import os
import sys
from pathlib import Path

PIPE = Path(__file__).resolve().parents[2]   # .../pipeline
for p in (PIPE / "feature_step", PIPE / "lc_classifier",
          PIPE / "libs" / "idmapper", PIPE / "libs" / "apf",
          PIPE / "alerce_classifiers"):
    sys.path.insert(0, str(p))

from features.offline.classifier_taxonomy_lut import TAXONOMY_LUT

# fixture classifier_id -> model dict_of_rf branch key
BRANCH_KEY = {6: "top", 7: "Transient", 8: "Stochastic", 9: "Periodic"}


def main() -> int:
    if not os.environ.get("MODEL_PATH"):
        print("MODEL_PATH not set — point it at the deployed pickle.", file=sys.stderr)
        return 2

    from features.offline.classify import load_squidward_model

    model, name, version = load_squidward_model()
    hrf = model.model  # HierarchicalRandomForestClassifier
    print(f"model: {name} version={version}")

    problems = []

    flat_model = [str(c) for c in hrf.list_of_classes]
    if flat_model != TAXONOMY_LUT[5]:
        problems.append(f"flat (classifier_id 5): fixture {TAXONOMY_LUT[5]} "
                        f"!= model {flat_model}")

    for cid, key in BRANCH_KEY.items():
        model_classes = [str(c) for c in hrf.dict_of_rf[key].classes_]
        if model_classes != TAXONOMY_LUT[cid]:
            problems.append(f"{key} (classifier_id {cid}): fixture {TAXONOMY_LUT[cid]} "
                            f"!= model {model_classes}")

    if problems:
        print("MISMATCH — fixture disagrees with the deployed model:")
        for p in problems:
            print("  -", p)
        return 1
    print("OK — fixture class names + order match the deployed model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
