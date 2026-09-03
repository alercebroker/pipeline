"""Print the five heads' class names, read off the model pickle itself.

`tests/integration/taxonomy_seed.py` holds a copy of this, because the tests must
be able to seed a database without loading a 1.6 GB pickle. Re-run this after a
model bump and paste the result into `HEAD_CLASSES` there; if the two ever
disagree, `build_probability_rows` logs the drifted class names and drops the
whole head, which the integration tests see as a short row count.

    python scripts/dump_model_taxonomy.py /path/to/model/2.1.0

The pickle is read directly rather than through `SquidwardFeaturesClassifier`, so
this needs no mapper and no `alerce_classifiers` import beyond pandas.
"""
import argparse
import json
import os

import pandas as pd

PICKLE_NAME = "hierarchical_random_forest_model.pkl"

# Keyed by the suffix `probabilities.HEAD_SUFFIXES` appends, so the output can be
# pasted into HEAD_CLASSES unchanged. The flat head's classes are the model's
# `list_of_classes`; the other four are their own estimator's `classes_`.
HEADS = {
    "_top": "top",
    "_transient": "Transient",
    "_stochastic": "Stochastic",
    "_periodic": "Periodic",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", help=f"the .pkl, or the directory holding {PICKLE_NAME}")
    args = parser.parse_args()

    path = args.model_path
    if not path.endswith(".pkl"):
        path = os.path.join(path, PICKLE_NAME)

    loaded = pd.read_pickle(path)
    classes = {"": list(loaded["list_of_classes"])}
    for suffix, head in HEADS.items():
        classes[suffix] = list(loaded["model"][head].classes_)

    print(json.dumps(classes, indent=4))


if __name__ == "__main__":
    main()
