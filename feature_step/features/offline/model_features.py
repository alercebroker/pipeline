"""Pure logic for the 199-feature coverage check.

`predict_input_columns` reproduces the EXACT column set that
HierarchicalRandomForestClassifier.classify_batch selects on via
`features[self.feature_list]`: parse_output names + band-suffixes the features,
SquidwardMapper turns None into NaN, and RandomForestPreprocessor renames columns.
Comparing that set against MODEL_FEATURE_LIST tells us whether predict() would
KeyError (missing name) for a given object.
"""
import numpy as np
import pandas as pd

from features.utils.parsers import parse_output
from alerce_classifiers.classifiers.preprocess import RandomForestPreprocessor


def predict_input_columns(ao, message: dict) -> list[str]:
    """Post-extract AstroObject + its message -> the column names predict() selects on.

    Mirrors features.offline.classify.classify_astro_object up to (but not
    including) the model call, then applies the model's own RandomForestPreprocessor
    so the returned names are in the same namespace as the model's feature_list.
    """
    candids = {message["oid"]: message.get("measurement_id", [])}
    out_message = parse_output([ao], [message], candids)[0]
    features = out_message.get("features") or {}
    df = pd.DataFrame([features], index=[message["oid"]])
    df.replace({None: np.nan}, inplace=True)          # SquidwardMapper.preprocess
    processed = RandomForestPreprocessor().preprocess_features(df)
    return list(processed.columns)


def diff_feature_coverage(produced, expected) -> dict:
    """Diff a produced name set against the expected model feature_list.

    Returns covered/missing/extra (sorted) plus counts. `missing` is the set the
    model would KeyError on.
    """
    produced_set = set(produced)
    expected_set = set(expected)
    missing = sorted(expected_set - produced_set)
    return {
        "covered": sorted(expected_set & produced_set),
        "missing": missing,
        "extra": sorted(produced_set - expected_set),
        "n_expected": len(expected_set),
        "n_missing": len(missing),
    }
