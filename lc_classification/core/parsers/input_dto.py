from typing import List
import pandas as pd
from alerce_classifiers.base.factories import input_dto_factory
import numpy as np


def create_input_dto(messages: List[dict], **kwargs):
    features = create_features_dto(messages, kwargs.get("feature_list"))
    detections = pd.DataFrame()
    non_detections = pd.DataFrame()
    xmatch = pd.DataFrame()
    stamps = pd.DataFrame()
    input_dto = input_dto_factory(detections, non_detections, features, xmatch, stamps)
    return input_dto


def create_features_dto(messages: List[dict], feature_list) -> pd.DataFrame:
    """Creates a pandas dataframe with all the features from all messages

    The index is the aid and each feature is a column.

    Parameters
    -------
    messages : list
        a list of dictionaries with at least aid, candid and features keys.
    feature_list : list
        a list of feature names used for validation
    Returns
    -------
    pd.DataFrame
        A dataframe where each feature is a column indexed by aid.
        The rows are sorted by descending candid and duplicated aid are removed
        leaving the one with the latest candid.

    Examples
    --------
    >>> messages = [
            {
                'aid': 'aid1',
                'candid': 'cand1',
                'features': {'feat1': 1, 'feat2': 2}
            },
            {
                'aid': 'aid1',
                'candid': 'cand2',
                'features': {'feat1': 2, 'feat2': 3}
            },
            {
                'aid': 'aid2',
                'candid': 'cand3',
                'features': {'feat1': 4, 'feat2': 5}
            }
        ]
    >>> create_features_dto(messages)

        feat1  feat2 candid
    aid
    aid2      4      5  cand3
    aid1      2      3  cand2
    """
    df = pd.DataFrame(
        [
            {"aid": message.get("aid"), "candid": message.get("candid", np.nan)}
            for message in messages
        ]
    )
    features = pd.DataFrame([message["features"] for message in messages])
    features["aid"] = df.aid
    features["candid"] = df.candid
    features.sort_values("candid", ascending=False, inplace=True)
    features.drop_duplicates("aid", inplace=True)
    features = features.set_index("aid")
    if features is not None:
        _validate_features(features, feature_list)
        return features
    else:
        raise ValueError("Could not set index aid on features dataframe")


def _validate_features(features: pd.DataFrame, feature_list):
    required_features = set(feature_list)
    missing_features = required_features.difference(set(features.columns))
    if len(missing_features) > 0:
        raise KeyError(f"Corrupted Batch: missing some features ({missing_features})")
