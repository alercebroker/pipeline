from typing import List
import pandas as pd
from alerce_classifiers.base.factories import input_dto_factory
import pickle


def create_input_dto(messages: List[dict]):
    """Creates a InputDTO object with necessary inputs for models.

    Parameters
    ----------
    messages : List[dict]
        The list of messages as they come from the step execute method.
        The schema of each dict should match the previous step's (FeaturesStep) schema
    """
    features = create_features_dto(messages)
    detections = create_detections_dto(messages)
    non_detections = pd.DataFrame()
    xmatch = pd.DataFrame()
    stamps = pd.DataFrame()
    input_dto = input_dto_factory(
        detections, non_detections, features, xmatch, stamps
    )
    return input_dto


def create_detections_dto(messages: List[dict]) -> pd.DataFrame:
    """Creates a pandas dataframe with all detections.

    Examples
    --------
    >>> messages = [
            {
                "detections": [
                    {"oid": "oid1", "candid": "cand1"},
                    {"oid": "oid1", "candid": "cand2"},
                ]
            },
            {
                "detections": [
                    {"oid": "oid2", "candid": "cand3"},
                ]
            },
        ]
    >>> create_detections_dto(messages)
                candid
        oid
        oid1    cand1
        oid2    cand3
    """
    detections = [
        pd.DataFrame.from_records(msg["detections"]) for msg in messages
    ]
    detections = pd.concat(detections)
    detections.drop_duplicates(["candid", "oid"], inplace=True)
    detections = detections.set_index("oid")
    detections["extra_fields"] = parse_extra_fields(detections)

    if detections is not None:
        return detections
    else:
        raise ValueError("Could not set index oid on features dataframe")


def parse_extra_fields(detections: pd.DataFrame) -> List[dict]:
    for ef in detections["extra_fields"]:
        for key in ef.copy():
            if type(ef[key]) is bytes:
                extra_field = pickle.loads(ef[key])
                # the loaded pickle is a list of one element
                ef[key] = extra_field[0]
    return detections["extra_fields"]


def create_features_dto(messages: List[dict]) -> pd.DataFrame:
    """Creates a pandas dataframe with all the features from all messages

    The index is the oid and each feature is a column.

    Parameters
    -------
    messages : list
        a list of dictionaries with at least oid and features keys.
    Returns
    -------
    pd.DataFrame
        A dataframe where each feature is a column indexed by oid.
        Duplicated oid are removed.

    Examples
    --------
    >>> messages = [
            {
                'oid': 'oid1',
                'features': {'feat1': 1, 'feat2': 2}
            },
            {
                'oid': 'oid1',
                'features': {'feat1': 2, 'feat2': 3}
            },
            {
                'oid': 'oid2',
                'features': {'feat1': 4, 'feat2': 5}
            }
        ]
    >>> create_features_dto(messages)

        feat1  feat2
    oid
    oid2      4      5
    oid1      2      3
    """
    if len(messages) == 0 or "features" not in messages[0]:
        return pd.DataFrame()
    entries = []
    for message in messages:
        if message["features"] is None:
            continue
        entry = {
            feat: message["features"][feat] for feat in message["features"]
        }
        entry["oid"] = message["oid"]
        entries.append(entry)
    if len(entries) == 0:
        return pd.DataFrame()

    features = pd.DataFrame.from_records(entries)
    features.drop_duplicates("oid", inplace=True, keep="last")
    features = features.set_index("oid")
    if features is not None:
        return features
    else:
        raise ValueError("Could not set index oid on features dataframe")
