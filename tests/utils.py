from alerce_classifiers.base.factories import input_dto_factory
from typing import List

import os
import pandas as pd
import pickle


def create_mock_dto() -> List[dict]:
    base_dir = os.path.dirname(__file__)
    pickle_messages = open(
        os.path.join(base_dir, "./mockdata/mock_messages.pickle"), "rb"
    )
    messages = pickle.load(pickle_messages)
    mock_input_dto = create_input_dto(messages)
    return mock_input_dto


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
    input_dto = input_dto_factory(detections, non_detections, features, xmatch, stamps)
    return input_dto


def create_detections_dto(messages: List[dict]) -> pd.DataFrame:
    """Creates a pandas dataframe with all the detections from all messages

    Each row of the dataframe corresponds to a detection.

    Parameters
    -------
    messages : list
        a list of dictionaries with at least aid and detection keys.
    Returns
    -------
    pd.DataFrame
        A dataframe where each detection is a row.
        Duplicated aid from the input are removed but the output can contain duplicated aid.

    Examples
    --------
    >>> messages = [
            {
                'aid': 'aid1',
                'detections': [{'candid': 1, 'tid': 2, ...}, {'candid': 3, 'tid': 4, ...}]
            },
            {
                'aid': 'aid2',
                'detections': [{'candid': 5, 'tid': 6, ...}]
            },
        ]
    >>> create_features_dto(messages)

        candid   tid    ...
    aid
    aid1    1      2    ...
    aid1    3      4    ...
    aid2    5      6    ...
    """
    detections = pd.DataFrame([{"aid": message.get("aid")} for message in messages])
    detections["detections"] = [
        message["detections"] for message in messages if message["detections"]
    ]
    detections.drop_duplicates("aid", inplace=True)

    exploded_det = detections.explode("detections")
    detections = pd.DataFrame.from_records(
        exploded_det["detections"].values, index=exploded_det.index
    )
    detections["type"] = "d"
    if detections is not None:
        return detections
    else:
        raise ValueError("Could not set index aid on features dataframe")


def create_features_dto(messages: List[dict]) -> pd.DataFrame:
    """Creates a pandas dataframe with all the features from all messages

    The index is the aid and each feature is a column.

    Parameters
    -------
    messages : list
        a list of dictionaries with at least aid and features keys.
    Returns
    -------
    pd.DataFrame
        A dataframe where each feature is a column indexed by aid.
        Duplicated aid are removed.

    Examples
    --------
    >>> messages = [
            {
                'aid': 'aid1',
                'features': {'feat1': 1, 'feat2': 2}
            },
            {
                'aid': 'aid1',
                'features': {'feat1': 2, 'feat2': 3}
            },
            {
                'aid': 'aid2',
                'features': {'feat1': 4, 'feat2': 5}
            }
        ]
    >>> create_features_dto(messages)

        feat1  feat2
    aid
    aid2      4      5
    aid1      2      3
    """
    df = pd.DataFrame([{"aid": message.get("aid")} for message in messages])
    features = pd.DataFrame(
        [message["features"] for message in messages if message["features"]]
    )
    features["aid"] = df.aid
    features.drop_duplicates("aid", inplace=True)
    features = features.set_index("aid")
    if features is not None:
        return features
    else:
        raise ValueError("Could not set index aid on features dataframe")
