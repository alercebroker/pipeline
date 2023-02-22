import sys
from unittest import mock

import pandas as pd
import pytest

if sys.version.startswith('3.6'):
    pytest.skip("Incompatible Python version")

from atlas_stamp_classifier_step.strategies.atlas import ATLASStrategy


@mock.patch("atlas_stamp_classifier_step.strategies.atlas.AtlasStampClassifier")
def test_transform_messages_to_dataframe(mock_classifier, alerts):
    strategy = ATLASStrategy()

    df = strategy._to_dataframe(alerts)
    assert df.iloc[0]["red"].shape == (61, 61)
    assert df.iloc[0]["diff"].shape == (61, 61)


def test_prediction_with_stamp_classifier(alerts):
    strategy = ATLASStrategy()

    df = strategy._to_dataframe(alerts)
    probs = strategy.predict(df)
    # assuming that the sample stamps are from an AGN object
    assert probs.idxmax(axis=1).iloc[0] == "agn"


@mock.patch("atlas_stamp_classifier_step.strategies.atlas.AtlasStampClassifier")
def test_get_probabilities_reformats_dictionary(mock_classifier, alerts):
    mock_classifier.return_value.predict_probs.return_value = pd.DataFrame(
        [[0.5, 0.1, 0.3, 0.05, 0.05]],
        columns=["agn", "asteroid", "bogus", "sn", "vs"],
        index=["ZTF20aaelulu"],
    )
    strategy = ATLASStrategy()

    # alerts works as a dummy here
    probs = strategy.get_probabilities(alerts)
    expected = {
            "ZTF20aaelulu": [
                {
                    "ranking": 1,
                    "class_name": "agn",
                    "classifier_name": "atlas_stamp_classifier",
                    "classifier_version": "1.0.0",
                    "probability": 0.5,
                },
                {
                    "ranking": 2,
                    "class_name": "bogus",
                    "classifier_name": "atlas_stamp_classifier",
                    "classifier_version": "1.0.0",
                    "probability": 0.3,
                },
                {
                    "ranking": 3,
                    "class_name": "asteroid",
                    "classifier_name": "atlas_stamp_classifier",
                    "classifier_version": "1.0.0",
                    "probability": 0.1,
                },
                {
                    "ranking": 4,
                    "class_name": "sn",
                    "classifier_name": "atlas_stamp_classifier",
                    "classifier_version": "1.0.0",
                    "probability": 0.05,
                },
                {
                    "ranking": 5,
                    "class_name": "vs",
                    "classifier_name": "atlas_stamp_classifier",
                    "classifier_version": "1.0.0",
                    "probability": 0.05,
                },
            ]
        }
    assert probs == expected
