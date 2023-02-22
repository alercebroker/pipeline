import sys
from unittest import mock

import pandas as pd
import pytest

if not sys.version.startswith('3.6'):
    pytest.skip("Incompatible Python version")

from atlas_stamp_classifier_step.strategies.ztf import ZTFStrategy


@mock.patch("atlas_stamp_classifier_step.strategies.ztf.StampClassifier")
def test_transform_messages_to_dataframe(mock_classifier, ztf_alerts):
    strategy = ZTFStrategy()

    df = strategy._to_dataframe(ztf_alerts)
    assert isinstance(df.iloc[0]["cutoutScience"], bytes)
    assert isinstance(df.iloc[0]["cutoutTemplate"], bytes)
    assert isinstance(df.iloc[0]["cutoutDifference"], bytes)


def test_prediction_with_stamp_classifier(ztf_alerts):
    strategy = ZTFStrategy()

    df = strategy._to_dataframe(ztf_alerts)
    probs = strategy.predict(df)
    # assuming that the sample stamps are from an AGN object
    assert probs.idxmax(axis=1).iloc[0] == "AGN"


@mock.patch("atlas_stamp_classifier_step.strategies.ztf.StampClassifier")
def test_get_probabilities_reformats_dictionary(mock_classifier, ztf_alerts):
    mock_classifier.return_value.execute.return_value = pd.DataFrame(
        [[0.5, 0.1, 0.3, 0.05, 0.05]],
        columns=["AGN", "asteroid", "bogus", "SN", "VS"],
        index=["ZTF20aaelulu"],
    )
    strategy = ZTFStrategy()

    # alerts works as a dummy here
    probs = strategy.get_probabilities(ztf_alerts)
    expected = {
            "ZTF20aaelulu": [
                {
                    "ranking": 1,
                    "class_name": "AGN",
                    "classifier_name": "ztf_stamp_classifier",
                    "classifier_version": "1.0.1",
                    "probability": 0.5,
                },
                {
                    "ranking": 2,
                    "class_name": "bogus",
                    "classifier_name": "ztf_stamp_classifier",
                    "classifier_version": "1.0.1",
                    "probability": 0.3,
                },
                {
                    "ranking": 3,
                    "class_name": "asteroid",
                    "classifier_name": "ztf_stamp_classifier",
                    "classifier_version": "1.0.1",
                    "probability": 0.1,
                },
                {
                    "ranking": 4,
                    "class_name": "SN",
                    "classifier_name": "ztf_stamp_classifier",
                    "classifier_version": "1.0.1",
                    "probability": 0.05,
                },
                {
                    "ranking": 5,
                    "class_name": "VS",
                    "classifier_name": "ztf_stamp_classifier",
                    "classifier_version": "1.0.1",
                    "probability": 0.05,
                },
            ]
        }
    assert probs == expected
