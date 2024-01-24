from unittest import mock

import pandas as pd
import pandas.testing
import tensorflow as tf
import pytest

if not tf.__version__.startswith("1"):
    pytest.skip("Incompatible TensorFlow version", allow_module_level=True)

from stamp_classifier_step.strategies.ztf import ZTFStrategy


@mock.patch("stamp_classifier_step.strategies.ztf.StampClassifier")
def test_transform_messages_to_dataframe(mock_classifier, ztf_alerts):
    strategy = ZTFStrategy()

    df = strategy._to_dataframe(ztf_alerts)
    assert isinstance(df.iloc[0]["cutoutScience"], bytes)
    assert isinstance(df.iloc[0]["cutoutTemplate"], bytes)
    assert isinstance(df.iloc[0]["cutoutDifference"], bytes)


@mock.patch("stamp_classifier_step.strategies.ztf.StampClassifier")
def test_prediction_with_stamp_classifier(mock_classifier, ztf_alerts):
    strategy = ZTFStrategy()

    mock_classifier.return_value.execute.return_value = pd.DataFrame(
        [[0.5, 0.1, 0.3, 0.05, 0.05]],
        columns=["AGN", "asteroid", "bogus", "SN", "VS"],
        index=["ZTF20aaelulu"],
    )

    df = strategy._to_dataframe(ztf_alerts)
    strategy.get_probabilities(ztf_alerts)

    (df_called,) = mock_classifier.return_value.execute.call_args[0]

    pandas.testing.assert_frame_equal(df, df_called)


@mock.patch("stamp_classifier_step.strategies.ztf.StampClassifier")
def test_duplicate_aid_keeps_first(mock_classifier, ztf_alerts):
    strategy = ZTFStrategy()

    mock_classifier.return_value.execute.return_value = pd.DataFrame(
        [[0.5, 0.1, 0.3, 0.05, 0.05]],
        columns=["AGN", "asteroid", "bogus", "SN", "VS"],
        index=["ZTF20aaelulu"],
    )

    (first,) = ztf_alerts
    second, third, fourth = first.copy(), first.copy(), first.copy()
    second["mjd"], third["mjd"], fourth["aid"] = 59995, 59996, "otherid"

    df = strategy._to_dataframe([first, fourth])
    strategy.get_probabilities([second, first, third, fourth])

    (df_called,) = mock_classifier.return_value.execute.call_args[0]

    pandas.testing.assert_frame_equal(df, df_called)


@mock.patch("stamp_classifier_step.strategies.ztf.StampClassifier")
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
        "ZTF20aaelulu": {
            "AGN": 0.5,
            "bogus": 0.3,
            "asteroid": 0.1,
            "SN": 0.05,
            "VS": 0.05,
        },
    }
    assert probs == expected
