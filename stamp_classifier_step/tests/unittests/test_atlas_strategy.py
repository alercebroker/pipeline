from unittest import mock

import pandas as pd
import pandas.testing
import tensorflow as tf
import pytest

if tf.__version__.startswith("1"):
    pytest.skip("Incompatible TensorFlow version", allow_module_level=True)

from stamp_classifier_step.strategies.atlas import ATLASStrategy


@mock.patch("stamp_classifier_step.strategies.atlas.AtlasStampClassifier")
def test_transform_messages_to_dataframe(mock_classifier, alerts):
    strategy = ATLASStrategy()

    df = strategy._to_dataframe(alerts)
    assert df.iloc[0]["red"].shape == (61, 61)
    assert df.iloc[0]["diff"].shape == (61, 61)


@mock.patch("stamp_classifier_step.strategies.atlas.AtlasStampClassifier")
def test_prediction_with_stamp_classifier(mock_classifier, alerts):
    strategy = ATLASStrategy()

    df = strategy._to_dataframe(alerts)
    strategy.get_probabilities(alerts)

    (df_called,) = mock_classifier.return_value.predict_probs.call_args[0]

    pandas.testing.assert_frame_equal(df, df_called)


@mock.patch("stamp_classifier_step.strategies.atlas.AtlasStampClassifier")
def test_duplicate_aid_keeps_first(mock_classifier, alerts):
    strategy = ATLASStrategy()

    (first,) = alerts
    second, third, fourth = first.copy(), first.copy(), first.copy()
    second["mjd"], third["mjd"], fourth["aid"] = 100, 1000, "otherid"

    df = strategy._to_dataframe([first, fourth])
    strategy.get_probabilities([second, first, third, fourth])

    (df_called,) = mock_classifier.return_value.predict_probs.call_args[0]

    pandas.testing.assert_frame_equal(df, df_called)


@mock.patch("stamp_classifier_step.strategies.atlas.AtlasStampClassifier")
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
        "ZTF20aaelulu": {
            "agn": 0.5,
            "bogus": 0.3,
            "asteroid": 0.1,
            "sn": 0.05,
            "vs": 0.05,
        },
    }
    assert probs == expected
