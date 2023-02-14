from pandas import DataFrame
from atlas_stamp_classifier_step.step import AtlasStampClassifierStep
from unittest.mock import MagicMock
import numpy as np
from atlas_stamp_classifier.inference import AtlasStampClassifier


def test_message_to_df(alerts):
    consumer_mock = MagicMock()
    database_mock = MagicMock()
    producer_mock = MagicMock()
    model_mock = MagicMock()
    step = AtlasStampClassifierStep(
        consumer=consumer_mock,
        config={"DB_CONFIG": {}},
        db_connection=database_mock,
        producer=producer_mock,
        model=model_mock,
    )
    df = step.message_to_df(alerts)
    assert df.iloc[0]["red"].shape == (61, 61)
    assert df.iloc[0]["diff"].shape == (61, 61)
    print(df)


def test_format_output_message():
    predictions = DataFrame(
        [[0.5, 0.1, 0.3, 0.05, 0.05]],
        columns=["agn", "asteroid", "bogus", "sn", "vs"],
        index=["oid"],
    )
    stamps = DataFrame([[np.zeros((61, 61))]], index=["oid"], columns=["red"])
    consumer_mock = MagicMock()
    database_mock = MagicMock()
    producer_mock = MagicMock()
    model_mock = MagicMock()
    step = AtlasStampClassifierStep(
        consumer=consumer_mock,
        config={"DB_CONFIG": {}},
        db_connection=database_mock,
        producer=producer_mock,
        model=model_mock,
    )
    output = step.format_output_message(predictions, stamps)
    assert output[0]["classifications"] == [
        {
            "classifier_name": "atlas_stamp_classifier",
            "model_version": "1.0.0",
            "class_name": "agn",
            "probability": 0.5,
        },
        {
            "classifier_name": "atlas_stamp_classifier",
            "model_version": "1.0.0",
            "class_name": "asteroid",
            "probability": 0.1,
        },
        {
            "classifier_name": "atlas_stamp_classifier",
            "model_version": "1.0.0",
            "class_name": "bogus",
            "probability": 0.3,
        },
        {
            "classifier_name": "atlas_stamp_classifier",
            "model_version": "1.0.0",
            "class_name": "sn",
            "probability": 0.05,
        },
        {
            "classifier_name": "atlas_stamp_classifier",
            "model_version": "1.0.0",
            "class_name": "vs",
            "probability": 0.05,
        },
    ]


def test_predict(alerts):
    consumer_mock = MagicMock()
    database_mock = MagicMock()
    producer_mock = MagicMock()
    model = AtlasStampClassifier(model_dir=None, batch_size=1)
    step = AtlasStampClassifierStep(
        consumer=consumer_mock,
        config={"DB_CONFIG": {}},
        db_connection=database_mock,
        producer=producer_mock,
        model=model,
    )
    stamps = step.message_to_df(alerts)
    probs = step.predict(stamps)
    ## asuming that the sample stamps are from an AGN object
    assert probs.idxmax(axis=1).iloc[0] == "agn"
