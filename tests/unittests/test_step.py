from apf.producers import KafkaProducer
from pandas import DataFrame
from atlas_stamp_classifier_step.step import AtlasStampClassifierStep
from unittest.mock import MagicMock
import numpy as np
import pandas as pd
from atlas_stamp_classifier.inference import AtlasStampClassifier
import json


def test_message_to_df(alerts):
    consumer_mock = MagicMock()
    producer_mock = MagicMock()
    model_mock = MagicMock()
    step = AtlasStampClassifierStep(
        consumer=consumer_mock,
        config={"DB_CONFIG": {}},
        producer=producer_mock,
        scribe_producer=producer_mock,
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
    producer_mock = MagicMock()
    model_mock = MagicMock()
    step = AtlasStampClassifierStep(
        consumer=consumer_mock,
        config={"DB_CONFIG": {}},
        producer=producer_mock,
        scribe_producer=producer_mock,
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
    producer_mock = MagicMock()
    model = AtlasStampClassifier(model_dir=None, batch_size=1)
    step = AtlasStampClassifierStep(
        consumer=consumer_mock,
        config={"DB_CONFIG": {}},
        producer=producer_mock,
        scribe_producer=producer_mock,
        model=model,
    )
    stamps = step.message_to_df(alerts)
    probs = step.predict(stamps)
    ## asuming that the sample stamps are from an AGN object
    assert probs.idxmax(axis=1).iloc[0] == "agn"


def test_write_predictions():
    consumer_mock = MagicMock()
    producer_mock = MagicMock()
    scribe_producer = MagicMock(spec=KafkaProducer)
    model = AtlasStampClassifier(model_dir=None, batch_size=1)
    step = AtlasStampClassifierStep(
        consumer=consumer_mock,
        config={"DB_CONFIG": {}},
        producer=producer_mock,
        scribe_producer=scribe_producer,
        model=model,
    )
    predictions = DataFrame(
        [[0.5, 0.1, 0.3, 0.05, 0.05]],
        columns=["agn", "asteroid", "bogus", "sn", "vs"],
        index=["ZTF20aaelulu"],
    )
    step.write_predictions(predictions)
    scribe_data = {
        "collection": "object",
        "type": "update-probabilities",
        "criteria": {"aid": "ZTF20aaelulu"},
        "data": {
            "probabilities": [
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
        },
    }
    scribe_producer.produce.assert_called_once_with(
        {"payload": json.dumps(scribe_data)}
    )


def test_produce():
    consumer_mock = MagicMock()
    producer_mock = MagicMock()
    scribe_producer = MagicMock(spec=KafkaProducer)
    model = AtlasStampClassifier(model_dir=None, batch_size=1)
    step = AtlasStampClassifierStep(
        consumer=consumer_mock,
        config={"DB_CONFIG": {}},
        producer=producer_mock,
        scribe_producer=scribe_producer,
        model=model,
    )
    message = {
        "aid": "AL123",
        "classifications": [
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
        ],
    }
    output = [message]
    step.produce(output)
    producer_mock.produce.assert_called_once_with(message, key="AL123")
