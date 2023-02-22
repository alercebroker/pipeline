import json
from apf.producers import KafkaProducer
from atlas_stamp_classifier_step.step import AtlasStampClassifierStep
from unittest.mock import MagicMock


def test_format_output_message():
    predictions = {
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
    consumer_mock = MagicMock()
    producer_mock = MagicMock()
    strategy_mock = MagicMock()
    step = AtlasStampClassifierStep(
        consumer=consumer_mock,
        config={},
        producer=producer_mock,
        scribe_producer=producer_mock,
        strategy=strategy_mock,
    )
    output = step.format_output_message(predictions)
    assert output[0]["classifications"] == [
        {
            "classifier_name": "atlas_stamp_classifier",
            "class_name": "agn",
            "probability": 0.5,
        },
        {
            "classifier_name": "atlas_stamp_classifier",
            "class_name": "bogus",
            "probability": 0.3,
        },
        {
            "classifier_name": "atlas_stamp_classifier",
            "class_name": "asteroid",
            "probability": 0.1,
        },
        {
            "classifier_name": "atlas_stamp_classifier",
            "class_name": "sn",
            "probability": 0.05,
        },
        {
            "classifier_name": "atlas_stamp_classifier",
            "class_name": "vs",
            "probability": 0.05,
        },
    ]


def test_write_predictions():
    consumer_mock = MagicMock()
    producer_mock = MagicMock()
    scribe_producer = MagicMock(spec=KafkaProducer)
    strategy_mock = MagicMock()
    step = AtlasStampClassifierStep(
        consumer=consumer_mock,
        config={},
        producer=producer_mock,
        scribe_producer=scribe_producer,
        strategy=strategy_mock,
    )
    predictions = {
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
    strategy_mock = MagicMock()
    step = AtlasStampClassifierStep(
        consumer=consumer_mock,
        config={"DB_CONFIG": {}},
        producer=producer_mock,
        scribe_producer=scribe_producer,
        strategy=strategy_mock,
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
