import json
from apf.producers import KafkaProducer
from stamp_classifier_step.step import StampClassifierStep
from unittest.mock import MagicMock


def test_format_output_message():
    predictions = {
        "ZTF20aaelulu": {
            "agn": 0.5,
            "bogus": 0.3,
            "asteroid": 0.1,
            "sn": 0.05,
            "vs": 0.05,
        },
    }
    consumer_mock = MagicMock()
    producer_mock = MagicMock()
    strategy_mock = MagicMock()
    step = StampClassifierStep(
        consumer=consumer_mock,
        producer=producer_mock,
        scribe_producer=producer_mock,
        strategy=strategy_mock,
    )
    output = step.format_output_message(predictions)
    assert output[0]["classifications"] == [
        {
            "class_name": "agn",
            "probability": 0.5,
        },
        {
            "class_name": "bogus",
            "probability": 0.3,
        },
        {
            "class_name": "asteroid",
            "probability": 0.1,
        },
        {
            "class_name": "sn",
            "probability": 0.05,
        },
        {
            "class_name": "vs",
            "probability": 0.05,
        },
    ]


def test_write_predictions():
    consumer_mock = MagicMock()
    producer_mock = MagicMock()
    scribe_producer = MagicMock(spec=KafkaProducer)
    strategy_mock = MagicMock()
    strategy_mock.name = "atlas_stamp_classifier"
    strategy_mock.version = "1.0.0"
    step = StampClassifierStep(
        consumer=consumer_mock,
        producer=producer_mock,
        scribe_producer=scribe_producer,
        strategy=strategy_mock,
    )
    predictions = {
        "ZTF20aaelulu": {
            "agn": 0.5,
            "bogus": 0.3,
            "asteroid": 0.1,
            "sn": 0.05,
            "vs": 0.05,
        }
    }

    step.write_predictions(predictions)
    scribe_data = {
        "collection": "object",
        "type": "update_probabilities",
        "criteria": {"_id": "ZTF20aaelulu"},
        "data": {
            "classifier_name": "atlas_stamp_classifier",
            "classifier_version": "1.0.0",
            "agn": 0.5,
            "bogus": 0.3,
            "asteroid": 0.1,
            "sn": 0.05,
            "vs": 0.05,
        },
        "options": {"upsert": True, "set_on_insert": True},
    }
    scribe_producer.produce.assert_called_once_with(
        {"payload": json.dumps(scribe_data)}
    )


def test_produce():
    consumer_mock = MagicMock()
    producer_mock = MagicMock()
    scribe_producer = MagicMock(spec=KafkaProducer)
    strategy_mock = MagicMock()
    step = StampClassifierStep(
        consumer=consumer_mock,
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
