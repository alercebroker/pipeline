import json
from apf.consumers import KafkaConsumer
from correction._step import CorrectionStep
import logging


def assert_result_has_correction_fields(message):
    fields = ["mag_corr", "e_mag_corr", "e_mag_corr_ext", "has_stamp", "corrected", "stellar"]
    assert all(all(f in det for f in fields) for det in message["detections"])


def assert_result_has_extra_fields(message):
    for det in message["detections"]:
        assert "extra_fields" in det
        if det["tid"] == "LSST":
            assert "diaObject" in det["extra_fields"]


def assert_scribe_has_detections(messages):
    for message in messages:
        assert "collection" in message and (
            message["collection"] == "detection" or message["collection"] == "forced_photometry"
        )
        assert "type" in message and message["type"] == "update"
        assert (
            "criteria" in message
            and "candid" in message["criteria"]
            and "oid" in message["criteria"]
        )
        assert "data" in message and len(message["data"]) > 0
        assert (
            "options" in message
            and "upsert" in message["options"]
            and "set_on_insert" in message["options"]
        )
        assert message["options"]["upsert"] is True
        assert "has_stamp" in message["data"]
        # Prevent duplication with _id
        assert "candid" not in message["data"]
        if message["data"]["has_stamp"]:
            assert message["options"]["set_on_insert"] is False
        else:
            assert message["options"]["set_on_insert"] is True
        if message["data"]["tid"] == "LSST":
            assert "diaObject" in message["data"]["extra_fields"]


def assert_any_forced(messages):
    any_forced = False
    for msg in messages:
        if msg["collection"] == "forced_photometry":
            any_forced = True
    assert any_forced


def deserialize_messages(messages):
    result = []
    for msg in messages:
        result.append(json.loads(msg["payload"]))
    return result


def test_result_has_everything(kafka_service, env_variables, kafka_consumer):
    CorrectionStep.create_step().start()
    for message in kafka_consumer.consume():
        assert "oid" in message
        assert "detections" in message
        assert "non_detections" in message
        assert "meanra" in message
        assert "meandec" in message
        assert "candid" in message
        assert_result_has_correction_fields(message)
        assert_result_has_extra_fields(message)
        kafka_consumer.commit()


def test_scribe_has_detections(kafka_service, env_variables, scribe_consumer):
    CorrectionStep.create_step().start()

    messages = list(scribe_consumer.consume())
    messages = deserialize_messages(messages)
    assert len(messages)
    assert_scribe_has_detections(messages)
    assert_any_forced(messages)


def test_works_with_batch(kafka_service, env_variables, kafka_consumer: KafkaConsumer, caplog):
    import os

    os.environ["CONSUME_MESSAGES"] = "10"
    os.environ["LOGGING_DEBUG"] = "yes"
    caplog.set_level(logging.DEBUG)
    CorrectionStep.create_step().start()
    processed = False
    for message in kafka_consumer.consume():
        processed = True
        assert_result_has_correction_fields(message)
        assert_result_has_extra_fields(message)
        kafka_consumer.commit()
    assert processed
