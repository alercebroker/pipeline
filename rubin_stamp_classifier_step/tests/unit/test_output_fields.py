"""pre_produce keeps the fields the producer's schema declares, nothing else.

The output schema is strict about extra fields, and the step adds internal
ones (oid, sid, the raw alert) that must not reach the topic. The field list
comes from the producer's loaded schema, not from a list kept in the step.
"""
import json
import os
from unittest import mock

from rubin_stamp_classifier_step.step import StampClassifierStep
from rubin_stamp_classifier_step.utils import LoggerProducer
from tests.unit.test_ss_object_handling import TAXONOMY, CLS_ID, processed

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUTPUT_SCHEMA = os.path.join(ROOT_DIR, "schemas", "rubin_stamp_classifier_step", "output.avsc")


def schema_fields(path):
    with open(path) as f:
        return {field["name"] for field in json.load(f)["fields"]}


def make_step(producer_config):
    config = {
        "CONSUMER_CONFIG": {"CLASS": "apf.core.step.DefaultConsumer"},
        "PRODUCER_CONFIG": producer_config,
        "DB_CONFIG": {},
        "MODEL_CONFIG": {
            "CLASS": "tests.unit.stub_model.StubModel",
            "PARAMS": {"model_path": "https://example.org/1.0.0/model.zip"},
            "CLS_ID": CLS_ID,
        },
    }
    with mock.patch("rubin_stamp_classifier_step.step.PSQLConnection"), \
         mock.patch("rubin_stamp_classifier_step.step.get_taxonomy_by_classifier_id", return_value=TAXONOMY):
        return StampClassifierStep(config=config)


def recording_producer(schema_path):
    return {"CLASS": "tests.unit.recording_producer.RecordingProducer", "SCHEMA_PATH": schema_path}


def test_pre_produce_keeps_the_fields_of_the_producer_schema():
    step = make_step(recording_producer(OUTPUT_SCHEMA))

    out = step.pre_produce(step.execute([processed(777, 2)]))

    assert set(out[0]) == schema_fields(OUTPUT_SCHEMA)


def test_pre_produce_follows_the_producer_schema_not_a_fixed_list(tmp_path):
    narrow = tmp_path / "narrow.avsc"
    narrow.write_text(json.dumps({
        "type": "record",
        "name": "narrow",
        "fields": [
            {"name": "diaSourceId", "type": "long"},
            {"name": "ra", "type": "double"},
        ],
    }))
    step = make_step(recording_producer(str(narrow)))

    out = step.pre_produce(step.execute([processed(777, 2)]))

    assert set(out[0]) == {"diaSourceId", "ra"}


def test_pre_produce_passes_messages_through_when_the_producer_has_no_schema():
    step = make_step({"CLASS": "apf.core.step.DefaultProducer"})
    messages = step.execute([processed(777, 2)])

    out = step.pre_produce(messages)

    assert out == messages
    assert {"oid", "sid", "alert"} <= set(out[0])


def test_logger_producer_loads_the_schema_it_is_given():
    producer = LoggerProducer({"SCHEMA_PATH": OUTPUT_SCHEMA})

    assert {f["name"] for f in producer.schema["fields"]} == schema_fields(OUTPUT_SCHEMA)
