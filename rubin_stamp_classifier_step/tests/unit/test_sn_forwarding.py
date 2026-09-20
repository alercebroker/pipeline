"""The SN forwarder: the rubin deployment re-emits the raw LSST alert of every
object whose ranking-1 class is SN into a second topic, so the hunter
deployment can consume exactly what the rubin deployment consumed.

The forwarder is optional (SN_FORWARD_PRODUCER_CONFIG). Without it the step
behaves as before. These tests run without a database, a broker, or a model.
"""
import io
import math
import os
from unittest import mock

import fastavro
import pytest
import yaml

from rubin_stamp_classifier_step.step import StampClassifierStep
from tests.unit.stub_model import StubModel
from tests.unit.test_ss_object_handling import TAXONOMY, CLS_ID, alert, processed, OUTPUT_SCHEMA_FIELDS

FORWARD_TOPIC = "sn_candidates"


def make_step(forward_config=None, forward_class=None):
    config = {
        "CONSUMER_CONFIG": {"CLASS": "apf.core.step.DefaultConsumer"},
        "PRODUCER_CONFIG": {"CLASS": "apf.core.step.DefaultProducer"},
        "DB_CONFIG": {},
        "MODEL_CONFIG": {
            "CLASS": "tests.unit.stub_model.StubModel",
            "PARAMS": {"model_path": "https://example.org/1.0.0/model.zip"},
            "CLS_ID": CLS_ID,
        },
    }
    if forward_config is not None:
        config["SN_FORWARD_PRODUCER_CONFIG"] = forward_config
    if forward_class is not None:
        config["SN_FORWARD_CLASS"] = forward_class
    with mock.patch("rubin_stamp_classifier_step.step.PSQLConnection"), \
         mock.patch("rubin_stamp_classifier_step.step.get_taxonomy_by_classifier_id", return_value=TAXONOMY):
        return StampClassifierStep(config=config)


@pytest.fixture
def forwarding_step():
    return make_step(
        forward_config={
            "CLASS": "tests.unit.recording_producer.RecordingProducer",
            "TOPIC": FORWARD_TOPIC,
        }
    )


@pytest.fixture
def plain_step():
    return make_step()


@pytest.fixture(autouse=True)
def no_db_writes():
    with mock.patch("rubin_stamp_classifier_step.step.store_probability"):
        yield


def prediction(oid, dia_source_id, top_class, raw_alert):
    """An output message as execute emits it, with the raw alert attached."""
    probs = {name: 0.0 for name in StubModel.ROW}
    probs[top_class] = 1.0
    return {
        "oid": oid,
        "sid": 1,
        "diaObjectId": oid,
        "ssObjectId": 0,
        "diaSourceId": dia_source_id,
        "probabilities": probs,
        "midpointMjdTai": 60000.5,
        "ra": 10.0,
        "dec": -20.0,
        "alert": raw_alert,
    }


# --- the raw alert travels with the message from pre_execute to post_execute ---


def test_pre_execute_keeps_the_raw_alert_on_the_message(plain_step):
    raw = alert(123, None)

    out = plain_step.pre_execute([raw])

    assert out[0]["alert"] is raw


def test_execute_carries_the_raw_alert_onto_the_output(plain_step):
    raw = alert(123, None)
    message = {**processed(123, 1), "alert": raw}

    out = plain_step.execute([message])

    assert out[0]["alert"] is raw


def test_pre_produce_strips_the_raw_alert(plain_step):
    out = plain_step.pre_produce([prediction(123, 500, "SN", alert(123, None))])

    assert set(out[0]) == OUTPUT_SCHEMA_FIELDS


# --- post_execute forwards ranking-1 SN alerts, unchanged, and nothing else ---


def test_forwards_the_raw_alert_of_a_ranking_1_sn(forwarding_step):
    raw = alert(123, None)

    forwarding_step.post_execute([prediction(123, 500, "SN", raw)])

    assert forwarding_step.sn_forward_producer.produced == [(raw, {"key": "123"})]


def test_forwards_nothing_when_the_top_class_is_not_sn(forwarding_step):
    messages = [
        prediction(1, 501, "AGN", alert(1, None)),
        prediction(2, 502, "bogus", alert(2, None)),
        prediction(3, 503, "asteroid", alert(3, None)),
    ]

    forwarding_step.post_execute(messages)

    assert forwarding_step.sn_forward_producer.produced == []


def test_forwards_only_the_sn_rows_of_a_mixed_batch(forwarding_step):
    sn_raw = alert(2, None)
    messages = [
        prediction(1, 501, "VS", alert(1, None)),
        prediction(2, 502, "SN", sn_raw),
        prediction(3, 503, "AGN", alert(3, None)),
    ]

    forwarding_step.post_execute(messages)

    assert [m for m, _ in forwarding_step.sn_forward_producer.produced] == [sn_raw]


def test_forward_class_is_configurable():
    step = make_step(
        forward_config={
            "CLASS": "tests.unit.recording_producer.RecordingProducer",
            "TOPIC": FORWARD_TOPIC,
        },
        forward_class="AGN",
    )
    agn_raw = alert(1, None)

    step.post_execute([prediction(1, 501, "AGN", agn_raw), prediction(2, 502, "SN", alert(2, None))])

    assert [m for m, _ in step.sn_forward_producer.produced] == [agn_raw]


def test_post_execute_returns_the_messages_it_was_given(forwarding_step):
    messages = [prediction(123, 500, "SN", alert(123, None))]

    assert forwarding_step.post_execute(messages) is messages


# --- without the config block the step behaves as before ---


def test_no_forward_producer_when_unconfigured(plain_step):
    assert plain_step.sn_forward_producer is None


def test_post_execute_without_forwarder_still_works(plain_step):
    messages = [prediction(123, 500, "SN", alert(123, None))]

    assert plain_step.post_execute(messages) is messages


# --- apf drains every producer before committing; the forwarder must be one ---


def test_forward_producer_is_flushed_before_commit(forwarding_step):
    assert forwarding_step.sn_forward_producer in forwarding_step._get_producers()


# --- the deployed forwarder writes what the hunter deployment's consumer reads ---

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CHART_VALUES = os.path.join(ROOT_DIR, "charts", "rubin_stamp_classifier_step", "values.yaml")
# The recorded alerts are LSST v7.4; the schema version does not matter here,
# only that the forwarded bytes decode schemaless against the configured one.
LSST_SCHEMA = os.path.join(ROOT_DIR, "schemas", "surveys", "lsst", "v7_4_alert.avsc")
RECORDED_ALERT = os.path.join(
    os.path.dirname(__file__), "..", "integration", "data", "avro_messages", "message_0.avro"
)


def recorded_alert(schema):
    with open(RECORDED_ALERT, "rb") as f:
        raw = f.read()
    # Strip the 5-byte Confluent wire prefix, as LsstKafkaConsumer does.
    return fastavro.schemaless_reader(io.BytesIO(raw[5:]), schema)


def test_chart_forwarder_writes_schemaless_avro_the_hunter_consumer_reads():
    """The multisurvey steps consume with KafkaSchemalessConsumer and the
    schema under schemas/, so the forwarder in the chart must write bare
    schemaless records, not apf container files."""
    with open(CHART_VALUES) as f:
        chart_forwarder = yaml.safe_load(f)["configYaml"]["SN_FORWARD_PRODUCER_CONFIG"]
    schema = fastavro.schema.load_schema(LSST_SCHEMA)
    raw = recorded_alert(schema)

    with mock.patch("apf.producers.kafka.Producer") as kafka_producer:
        step = make_step(
            forward_config={
                "CLASS": chart_forwarder["CLASS"],
                "TOPIC": FORWARD_TOPIC,
                "PARAMS": {"bootstrap.servers": "localhost:9092"},
                "SCHEMA_PATH": LSST_SCHEMA,
            }
        )
        step.forward_sn_candidates([prediction(123, 500, "SN", raw)])

    (call,) = kafka_producer.return_value.produce.call_args_list
    forwarded = fastavro.schemaless_reader(io.BytesIO(call.kwargs["value"]), schema)
    assert same_alert(forwarded, raw)


def same_alert(a, b):
    """Equality where NaN == NaN, since LSST alerts carry NaN in many fields."""
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same_alert(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(map(same_alert, a, b))
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b
