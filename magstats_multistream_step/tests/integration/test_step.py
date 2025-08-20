import json
from scripts.run_step import step_factory
from tests.unittests.data.messages import data as input_data


def assert_message_schema(command):
    assert "_id" in command["criteria"]
    assert "data" in command
    if command["collection"] == "magstats":
        assert command["type"] == "upsert"
    elif command["collection"] == "object":
        assert command["type"] == "update"
    else:
        assert False


def assert_command_data_schema(data):
    expected_fields = [
        "lastmjd",
        "meandec",
        "meanra",
        "sigmadec",
        "corrected",
        "firstmjd",
        "lastmjd",
        "deltajd",
        "magstats",
        "ndet",
    ]
    for field in expected_fields:
        assert field in data


def assert_magstats_is_list(data):
    magstats = data["magstats"]
    assert isinstance(magstats, list)
    assert len(magstats)


def assert_ndet(data, oid):
    dets_by_oid = {}
    for msg in input_data:
        detections = filter(lambda det: not det["forced"], msg["detections"])
        for det in detections:
            dets_by_oid[det["oid"]] = dets_by_oid.get(det["oid"], []) + [det]
    assert len(dets_by_oid[oid]) == data["ndet"]


def test_step(kafka_service, env_variables, kafka_consumer):
    step = step_factory()
    step.start()
    messages = list(kafka_consumer.consume())
    for msg in messages:
        loaded_message = json.loads(msg["payload"])
        assert_message_schema(loaded_message)
        assert_command_data_schema(loaded_message["data"])
        assert_magstats_is_list(loaded_message["data"])
        assert_ndet(loaded_message["data"], loaded_message["criteria"]["_id"])
