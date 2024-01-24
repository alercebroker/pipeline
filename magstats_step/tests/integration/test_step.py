import json
from scripts.run_step import step_factory


def assert_message_schema(command):
    if command["collection"] == "magstats":
        assert command["type"] == "upsert"
        assert "_id" in command["criteria"]
    elif command["collection"] == "object":
        assert command["type"] == "update"
        assert "_id" in command["criteria"]
    else:
        assert False
    assert "data" in command


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
    ]
    for field in expected_fields:
        assert field in data

def assert_magstats_is_list(data):
    magstats = data["magstats"]
    assert isinstance(magstats, list)
    assert len(magstats)

def test_step(kafka_service, env_variables, kafka_consumer):
    step = step_factory()
    step.start()
    messages = list(kafka_consumer.consume())
    for msg in messages:
        loaded_message = json.loads(msg["payload"])
        assert_message_schema(loaded_message)
        assert_command_data_schema(loaded_message["data"])
        assert_magstats_is_list(loaded_message["data"])
