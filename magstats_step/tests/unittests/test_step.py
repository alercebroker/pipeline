import json
from unittest import mock

from .data.messages import data
from scripts.run_step import step_factory


def test_execute_multistream(env_variables):
    step = step_factory()
    formatted_data = step.pre_execute(data)
    result = step.execute(formatted_data)
    dcopy = []
    for d in data:
        non_forced = list(
            filter(
                lambda det: not det["forced"],
                d["detections"],
            )
        )
        d["detections"] = non_forced
        if len(d["detections"]) > 0:
            dcopy.append(d)

    assert len(result) == len({d["oid"] for d in dcopy})
    for d in dcopy:
        assert d["oid"] in result
        assert "meanra" in result[d["oid"]]
        assert "meandec" in result[d["oid"]]
        assert "magstats" in result[d["oid"]]
        assert "oid" in result[d["oid"]]
        assert "tid" in result[d["oid"]]
        assert "firstmjd" in result[d["oid"]]
        assert "lastmjd" in result[d["oid"]]
        assert "ndet" in result[d["oid"]]
        assert "sigmara" in result[d["oid"]]
        assert "sigmadec" in result[d["oid"]]


def test_scribe_message_multistream(env_variables):
    step = step_factory()
    formatted_data = step.pre_execute(data)
    result = step.execute(formatted_data)
    step.scribe_producer = mock.MagicMock()
    step.post_execute(result)
    dcopy = []
    for d in data:
        non_forced = list(
            filter(
                lambda det: not det["forced"],
                d["detections"],
            )
        )
        d["detections"] = non_forced
        if len(d["detections"]) > 0:
            dcopy.append(d)
    for i, d in enumerate(dcopy):
        oid_calls = list(
            filter(
                lambda call: call.kwargs["key"] == d["oid"],
                step.scribe_producer.produce.call_args_list,
            )
        )
        assert len(oid_calls) == 2
        parsed_commands = list(
            map(lambda call: json.loads(call[0][0]["payload"]), oid_calls)
        )
        assert len(parsed_commands) == 2
        assert parsed_commands[0]["collection"] == "object"
        assert parsed_commands[0]["type"] == "update"
        assert parsed_commands[1]["collection"] == "magstats"
        assert parsed_commands[1]["type"] == "upsert"
        assert parsed_commands[0]["criteria"]["_id"] == d["oid"]
        assert parsed_commands[1]["criteria"]["_id"] == d["oid"]
