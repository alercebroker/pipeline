import json
from unittest import mock

from .data.messages import data
from scripts.run_step import step_factory


def test_execute_multistream(env_variables):
    step = step_factory()
    formatted_data = step.pre_execute(data)
    result = step.execute(formatted_data)
    result = result["multistream"]
    dcopy = filter(
        lambda d: all(
            [
                det["sid"] == "ZTF" and not det["forced"]
                for det in d["detections"]
            ]
        ),
        data,
    )
    for d in dcopy:
        if not all([det["sid"] == "ZTF" for det in d["detections"]]):
            continue
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
    result = result["multistream"]
    dcopy = filter(
        lambda d: all(
            [
                det["sid"] == "ZTF" and not det["forced"]
                for det in d["detections"]
            ]
        ),
        data,
    )
    for i, d in enumerate(dcopy):
        to_write = result[d["oid"]]
        to_write.update(
            {
                "loc": {
                    "type": "Point",
                    "coordinates": [
                        to_write["meanra"] - 180,
                        to_write["meandec"],
                    ],
                }
            }
        )
        command = {
            "collection": "object",
            "type": "update",
            "criteria": {"_id": d["oid"]},
            "data": to_write,
            "options": {"upsert": True},
        }
        assert any(
            call[0][0]["payload"] == json.dumps(command)
            and call.kwargs["key"] == d["oid"]
            for call in step.scribe_producer.produce.call_args_list
        )


def test_execute_ztf(env_variables):
    step = step_factory()
    formatted_data = step.pre_execute(data)
    result = step.execute(formatted_data)
    result = result["ztf"]
    dcopy = filter(
        lambda d: all(
            [
                det["sid"] == "ZTF" and not det["forced"]
                for det in d["detections"]
            ]
        ),
        data,
    )
    for d in dcopy:
        if not any([det["sid"] == "ZTF" for det in d["detections"]]):
            continue
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
