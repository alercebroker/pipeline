import json
from copy import deepcopy
from unittest import mock

from correction_step.step import CorrectionStep

from tests.utils import ztf_alert, atlas_alert, non_detection

messages = [
    {
        "aid": "AID1",
        "new_alert": ztf_alert(candid="a"),
        "prv_detections": [ztf_alert(candid="b")],
        "non_detections": []
    },
    {
        "aid": "AID2",
        "new_alert": ztf_alert(aid="AID2", candid="c"),
        "prv_detections": [ztf_alert(aid="AID2", candid="d")],
        "non_detections": [non_detection(aid="AID2", mjd=1, oid="oid1", fid=1)]
    },
    {
        "aid": "AID3",
        "new_alert": atlas_alert(aid="AID3", candid="e"),
        "prv_detections": [],
        "non_detections": []
    },
]

message4produce = [
    {
        "aid": "AID1",
        "detections": [ztf_alert(candid="a"), ztf_alert(candid="b", has_stamp=False)],
        "non_detections": []
    },
    {
        "aid": "AID2",
        "detections": [ztf_alert(aid="AID2", candid="c"), ztf_alert(aid="AID2", candid="d", has_stamp=False)],
        "non_detections": [non_detection(aid="AID2", mjd=1, oid="oid1", fid=1)]
    },
    {
        "aid": "AID3",
        "detections": [atlas_alert(aid="AID3", candid="e")],
        "non_detections": []
    },
]

message4execute = {
    "detections": [
        ztf_alert(aid="AID1", candid="a"),
        ztf_alert(aid="AID1", candid="b", has_stamp=False),
        ztf_alert(aid="AID2", candid="c"),
        ztf_alert(aid="AID2", candid="d", has_stamp=False),
        atlas_alert(aid="AID3", candid="e"),
    ],
    "non_detections": [
        non_detection(aid="AID2", mjd=1, oid="oid1", fid=1),
    ]
}


def test_pre_execute_formats_message_with_all_detections_and_non_detections():
    formatted = CorrectionStep.pre_execute(messages)
    assert "detections" in formatted
    assert formatted["detections"] == message4execute["detections"]
    assert "non_detections" in formatted
    assert formatted["non_detections"] == message4execute["non_detections"]


@mock.patch("correction_step.step.Corrector")
def test_execute_calls_corrector_for_detection_records_and_keeps_non_detections(mock_corrector):
    formatted = CorrectionStep.execute(message4execute)
    assert "detections" in formatted
    assert "non_detections" in formatted
    assert formatted["non_detections"] == message4execute["non_detections"]
    mock_corrector.assert_called_with(message4execute["detections"])
    mock_corrector.return_value.corrected_records.assert_called_once()


@mock.patch("correction_step.step.Corrector")
def test_execute_removes_duplicate_non_detections(mock_corrector):
    message4execute_copy = deepcopy(message4execute)
    message4execute_copy["non_detections"] = message4execute_copy["non_detections"] + message4execute_copy["non_detections"]
    formatted = CorrectionStep.execute(message4execute_copy)
    assert "non_detections" in formatted
    assert formatted["non_detections"] == message4execute["non_detections"]


@mock.patch("correction_step.step.Corrector")
def test_execute_works_with_empty_non_detections(mock_corrector):
    message4execute_copy = deepcopy(message4execute)
    message4execute_copy["non_detections"] = []
    formatted = CorrectionStep.execute(message4execute_copy)
    assert "non_detections" in formatted
    assert formatted["non_detections"] == []


def test_post_execute_calls_scribe_producer_for_each_detection():
    class MockCorrectionStep(CorrectionStep):
        def __init__(self):
            self.scribe_producer = mock.MagicMock()

    step = MockCorrectionStep()
    output = step.post_execute(message4execute)
    assert output == message4execute
    data1 = {
        "collection": "detection",
        "type": "update",
        "criteria": {"_id": "a"},
        "data": {k: v for k, v in ztf_alert().items() if k != "candid"},
        "options": {"upsert": True, "set_on_insert": False},
    }
    step.scribe_producer.produce.assert_any_call({"payload": json.dumps(data1)})
    data2 = {
        "collection": "detection",
        "type": "update",
        "criteria": {"_id": "b"},
        "data": {k: v for k, v in ztf_alert(has_stamp=False).items() if k != "candid"},
        "options": {"upsert": True, "set_on_insert": True},
    }
    step.scribe_producer.produce.assert_any_call({"payload": json.dumps(data2)})


def test_pre_produce_unpacks_detections_and_non_detections_by_aid():
    formatted = CorrectionStep.pre_produce(message4execute)
    assert formatted == message4produce
