import copy
import json
import pickle
from copy import deepcopy
from unittest import mock

from correction._step import CorrectionStep

from tests.utils import ztf_alert, atlas_alert, non_detection, elasticc_alert

messages = [
    {
        "oid": "OID1",
        "candid": "a",
        "detections": [ztf_alert(candid="a", new=True), ztf_alert(candid="b", has_stamp=False, new=False, forced=True)],
        "non_detections": [],
    },
    {
        "oid": "OID2",
        "candid": "c",
        "detections": [
            ztf_alert(oid="OID2", candid="c", new=True),
            ztf_alert(oid="OID2", candid="d", has_stamp=False, new=True),
        ],
        "non_detections": [non_detection(oid="OID2", mjd=1, fid=1)],
    },
    {"oid": "OID3", "candid": "e", "detections": [atlas_alert(oid="OID3", candid="e", new=True)], "non_detections": []},
    {
        "oid": "OID4",
        "candid": "hehe",
        "detections": [elasticc_alert(oid="OID4", candid="hehe", new=True)],
        "non_detections": [],
    },
]

message4produce = [
    {
        "oid": "OID1",
        "candid": "a",
        "meanra": 1,
        "meandec": 1,
        "detections": [ztf_alert(candid="a", new=True), ztf_alert(candid="b", has_stamp=False, forced=True, new=False)],
        "non_detections": [],
    },
    {
        "oid": "OID2",
        "candid": "c",
        "meanra": 1,
        "meandec": 1,
        "detections": [
            ztf_alert(oid="OID2", candid="c", new=True),
            ztf_alert(oid="OID2", candid="d", has_stamp=False, new=True),
        ],
        "non_detections": [non_detection(oid="OID2", mjd=1, fid=1)],
    },
    {
        "oid": "OID3",
        "candid": "e",
        "meanra": 1,
        "meandec": 1,
        "detections": [atlas_alert(oid="OID3", candid="e", new=True)],
        "non_detections": [],
    },
    {
        "oid": "OID4",
        "candid": "hehe",
        "meanra": 1,
        "meandec": 1,
        "detections": [elasticc_alert(oid="OID4", candid="hehe", new=True)],
        "non_detections": [],
    },
]

message4execute = {
    "candids": {"OID1": "a", "OID2": "c", "OID3": "e", "OID4": "hehe"},
    "detections": [
        ztf_alert(oid="OID1", candid="a", new=True),
        ztf_alert(oid="OID1", candid="b", has_stamp=False, new=False, forced=True),
        ztf_alert(oid="OID2", candid="c", new=True),
        ztf_alert(oid="OID2", candid="d", has_stamp=False, new=True),
        atlas_alert(oid="OID3", candid="e", new=True),
        elasticc_alert(oid="OID4", candid="hehe", new=True),
    ],
    "non_detections": [
        non_detection(oid="OID2", mjd=1, fid=1),
    ],
    "coords": {
        "OID1": {"meanra": 1, "meandec": 1},
        "OID2": {"meanra": 1, "meandec": 1},
        "OID3": {"meanra": 1, "meandec": 1},
        "OID4": {"meanra": 1, "meandec": 1},
    },
}


def test_pre_execute_formats_message_with_all_detections_and_non_detections():
    formatted = CorrectionStep.pre_execute(messages)
    assert "detections" in formatted
    assert formatted["detections"] == message4execute["detections"]
    assert "non_detections" in formatted
    assert formatted["non_detections"] == message4execute["non_detections"]


@mock.patch("correction._step.step.Corrector")
def test_execute_calls_corrector_for_detection_records_and_keeps_non_detections(mock_corrector):
    formatted = CorrectionStep.execute(message4execute)
    assert "detections" in formatted
    assert "non_detections" in formatted
    assert formatted["non_detections"] == message4execute["non_detections"]
    mock_corrector.assert_called_with(message4execute["detections"])
    mock_corrector.return_value.corrected_as_records.assert_called_once()


@mock.patch("correction._step.step.Corrector")
def test_execute_removes_duplicate_non_detections(_):
    message4execute_copy = deepcopy(message4execute)
    message4execute_copy["non_detections"] = (
        message4execute_copy["non_detections"] + message4execute_copy["non_detections"]
    )
    formatted = CorrectionStep.execute(message4execute_copy)
    assert "non_detections" in formatted
    assert formatted["non_detections"] == message4execute["non_detections"]


@mock.patch("correction._step.step.Corrector")
def test_execute_works_with_empty_non_detections(_):
    message4execute_copy = deepcopy(message4execute)
    message4execute_copy["non_detections"] = []
    formatted = CorrectionStep.execute(message4execute_copy)
    assert "non_detections" in formatted
    assert formatted["non_detections"] == []


def test_post_execute_calls_scribe_producer_for_each_detection():
    # To check the "new" flag is removed
    message4execute_copy = copy.deepcopy(message4execute)
    message4execute_copy["detections"] = [{k: v for k, v in det.items()} for det in message4execute_copy["detections"]]

    class MockCorrectionStep(CorrectionStep):
        def __init__(self):
            self.scribe_producer = mock.MagicMock()
            self.logger = mock.MagicMock()

    step = MockCorrectionStep()
    output = step.post_execute(copy.deepcopy(message4execute))
    assert output == message4execute_copy
    count = 0
    for det in message4execute_copy["detections"]:
        count += 1
        flush = False
        if not det["new"]:  # does not write
            continue
        det["extra_fields"] = {
            k: v for k, v in det["extra_fields"].items() if k not in ["prvDiaSources", "prvDiaForcedSources"]
        }

        if "diaObject" in det["extra_fields"]:
            det["extra_fields"]["diaObject"] = pickle.loads(det["extra_fields"]["diaObject"])

        data = {
            "collection": "detection" if not det["forced"] else "forced_photometry",
            "type": "update",
            "criteria": {"_id": det["candid"]},
            "data": {k: v for k, v in det.items() if k not in ["candid", "forced", "new"]},
            "options": {"upsert": True, "set_on_insert": not det["has_stamp"]},
        }
        if count == len(message4execute_copy["detections"]):
            flush = True
        step.scribe_producer.produce.assert_any_call({"payload": json.dumps(data)}, flush=flush)


def test_pre_produce_unpacks_detections_and_non_detections_by_oid():
    # Input with the "new" flag is removed
    message4execute_copy = copy.deepcopy(message4execute)
    message4execute_copy["detections"] = [{k: v for k, v in det.items()} for det in message4execute_copy["detections"]]

    formatted = CorrectionStep.pre_produce(message4execute_copy)
    assert formatted == message4produce
