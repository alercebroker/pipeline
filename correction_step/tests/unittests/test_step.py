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
        "detections": [
            ztf_alert(candid="a", new=True, forced=False),
            ztf_alert(candid="b", has_stamp=False, new=False, forced=True),
        ],
        "non_detections": [],
    },
    {
        "oid": "OID2",
        "candid": "c",
        "detections": [
            ztf_alert(oid="OID2", candid="c", new=True, forced=False),
            ztf_alert(oid="OID2", candid="d", has_stamp=False, new=True, forced=False),
        ],
        "non_detections": [non_detection(oid="OID2", mjd=1, fid=1)],
    },
    {
        "oid": "OID3",
        "candid": "e",
        "detections": [atlas_alert(oid="OID3", candid="e", new=True)],
        "non_detections": [],
    },
    {
        "oid": "OID4",
        "candid": "hehe",
        "detections": [elasticc_alert(oid="OID4", candid="hehe", new=True, forced=False)],
        "non_detections": [],
    },
]

message4produce = [
    {
        "oid": "OID1",
        "candid": "a",
        "meanra": 1,
        "meandec": 1,
        "detections": [
            ztf_alert(
                candid="a",
                new=True,
                forced=False,
                extra_fields=messages[0]["detections"][0]["extra_fields"],
            ),
            ztf_alert(
                candid="b",
                has_stamp=False,
                forced=True,
                new=False,
                extra_fields=messages[0]["detections"][1]["extra_fields"],
            ),
        ],
        "non_detections": [],
    },
    {
        "oid": "OID2",
        "candid": "c",
        "meanra": 1,
        "meandec": 1,
        "detections": [
            ztf_alert(
                oid="OID2",
                candid="c",
                new=True,
                forced=False,
                extra_fields=messages[1]["detections"][0]["extra_fields"],
            ),
            ztf_alert(
                oid="OID2",
                candid="d",
                has_stamp=False,
                new=True,
                forced=False,
                extra_fields=messages[1]["detections"][1]["extra_fields"],
            ),
        ],
        "non_detections": [non_detection(oid="OID2", mjd=1, fid=1)],
    },
    {
        "oid": "OID3",
        "candid": "e",
        "meanra": 1,
        "meandec": 1,
        "detections": [
            atlas_alert(
                oid="OID3",
                candid="e",
                new=True,
                extra_fields=messages[2]["detections"][0]["extra_fields"],
            )
        ],
        "non_detections": [],
    },
    {
        "oid": "OID4",
        "candid": "hehe",
        "meanra": 1,
        "meandec": 1,
        "detections": [
            elasticc_alert(
                oid="OID4",
                candid="hehe",
                new=True,
                forced=False,
                extra_fields=messages[3]["detections"][0]["extra_fields"],
            )
        ],
        "non_detections": [],
    },
]

message4execute = {
    "candids": {"OID1": "a", "OID2": "c", "OID3": "e", "OID4": "hehe"},
    "detections": [
        ztf_alert(
            oid="OID1",
            candid="a",
            new=True,
            forced=False,
            extra_fields=messages[0]["detections"][0]["extra_fields"],
        ),
        ztf_alert(
            oid="OID1",
            candid="b",
            has_stamp=False,
            new=False,
            forced=True,
            extra_fields=messages[0]["detections"][1]["extra_fields"],
        ),
        ztf_alert(
            oid="OID2",
            candid="c",
            new=True,
            forced=False,
            extra_fields=messages[1]["detections"][0]["extra_fields"],
        ),
        ztf_alert(
            oid="OID2",
            candid="d",
            has_stamp=False,
            new=True,
            forced=False,
            extra_fields=messages[1]["detections"][1]["extra_fields"],
        ),
        atlas_alert(
            oid="OID3",
            candid="e",
            new=True,
            extra_fields=messages[2]["detections"][0]["extra_fields"],
        ),
        elasticc_alert(
            oid="OID4",
            candid="hehe",
            new=True,
            forced=False,
            extra_fields=messages[3]["detections"][0]["extra_fields"],
        ),
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
    mock_corrector.assert_any_call(message4execute["detections"])
    mock_corrector.return_value.corrected_as_records.assert_called()


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

    class MockCorrectionStep(CorrectionStep):
        def __init__(self):
            self.scribe_producer = mock.MagicMock()
            self.logger = mock.MagicMock()

    step = MockCorrectionStep()
    execute_output = step.execute(message4execute_copy)
    output = step.post_execute(execute_output)
    assert output == execute_output
    # verify that there are new detections
    assert len(list(filter(lambda x: x["new"], output["detections"]))) > 0
    count = 0
    for det in execute_output["detections"]:
        count += 1
        flush = False
        if not det["new"]:  # does not write
            continue
        if not det["tid"] == "LSST":  # dont know why
            continue
        det["extra_fields"] = {
            k: v
            for k, v in det["extra_fields"].items()
            if k not in ["prvDiaSources", "prvDiaForcedSources"]
        }
        if "diaObject" in det["extra_fields"]:
            det["extra_fields"]["diaObject"] = pickle.loads(det["extra_fields"]["diaObject"])
        data = {
            "collection": "detection" if not det["forced"] else "forced_photometry",
            "type": "update",
            "criteria": {"candid": det["candid"], "oid": det["oid"]},
            "data": {k: v for k, v in det.items() if k not in ["candid", "forced", "new"]},
            "options": {"upsert": True, "set_on_insert": not det["has_stamp"]},
        }
        if count == len(execute_output["detections"]):
            flush = True
        step.scribe_producer.produce.assert_has_calls

        mock_args, _ = step.scribe_producer.produce.call_args
        from unittest import TestCase
        tc = TestCase()
        tc.maxDiff = None
        tc.assertDictEqual(mock_args[0], {"payload": json.dumps(data)})
        
    assert step.scribe_producer.produce.call_count == len(
        list(filter(lambda x: x["new"], message4execute_copy["detections"]))
    )


def test_pre_produce_unpacks_detections_and_non_detections_by_oid():
    # Input with the "new" flag is removed
    message4execute_copy = copy.deepcopy(message4execute)
    message4execute_copy["detections"] = [
        {k: v for k, v in det.items()} for det in message4execute_copy["detections"]
    ]

    formatted = CorrectionStep.pre_produce(message4execute_copy)
    assert formatted == message4produce
