import copy
import json
import pickle
from copy import deepcopy
from unittest import mock

from correction._step import CorrectionStep

from tests.utils import ztf_detection, atlas_alert, non_detection, elasticc_alert
from typing import Dict, Any


messages = [
    {
        "oid": "OID1",
        "candid": "a",
        "detections": [
            ztf_detection(is_new=True, candid="a", forced=False),
            ztf_detection(candid="b", has_stamp=False, is_new=False, forced=True),
        ],
        "non_detections": [],
    },
    {
        "oid": "OID2",
        "candid": "c",
        "detections": [
            ztf_detection(oid="OID2", candid="c", is_new=True, forced=False),
            ztf_detection(oid="OID2", candid="d", has_stamp=False, is_new=True, forced=False),
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
        "detections": [elasticc_alert(oid="OID4", candid="hehe", is_new=True, forced=False)],
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
            ztf_detection(
                candid="a",
                is_new=True,
                forced=False,
                extra_fields=messages[0]["detections"][0]["extra_fields"],
            ),
            ztf_detection(
                candid="b",
                has_stamp=False,
                forced=True,
                is_new=False,
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
            ztf_detection(
                oid="OID2",
                candid="c",
                is_new=True,
                forced=False,
                extra_fields=messages[1]["detections"][0]["extra_fields"],
            ),
            ztf_detection(
                oid="OID2",
                candid="d",
                has_stamp=False,
                is_new=True,
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
                is_new=True,
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
        ztf_detection(
            oid="OID1",
            candid="a",
            is_new=True,
            forced=False,
            extra_fields=messages[0]["detections"][0]["extra_fields"],
        ),
        ztf_detection(
            oid="OID1",
            candid="b",
            has_stamp=False,
            is_new=False,
            forced=True,
            extra_fields=messages[0]["detections"][1]["extra_fields"],
        ),
        ztf_detection(
            oid="OID2",
            candid="c",
            is_new=True,
            forced=False,
            extra_fields=messages[1]["detections"][0]["extra_fields"],
        ),
        ztf_detection(
            oid="OID2",
            candid="d",
            has_stamp=False,
            is_new=True,
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
            is_new=True,
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


def add_corr_mags_to_message(message: Dict[str, Any]):
    for detection in message["detections"]:
        if detection["new"]:
            continue
        else:
            detection["extra_fields"]["mag_corr"] = 15.2
            detection["extra_fields"]["e_mag_corr"] = 0.02
            detection["extra_fields"]["e_mag_corr_ext"] = 0.08


add_corr_mags_to_message(message4execute)


def test_pre_execute_formats_message_with_all_detections_and_non_detections():
    for m in messages:
        add_corr_mags_to_message(m)

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


class FakeScribeProducer:
    """librdkafka-like buffer: ``produce`` queues, ``flush`` drains to ``delivered``."""

    def __init__(self):
        self.buffered, self.delivered = [], []
        # the fixed path reaches the underlying client via self.scribe_producer.producer.flush()
        self.producer = self

    def produce(self, message, flush=False, key=None):
        self.buffered.append(message)
        if flush:
            self.flush()

    def flush(self, *args, **kwargs):
        self.delivered.extend(self.buffered)
        self.buffered = []


class FakeConsumer:
    """Snapshots what the scribe producer still has buffered at offset-commit time."""

    def __init__(self, scribe):
        self.scribe = scribe
        self.commit_called = False
        self.buffered_at_commit = None

    def commit(self):
        self.commit_called = True
        self.buffered_at_commit = list(self.scribe.buffered)


def test_scribe_is_flushed_before_offset_commit_when_last_detection_is_not_new():
    """Regression for the scribe flush/commit ordering bug (correction-step-flush-fix).

    Reproduces the silent-loss window deterministically: a batch whose LAST detection is
    ``new=False`` skips the per-message ``flush=True`` in the unfixed code, so buffered
    scribe messages are still undelivered when GenericStep._post_produce commits the Kafka
    offset. If the pod dies in that window the detections are lost. The invariant is that
    nothing is buffered at commit time.
    """

    class MockCorrectionStep(CorrectionStep):
        def __init__(self):
            self.scribe_producer = FakeScribeProducer()
            self.consumer = FakeConsumer(self.scribe_producer)
            self.logger = mock.MagicMock()
            self.metrics = mock.MagicMock()
            self.commit = True

    step = MockCorrectionStep()
    # Batch crafted so the last detection is new=False (skipped, never triggers a flush).
    result = {
        "detections": [
            {
                "new": True,
                "candid": "a",
                "oid": "OID1",
                "forced": False,
                "has_stamp": True,
                "extra_fields": {},
            },
            {
                "new": False,
                "candid": "b",
                "oid": "OID1",
                "forced": False,
                "has_stamp": True,
                "extra_fields": {},
            },
        ]
    }

    step.post_execute(result)  # real produce_scribe: buffers "a", skips "b"
    step._post_produce()  # real framework path: consumer.commit()

    assert step.consumer.commit_called
    # The invariant: no scribe message may still be buffered once the offset is committed.
    assert step.consumer.buffered_at_commit == []
    # And every produced message must actually have been delivered.
    assert len(step.scribe_producer.delivered) == 1
    assert step.scribe_producer.buffered == []
