from unittest.mock import MagicMock
import json

from apf.producers import KafkaProducer


def test_produce_scribe(env_variables):
    from scripts.run_step import step_creator

    step = step_creator()

    non_detection = {
        "aid": "aid",
        "tid": "tid",
        "oid": "oid",
        "mjd": 1.5,
        "fid": 1,
        "diffmaglim": 1,
    }
    non_detections = [non_detection]
    step.scribe_producer = MagicMock(KafkaProducer)
    step.produce_scribe(non_detections)
    expected_data = {
        "collection": "non_detection",
        "type": "update",
        "criteria": {
            "oid": non_detection["oid"],
            "fid": non_detection["fid"],
            "mjd": non_detection["mjd"],
        },
        "data": non_detection,
        "options": {"upsert": True},
    }
    step.scribe_producer.produce.assert_called_once_with(
        {"payload": json.dumps(expected_data)}
    )


def test_post_execute(env_variables):
    from scripts.run_step import step_creator

    step = step_creator()

    non_detection = {
        "aid": "aid",
        "tid": "tid",
        "oid": "oid",
        "mjd": 1.5,
        "fid": 1,
        "diffmaglim": 1,
    }
    non_detections = [non_detection]
    step.scribe_producer = MagicMock(KafkaProducer)
    step.post_execute([{"aid": "aid", "non_detections": non_detections}])
    expected_data = {
        "collection": "non_detection",
        "type": "update",
        "criteria": {
            "oid": non_detection["oid"],
            "fid": non_detection["fid"],
            "mjd": non_detection["mjd"],
        },
        "data": non_detection,
        "options": {"upsert": True},
    }
    step.scribe_producer.produce.assert_called_once_with(
        {"payload": json.dumps(expected_data)}
    )
