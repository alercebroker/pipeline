from apf.consumers import KafkaConsumer
from prv_candidates_step.step import PrvCandidatesStep
import json


def test_step_initialization(kafka_service, env_variables):
    from scripts.run_step import step_creator

    assert isinstance(step_creator(), PrvCandidatesStep)


def test_result_has_everything(
    kafka_service, env_variables, kafka_consumer: KafkaConsumer
):
    from scripts.run_step import step_creator

    step_creator().start()
    count = 0
    for message in kafka_consumer.consume():
        count += 1
        assert_result_has_non_detections(message)
        assert_result_has_prv_detections(message)
        assert_result_has_alert(message)
        kafka_consumer.commit()
    assert count == 10


def assert_result_has_prv_detections(message):
    assert message["detections"] is not None
    if message["detections"][0]["sid"].lower() == "atlas":
        assert len(message["detections"]) == 1
    else:
        assert len(message["detections"]) == 4


def assert_result_has_non_detections(message):
    assert message["non_detections"] is not None
    if message["detections"][0]["sid"].lower() == "atlas":
        assert len(message["non_detections"]) == 0
    else:
        assert len(message["non_detections"]) == 2


def assert_result_has_alert(message):
    assert message["detections"][0] is not None
    assert message["detections"][0].get("stamps") is None
    assert message["detections"][0]["extra_fields"].get("prv_candidates") is None


def test_scribe_has_non_detections(kafka_service, env_variables, scribe_consumer):
    from scripts.run_step import step_creator

    step = step_creator()
    step.start()

    for message in scribe_consumer.consume():
        assert_scribe_has_non_detections(message)
        scribe_consumer.commit()


def assert_scribe_has_non_detections(message):
    data = json.loads(message["payload"])
    assert data["collection"] == "non_detection"
    assert data["type"] == "update"
    assert "fid" in data["criteria"]
    assert "mjd" in data["criteria"]
    assert len(data["data"]) > 0


def test_works_with_batch(kafka_service, env_variables, kafka_consumer: KafkaConsumer):
    from scripts.run_step import step_creator
    import os

    os.environ["CONSUME_MESSAGES"] = "10"
    step_creator().start()

    for message in kafka_consumer.consume():
        assert_result_has_non_detections(message)
        assert_result_has_prv_detections(message)
        assert_result_has_alert(message)
        kafka_consumer.commit()


def test_works_with_schemaless_producer(kafka_service, env_variables):
    from scripts.run_step import step_creator
    import os

    os.environ["PRODUCER_CLASS"] = "apf.producers.KafkaSchemalessProducer"

    step_creator().start()
