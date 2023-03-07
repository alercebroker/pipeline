from apf.consumers import KafkaConsumer
from prv_candidates_step.step import PrvCandidatesStep


def test_step_initialization(kafka_service, env_variables):
    from scripts.run_step import step

    assert isinstance(step, PrvCandidatesStep)


def test_result_has_everything(
    kafka_service, env_variables, kafka_consumer: KafkaConsumer
):
    from scripts.run_step import step

    step.start()
    for message in kafka_consumer.consume():
        assert_result_has_non_detections(message)
        assert_result_has_prv_detections(message)
        assert_result_has_alert(message)
        kafka_consumer.commit()


def assert_result_has_prv_detections(message):
    assert message["prv_detections"] is not None
    if message["new_alert"]["tid"].lower() == "atlas":
        assert len(message["prv_detections"]) == 0
    else:
        assert len(message["prv_detections"]) == 2


def assert_result_has_non_detections(message):
    assert message["non_detections"] is not None
    if message["new_alert"]["tid"].lower() == "atlas":
        assert len(message["non_detections"]) == 0
    else:
        assert len(message["non_detections"]) == 2


def assert_result_has_alert(message):
    assert message["new_alert"] is not None


def test_scribe_has_non_detections():
    pass


def test_works_with_ztf_messages():
    pass


def test_works_with_atlas_message():
    pass
