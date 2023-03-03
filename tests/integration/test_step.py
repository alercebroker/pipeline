from apf.consumers import KafkaConsumer
from prv_candidates_step.step import PrvCandidatesStep


def test_step_initialization(kafka_service, env_variables):
    from scripts.run_step import step

    assert isinstance(step, PrvCandidatesStep)


def test_step_start(kafka_service, env_variables):
    from scripts.run_step import step

    step.start()

    consumer = KafkaConsumer(
        {
            "PARAMS": {
                "bootstrap.servers": "localhost:9092",
                "group.id": "test_step_start",
                "auto.offset.reset": "beginning",
                "enable.partition.eof": True,
            },
            "TOPICS": ["prv-candidates"],
        }
    )

    for message in consumer.consume():
        assert message["new_alert"] is not None
        assert len(message["prv_detections"]) == 2
        assert len(message["non_detections"]) == 2
