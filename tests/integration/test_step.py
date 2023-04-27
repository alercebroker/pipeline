import json
from lc_classification.step import LateClassifier
from apf.consumers import KafkaConsumer

def assert_object_is_correct(obj):
    assert "aid" in obj
    assert "candid" in obj
    assert "features" in obj
    assert "lc_classification" in obj

def assert_command_is_correct(command):
    assert command["collection"] == "object"
    assert command["type"] == "update_probabilities"
    assert command["criteria"]["_id"] is not None
    assert "aid" not in command["data"]
    assert not command["options"]["set_on_insert"]

def test_step_result(kafka_service, env_variables, kafka_consumer: KafkaConsumer,):
    from settings import STEP_CONFIG
    step = LateClassifier(config=STEP_CONFIG)
    step.start()

    for message in kafka_consumer.consume():
        assert_object_is_correct(message)
        kafka_consumer.commit()

def test_scribe_result(kafka_service, env_variables, scribe_consumer: KafkaConsumer):
    from settings import STEP_CONFIG
    step = LateClassifier(config=STEP_CONFIG)
    step.start()

    for message in scribe_consumer.consume():
        command = json.loads(message["payload"])
        assert_command_is_correct(command)
        scribe_consumer.commit()
