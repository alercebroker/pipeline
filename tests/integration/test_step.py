from scripts.run_step import step_factory
from apf.consumers import KafkaConsumer


def test_step(kafka_service, env_variables, kafka_consumer):
    step = step_factory()
    step.start()

    for msg in kafka_consumer.consume():
        print(msg)
        kafka_consumer.commit()
