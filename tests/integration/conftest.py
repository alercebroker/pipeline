import os
import pathlib
import logging
import random
import uuid
from datetime import datetime
import pytest
from apf.producers import KafkaProducer
from apf.consumers import KafkaConsumer
from confluent_kafka.admin import AdminClient, NewTopic
from fastavro.utils import generate_many

from tests.mockdata.inputschema import INPUT_SCHEMA as SCHEMA


@pytest.fixture(scope="session")
def docker_compose_file(pytestconfig):
    return (
        pathlib.Path(pytestconfig.rootdir) / "tests/integration/docker-compose.yml"
    ).absolute()

def get_lc_classifier_topic():
    return "lc_classifier%s" % datetime.utcnow().strftime("%Y%m%d")

def is_responsive_kafka(url):
    client = AdminClient({"bootstrap.servers": url})
    futures = client.create_topics(
        [
            NewTopic("features", num_partitions=1),
            NewTopic("w_object", num_partitions=1),
            NewTopic(get_lc_classifier_topic(), num_partitions=1),
            NewTopic("metrics", num_partitions=1),
        ]
    )
    for topic, future in futures.items():
        try:
            future.result()
            if topic == "features":
                produce_messages("features")
        except Exception as e:
            logging.error(f"Can't create topic {topic}: {e}")
            return False
    return True


@pytest.fixture(scope="session")
def kafka_service(docker_ip, docker_services):
    """Ensure that Kafka service is up and responsive."""
    port = docker_services.port_for("kafka", 9092)
    server = "{}:{}".format(docker_ip, port)
    docker_services.wait_until_responsive(
        timeout=30.0, pause=0.1, check=lambda: is_responsive_kafka(server)
    )
    return server


@pytest.fixture
def env_variables():
    random_string = uuid.uuid4().hex
    env_variables_dict = {
        "CONSUMER_SERVER": "localhost:9092",
        "CONSUMER_TOPICS": "features",
        "CONSUMER_GROUP_ID": random_string,
        "PRODUCER_SERVER": "localhost:9092",
        "PRODUCER_TOPIC_FORMAT": "lc_classifier%s",
        "PRODUCER_DATE_FORMAT": "%Y%m%d",
        "PRODUCER_CHANGE_HOUR": "23",
        "PRODUCER_RETENTION_DAYS": "1",
        "SCRIBE_SERVER": "localhost:9092",
        "METRICS_HOST": "localhost:9092",
        "METRICS_TOPIC": "metrics",
        "SCRIBE_TOPIC": "w_object",
        "CONSUME_MESSAGES": "1",
        "ENABLE_PARTITION_EOF": "True",
    }
    for key, value in env_variables_dict.items():
        os.environ[key] = value

    return env_variables_dict


def produce_messages(topic):
    producer = KafkaProducer(
        {
            "PARAMS": {"bootstrap.servers": "localhost:9092"},
            "TOPIC": topic,
            "SCHEMA": SCHEMA,
        }
    )
    messages = generate_many(SCHEMA, 10)
    producer.set_key_field("aid")
    random.seed(42)

    for message in messages:
        producer.produce(message)


@pytest.fixture(scope="session")
def kafka_consumer():
    consumer = KafkaConsumer(
        {
            "PARAMS": {
                "bootstrap.servers": "localhost:9092",
                "group.id": "test_steppu",
                "auto.offset.reset": "beginning",
                "enable.partition.eof": True,
            },
            "TOPICS": [get_lc_classifier_topic()],
        }
    )
    yield consumer
    consumer.consumer.close()


@pytest.fixture(scope="session")
def scribe_consumer():
    consumer = KafkaConsumer(
        {
            "PARAMS": {
                "bootstrap.servers": "localhost:9092",
                "group.id": "test_step_",
                "auto.offset.reset": "beginning",
                "enable.partition.eof": True,
            },
            "TOPICS": ["w_object"],
        }
    )
    yield consumer
    consumer.consumer.close()
