import os
import pathlib
import random
import uuid

import pytest
from apf.producers import KafkaProducer
from apf.consumers import KafkaConsumer
from confluent_kafka.admin import AdminClient, NewTopic
from fastavro.utils import generate_many

from schema import SCHEMA
from tests.utils import ztf_extra_fields


@pytest.fixture(scope="session")
def docker_compose_file(pytestconfig):
    return (
        pathlib.Path(pytestconfig.rootdir) / "tests/integration/docker-compose.yaml"
    ).absolute()


@pytest.fixture(scope="session")
def docker_compose_command():
    v2 = False
    if os.getenv("COMPOSE", "v1") == "v2":
        v2 = True
    return "docker compose" if v2 else "docker-compose"


def is_responsive_kafka(url):
    client = AdminClient({"bootstrap.servers": url})
    topics = ["prv-candidates"]
    new_topics = [NewTopic(topic, num_partitions=1) for topic in topics]
    fs = client.create_topics(new_topics)
    for topic, f in fs.items():
        try:
            f.result()
        except Exception as e:
            print(f"Can't create topic {topic}")
            print(e)
            return False
    produce_messages("prv-candidates")
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
        "CONSUMER_TOPICS": "prv-candidates",
        "CONSUME_MESSAGES": "10",
        "CONSUMER_GROUP_ID": random_string,
        "METRICS_HOST": "localhost:9092",
        "PRODUCER_SERVER": "localhost:9092",
        "PRODUCER_TOPIC": "corrections",
        "ENABLE_PARTITION_EOF": "True",
        "SCRIBE_SERVER": "localhost:9092",
        "SCRIBE_TOPIC": "w_detections",
    }
    for key in env_variables_dict:
        os.environ[key] = env_variables_dict[key]

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
        message["new_alert"]["tid"] = "ZTF" if random.random() > 0.5 else "ATLAS"
        if str(message["new_alert"]["tid"]).lower() == "ztf":
            message["new_alert"]["extra_fields"] = ztf_extra_fields()
            for prv in message["prv_detections"]:
                prv["extra_fields"] = ztf_extra_fields()
        producer.produce(message)


@pytest.fixture(scope="session")
def kafka_consumer():
    consumer = KafkaConsumer(
        {
            "PARAMS": {
                "bootstrap.servers": "localhost:9092",
                "group.id": "test_step",
                "auto.offset.reset": "beginning",
                "enable.partition.eof": True,
            },
            "TOPICS": ["corrections"],
        }
    )
    yield consumer


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
            "TOPICS": ["w_detections"],
        }
    )
    yield consumer
