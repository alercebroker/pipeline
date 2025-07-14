import pytest
import uuid
import os
import pathlib

from confluent_kafka.admin import AdminClient, NewTopic
from apf.producers import KafkaProducer
from apf.consumers import KafkaConsumer
from tests.unittests import data as messages


@pytest.fixture(scope="session")
def docker_compose_file(pytestconfig):
    return os.path.join(
        str(pytestconfig.rootdir), "tests/integration", "docker-compose.yaml"
    )


@pytest.fixture(scope="session")
def docker_compose_command():
    version = os.getenv("COMPOSE", "v2")
    return "docker compose" if version == "v2" else "docker-compose"


def is_responsive_kafka(url):
    client = AdminClient({"bootstrap.servers": url})
    topics = ["w_object", "correction"]
    new_topics = [NewTopic(topic, num_partitions=1) for topic in topics]
    fs = client.create_topics(new_topics)
    for topic, f in fs.items():
        try:
            f.result()
        except Exception as e:
            print(f"Can't create topic {topic}")
            print(e)
            return False
    produce_messages("correction")
    return True


@pytest.fixture(scope="session")
def kafka_service(docker_ip, docker_services):
    """Ensure that Kafka service is up and responsive."""
    port = docker_services.port_for("kafka", 9092)
    server = f"{docker_ip}:{port}"
    docker_services.wait_until_responsive(
        timeout=60.0, pause=1, check=lambda: is_responsive_kafka(server)
    )
    return server


@pytest.fixture
def env_variables():
    random_string = uuid.uuid4().hex
    env_variables_dict = {
        "PRODUCER_SCHEMA_PATH": "",
        "CONSUMER_SCHEMA_PATH": "",
        "METRIS_SCHEMA_PATH": "../schemas/magstats_step/metrics.json",
        "SCRIBE_SCHEMA_PATH": "../schemas/scribe_step/scribe.avsc",
        "CONSUMER_SERVER": "localhost:9092",
        "CONSUMER_TOPICS": "correction",
        "CONSUMER_GROUP_ID": random_string,
        "CONSUME_MESSAGES": "10",
        "METRICS_SERVER": "localhost:9092",
        "METRICS_TOPIC": "metrics",
        "ENABLE_PARTITION_EOF": "True",
        "SCRIBE_PRODUCER_SERVER": "localhost:9092",
        "SCRIBE_PRODUCER_TOPIC": "w_object",
    }
    for key, value in env_variables_dict.items():
        os.environ[key] = value

    return env_variables_dict


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
            "TOPICS": ["w_object"],
        }
    )
    yield consumer


def produce_messages(topic):
    producer = KafkaProducer(
        {
            "PARAMS": {"bootstrap.servers": "localhost:9092"},
            "TOPIC": topic,
            "SCHEMA_PATH": str(
                pathlib.Path(
                    pathlib.Path(__file__).parent.parent.parent.parent,
                    "schemas/correction_step",
                    "output.avsc",
                )
            ),
        }
    )

    for message in messages:
        producer.produce(message)
