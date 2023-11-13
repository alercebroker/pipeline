import pytest
from confluent_kafka.admin import AdminClient, NewTopic
from apf.producers import KafkaProducer
from apf.consumers import KafkaConsumer
import pathlib
import os
from fastavro.utils import generate_many
import random
from tests.shared.sorting_hat_schema import SCHEMA
from tests.mocks.mock_alerts import ztf_extra_fields_generator
import uuid


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
    topics = ["sorting-hat"]
    new_topics = [NewTopic(topic, num_partitions=1) for topic in topics]
    fs = client.create_topics(new_topics)
    for topic, f in fs.items():
        try:
            f.result()
        except Exception as e:
            print(f"Can't create topic {topic}")
            print(e)
            return False
    produce_messages("sorting-hat")
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
        "PRODUCER_SCHEMA_PATH": "../schemas/prv_candidate_step/output.avsc",
        "CONSUMER_SCHEMA_PATH": "",
        "METRIS_SCHEMA_PATH": "../schemas/prv_candidate_step/metrics.json",
        "SCRIBE_SCHEMA_PATH": "../schemas/scribe.avsc",
        "CONSUMER_SERVER": "localhost:9092",
        "CONSUMER_TOPICS": "sorting-hat",
        "CONSUMER_GROUP_ID": random_string,
        "METRICS_HOST": "localhost:9092",
        "PRODUCER_SERVER": "localhost:9092",
        "PRODUCER_TOPIC": "prv-candidates",
        "ENABLE_PARTITION_EOF": "True",
        "SCRIBE_PRODUCER_SERVER": "localhost:9092",
        "SCRIBE_PRODUCER_TOPIC": "w_non_detections",
    }
    for key in env_variables_dict:
        os.environ[key] = env_variables_dict[key]

    return env_variables_dict


def produce_messages(topic):
    producer = KafkaProducer(
        {
            "PARAMS": {"bootstrap.servers": "localhost:9092"},
            "TOPIC": topic,
            "SCHEMA_PATH": os.path.join(os.path.dirname(__file__), "../shared/sorting_hat_schema.avsc"),
        }
    )
    messages = generate_many(SCHEMA, 10)
    producer.set_key_field("aid")
    random.seed(42)

    for message in messages:
        message["sid"] = "ZTF" if random.random() > 0.5 else "ATLAS"
        if message["sid"] == "ZTF":
            message["extra_fields"] = ztf_extra_fields_generator()
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
            "TOPICS": ["prv-candidates"],
            "TIMEOUT": 0,
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
            "TOPICS": ["w_non_detections"],
            "TIMEOUT": 0,
        }
    )
    yield consumer
