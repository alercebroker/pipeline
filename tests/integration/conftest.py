import os
from apf.consumers import KafkaConsumer
import pytest
from confluent_kafka.admin import AdminClient, NewTopic


@pytest.fixture(scope="session")
def docker_compose_file(pytestconfig):
    return os.path.join(
        str(pytestconfig.rootdir), "tests/integration", "docker-compose.yml"
    )


@pytest.fixture(scope="session")
def docker_compose_command():
    v2 = False
    if os.getenv("COMPOSE", "v1") == "v2":
        v2 = True
    return "docker compose" if v2 else "docker-compose"


def is_responsive_kafka(url):
    client = AdminClient({"bootstrap.servers": url})
    topics = ["test_topic"]
    new_topics = [NewTopic(topic, num_partitions=1) for topic in topics]
    fs = client.create_topics(new_topics)
    for topic, f in fs.items():
        try:
            f.result()
        except Exception as e:
            print(f"Can't create topic {topic}")
            print(e)
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


@pytest.fixture()
def env_variables():
    os.environ["SCHEMA_PATH"] = "schemas/lsst.json"
    os.environ["PRODUCER_SERVER"] = "localhost:9092"
    os.environ["PRODUCER_TOPIC"] = "test_topic"
    os.environ["MESSAGES"] = "1000"
    os.environ["CONSUME_TIMEOUT"] = "30"


@pytest.fixture()
def kafka_consumer():
    config = {
        "PARAMS": {
            "bootstrap.servers": "localhost:9092",
            "group.id": "consumer",
            "auto.offset.reset": "beginning",
            "enable.partition.eof": True,
        },
        "TOPICS": ["test_topic"],
    }
    consumer = KafkaConsumer(config)
    yield consumer
