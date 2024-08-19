#!/usr/bin/env python3

import pytest
from confluent_kafka.admin import AdminClient, NewTopic
from apf.producers.kafka import KafkaProducer
import os
import pathlib
from ..message_factory import generate_input_batch


os.environ["FEATURE_EXTRACTOR"] = ""
os.environ["CONSUMER_TOPICS"] = ""
os.environ["CONSUMER_SERVER"] = ""
os.environ["CONSUMER_GROUP_ID"] = ""
os.environ["PRODUCER_TOPIC"] = ""
os.environ["PRODUCER_SERVER"] = ""
os.environ["SCRIBE_SERVER"] = ""
os.environ["SCRIBE_TOPIC"] = ""


@pytest.fixture(scope="session")
def docker_compose_file(pytestconfig):
    try:
        path = (
            pathlib.Path(pytestconfig.rootdir) / "tests/integration/docker-compose.yml"
        )
        assert path.exists()
        return path
    except Exception as e:
        path = (
            pathlib.Path(pytestconfig.rootdir)
            / "feature_step/tests/integration/docker-compose.yml"
        )
        assert path.exists()
        return path


@pytest.fixture(scope="session")
def docker_compose_command():
    version = os.getenv("COMPOSE", "v2")
    return "docker compose" if version == "v2" else "docker-compose"


def is_responsive_kafka(url):
    client = AdminClient({"bootstrap.servers": url})
    topics = ["ztf"]
    new_topics = [NewTopic(topic, num_partitions=1) for topic in topics]
    fs = client.create_topics(new_topics)
    for topic, f in fs.items():
        try:
            f.result()
            return True
        except Exception as e:
            return False


@pytest.fixture(scope="session")
def kafka_service(docker_ip, docker_services):
    """Ensure that Kafka service is up and responsive."""
    # `port_for` takes a container port and returns the corresponding host port
    port = docker_services.port_for("kafka", 9092)
    server = "{}:{}".format(docker_ip, port)
    docker_services.wait_until_responsive(
        timeout=30.0, pause=0.1, check=lambda: is_responsive_kafka(server)
    )
    schema_path = pathlib.Path(
        pathlib.Path(__file__).parent.parent.parent.parent,
        "schemas/feature_step",  # TODO: repair this
        "output.avsc",
    )
    data_ztf = generate_input_batch(10, ["g", "r"], survey="ZTF")
    config = {
        "PARAMS": {"bootstrap.servers": "localhost:9092"},
        "TOPIC": "ztf",
        "SCHEMA_PATH": schema_path,
    }
    producer = KafkaProducer(config)
    for data in data_ztf:
        producer.produce(data)
    producer.producer.flush(10)
    return server
