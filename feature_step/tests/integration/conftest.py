#!/usr/bin/env python3

import pytest
from confluent_kafka.admin import AdminClient, NewTopic
from apf.producers.kafka import KafkaProducer
import glob
import os
import pathlib
from tests.data.message_factory import (
    generate_input_batch,
)
from tests.data.elasticc_message_factory import (
    generate_input_batch as generate_elasticc_batch,
    ELASTICC_BANDS,
)

PRODUCER_SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "input_schema.avsc")

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
            pathlib.Path(pytestconfig.rootdir)
            / "tests/integration/docker-compose.yml"
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


def is_responsive_kafka(url):
    client = AdminClient({"bootstrap.servers": url})
    topics = ["elasticc", "ztf"]
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
    config = {
        "PARAMS": {"bootstrap.servers": "localhost:9092"},
        "TOPIC": "elasticc",
        "SCHEMA_PATH": PRODUCER_SCHEMA_PATH,
    }
    producer = KafkaProducer(config)
    data_elasticc = generate_elasticc_batch(5, ELASTICC_BANDS)
    data_ztf = generate_input_batch(5)
    for data in data_elasticc:
        producer.produce(data)
    producer.producer.flush(10)
    config = {
        "PARAMS": {"bootstrap.servers": "localhost:9092"},
        "TOPIC": "ztf",
        "SCHEMA_PATH": PRODUCER_SCHEMA_PATH,
    }
    producer = KafkaProducer(config)
    for data in data_ztf:
        producer.produce(data)
    producer.producer.flush(10)
    return server
