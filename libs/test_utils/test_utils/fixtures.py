from typing import List
import pytest
from test_utils.utils import (
    is_responsive_psql,
    is_responsive_mongo,
    is_responsive_kafka,
)
from apf.consumers import KafkaConsumer
from confluent_kafka.admin import AdminClient
import uuid


@pytest.fixture(scope="session")
def psql_service(docker_ip, docker_services):
    port = docker_services.port_for("postgres", 5432)
    server = "{}:{}".format(docker_ip, port)
    docker_services.wait_until_responsive(
        timeout=30.0, pause=0.1, check=lambda: is_responsive_psql()
    )
    return server


@pytest.fixture(scope="session")
def mongo_service(docker_ip, docker_services):
    """Ensure that mongo service is up and responsive."""
    port = docker_services.port_for("mongo", 27017)
    server = "{}:{}".format(docker_ip, port)
    docker_services.wait_until_responsive(
        timeout=120.0, pause=0.5, check=lambda: is_responsive_mongo()
    )
    return server


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
def kafka_consumer():
    topics_to_delete = []

    def consumer(topics: List[str]):
        nonlocal topics_to_delete
        topics_to_delete = topics
        consumer = KafkaConsumer(
            {
                "PARAMS": {
                    "bootstrap.servers": "localhost:9092",
                    "group.id": uuid.uuid4().hex,
                    "auto.offset.reset": "earliest",
                    "enable.partition.eof": True,
                },
                "TOPICS": topics,
                "exit_on_consume": True,
            }
        )
        return consumer

    yield consumer
    print("Deleting topics: ", topics_to_delete)
    admin_client = AdminClient({"bootstrap.servers": "localhost:9092"})
    futures = admin_client.delete_topics(
        topics_to_delete, operation_timeout=30, request_timeout=30
    )
    for f in futures.values():
        f.result()
