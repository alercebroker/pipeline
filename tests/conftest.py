import pytest
import os
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

from confluent_kafka.admin import AdminClient, NewTopic


@pytest.fixture(scope="session")
def docker_compose_file(pytestconfig):
    return os.path.join(
        str(pytestconfig.rootdir), "tests/integration", "docker-compose.yml"
    )


def is_mongo_responsive(ip, port):
    client = MongoClient(f"mongodb://mongo:mongo@{ip}:{port}")
    try:
        client.admin.command("ismaster")
        return True
    except ConnectionFailure:
        return False


@pytest.fixture(scope="session")
def mongo_service(docker_ip, docker_services):
    port = docker_services.port_for("mongodb", 27017)
    docker_services.wait_until_responsive(
        timeout=90.0,
        pause=0.1,
        check=lambda: is_mongo_responsive(docker_ip, port),
    )
    return (docker_ip, port)


def is_kafka_responsive(url="localhost:9092"):
    client = AdminClient({"bootstrap.servers": url})
    new_topics = [NewTopic("test_topic", num_partitions=1)]
    fs = client.create_topics(new_topics)
    for _, f in fs.items():
        try:
            f.result()
        except Exception as e:
            print(e)
            return False
    return True


@pytest.fixture(scope="session")
def kafka_service(docker_ip, docker_services):
    port = docker_services.port_for("kafka", 9092)
    server = f"{docker_ip}:{port}"
    docker_services.wait_until_responsive(
        timeout=40.0, pause=0.1, check=lambda: is_kafka_responsive(server)
    )
    return server
