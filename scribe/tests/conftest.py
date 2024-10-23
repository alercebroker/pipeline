import logging
import os

import psycopg2
import pytest
from confluent_kafka.admin import AdminClient, NewTopic
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure


@pytest.fixture(scope="session")
def docker_compose_file(pytestconfig):
    return os.path.join(
        str(pytestconfig.rootdir), "tests/integration", "docker-compose.yml"
    )

@pytest.fixture(scope="session")
def docker_compose_command():
    version = os.getenv("COMPOSE", "v2")
    return "docker compose" if version == "v2" else "docker-compose"


def is_mongo_responsive(ip, port):
    client = MongoClient(f"mongodb://{ip}:{port}")
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


def is_psql_responsive(ip, port):
    try:
        conn = psycopg2.connect(
            "dbname='postgres' user='postgres' host=localhost password='postgres'"
        )
        conn.close()
        return True
    except Exception as e:
        print(e)
        return False


@pytest.fixture(scope="session")
def psql_service(docker_ip, docker_services):
    port = docker_services.port_for("postgres", 5432)
    server = f"{docker_ip}:{port}"
    docker_services.wait_until_responsive(
        timeout=30.0, pause=0.1, check=lambda: is_psql_responsive(server, port)
    )
    return server


def create_topics(client: AdminClient):
    new_topics = [
        NewTopic("test_topic", num_partitions=1),
        NewTopic("test_topic_2", num_partitions=1),
        NewTopic("test_topic_sql", num_partitions=1),
    ]
    fs = client.create_topics(new_topics)
    for topic, f in fs.items():
        try:
            f.result()
        except Exception as e:
            logging.error(f"Can't create topic {topic}: {e}")
            return False
    return True


def delete_topics(client: AdminClient):
    fs = client.delete_topics(
        ["test_topic", "test_topic_2", "test_topic_sql"],
        operation_timeout=60,
        request_timeout=60,
    )
    for topic, f in fs.items():
        try:
            f.result()
        except Exception as e:
            logging.error(f"Can't delete topic {topic}: {e}")
            return False


def is_kafka_responsive(url="localhost:9092"):
    client = AdminClient({"bootstrap.servers": url})
    responsive = delete_topics(client)
    responsive = create_topics(client)
    return responsive


@pytest.fixture(scope="session")
def kafka_service(docker_ip, docker_services):
    port = docker_services.port_for("kafka", 9092)
    server = f"{docker_ip}:{port}"
    docker_services.wait_until_responsive(
        timeout=30.0, pause=1, check=lambda: is_kafka_responsive(server)
    )
    return server
