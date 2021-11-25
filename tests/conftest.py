import os

# packages not considered in requirements.txt because is only for test. Install it independently or see the gh actions.
import pytest
from confluent_kafka.admin import AdminClient, NewTopic
from pymongo import MongoClient


@pytest.fixture(scope="session")
def docker_compose_file(pytestconfig):
    return os.path.join(
        str(pytestconfig.rootdir), "tests/integration", "docker-compose.yml"
        )


def is_responsive_mongo(url):
    try:
        print(url)
        client = MongoClient(
            "localhost",
            27017,
            username="poolento",
            password="poolento",
            authSource="develop",
        )
        print(client)
        return True
    except Exception as e:
        print(e)
        return False


@pytest.fixture(scope="session")
def mongo_service(docker_ip, docker_services):
    """Ensure that mongo service is up and responsive."""
    port = docker_services.port_for("mongo", 27017)
    server = "{}:{}".format(docker_ip, port)
    docker_services.wait_until_responsive(timeout=30.0, pause=0.1, check=lambda: is_responsive_mongo(server))
    return server


def is_responsive_kafka(url):
    client = AdminClient({"bootstrap.servers": url})
    topics = ["test"]
    new_topics = [NewTopic(topic, num_partitions=1) for topic in topics]
    fs = client.create_topics(new_topics)
    for topic, f in fs.items():
        try:
            f.result()
            return True
        except Exception as e:
            print(e)
            return False


@pytest.fixture(scope="session")
def kafka_service(docker_ip, docker_services):
    """Ensure that Kafka service is up and responsive."""
    print("Kafka", docker_ip)
    # `port_for` takes a container port and returns the corresponding host port
    port = docker_services.port_for("kafka", 9092)
    server = "{}:{}".format(docker_ip, port)
    docker_services.wait_until_responsive(timeout=30.0, pause=0.1, check=lambda: is_responsive_kafka(server))
