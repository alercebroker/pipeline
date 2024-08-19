import pytest
import os
from confluent_kafka.admin import AdminClient, NewTopic


@pytest.fixture(scope="session")
def docker_compose_command():
    version = os.getenv("COMPOSE", "v2")
    return "docker compose" if version == "v2" else "docker-compose"


@pytest.fixture(scope="session")
def docker_compose_file(pytestconfig):
    try:
        compose = pytestconfig.rootdir / "tests/integration/docker-compose.yml"
        assert compose.exists()
    except AssertionError:
        compose = (
            pytestconfig.rootdir
            / "reflector_step/tests/integration/docker-compose.yml"
        )
        assert compose.exists()
    return compose


def is_responsive_kafka(url):
    client = AdminClient({"bootstrap.servers": url})
    topics = ["test_topic"]
    new_topics = [NewTopic(topic, num_partitions=1) for topic in topics]
    fs = client.create_topics(new_topics)
    for topic, f in fs.items():
        try:
            f.result()
        except Exception as e:
            print(e)
            return False
    return True


@pytest.fixture(scope="session")
def kafka_service(docker_ip, docker_services):
    """Ensure that Kafka service is up and responsive."""
    port1 = docker_services.port_for("kafka1", 9092)
    port2 = docker_services.port_for("kafka2", 9093)
    server1 = "{}:{}".format(docker_ip, port1)
    server2 = "{}:{}".format(docker_ip, port2)
    docker_services.wait_until_responsive(
        timeout=60.0, pause=0.5, check=lambda: is_responsive_kafka(server1)
    )
    docker_services.wait_until_responsive(
        timeout=60.0, pause=0.5, check=lambda: is_responsive_kafka(server2)
    )
    return server1, server2
