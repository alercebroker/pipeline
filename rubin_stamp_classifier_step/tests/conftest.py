import pytest
import os
import time
from confluent_kafka.admin import AdminClient


# General docker compose command fixture
@pytest.fixture(scope="session")
def docker_compose_command():
    version = os.getenv("COMPOSE", "v2")
    return "docker compose" if version == "v2" else "docker-compose"


# General docker compose file fixture
@pytest.fixture(scope="session")
def docker_compose_file(pytestconfig):
    compose = pytestconfig.rootdir / "tests/integration/docker-compose.yml"
    assert compose.exists()
    return compose


def is_responsive_kafka(url):
    client = AdminClient({"bootstrap.servers": url})
    try:
        # Try to list topics to check if Kafka is up
        client.list_topics(timeout=5)
        return True
    except Exception as e:
        current_epoch_time = time.time()
        print(f"Kafka not responsive: {e}, timestamp: {current_epoch_time}")
        return False


@pytest.fixture(scope="session")
def kafka_service(docker_ip, docker_services):
    """Ensure that Kafka service is up and responsive."""
    port = docker_services.port_for("kafka", 9092)
    server = f"{docker_ip}:{port}"
    print(f"[kafka_service] Waiting for Kafka at {server}")
    docker_services.wait_until_responsive(
        timeout=60.0, pause=0.5, check=lambda: is_responsive_kafka(server)
    )
    current_epoch_time = time.time()
    print(
        f"[kafka_service] Kafka is ready at {server}, timestamp: {current_epoch_time}"
    )
    return server
