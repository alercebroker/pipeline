from confluent_kafka.admin import AdminClient
from confluent_kafka.admin import NewTopic
from consolidated_metrics_step.utils.metric import ConsolidatedMetric
from redis_om import get_redis_connection
from redis_om import Migrator

import os
import pytest


@pytest.fixture(scope="session")
def docker_compose_file(pytestconfig):
    return os.path.join(
        str(pytestconfig.rootdir), "tests/integration", "docker-compose.yaml"
    )


def is_responsive_kafka(url):
    client = AdminClient({"bootstrap.servers": "localhost:9092"})
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
    port = docker_services.port_for("kafka", 9092)
    server = "{}:{}".format(docker_ip, port)
    docker_services.wait_until_responsive(
        timeout=40.0, pause=0.1, check=lambda: is_responsive_kafka(server)
    )
    return server


def is_responsive_redis(url):
    redis_conn = get_redis_connection(url=url)
    test = ConsolidatedMetric(candid="test", survey="test")
    test.save()
    Migrator().run()
    return True


@pytest.fixture(scope="session")
def redis_service(docker_ip, docker_services):
    port = docker_services.port_for("redis", 6379)
    server = f"redis://:@localhost:{port}/0"
    docker_services.wait_until_responsive(
        timeout=40.0, pause=0.1, check=lambda: is_responsive_redis(server)
    )
    return server
