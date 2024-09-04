import os
from io import BytesIO
import pathlib

import psycopg2
import pytest
from apf.producers import KafkaProducer, KafkaSchemalessProducer
from apf.consumers import KafkaConsumer
from confluent_kafka import Producer
from fastavro.schema import load_schema
from fastavro.utils import generate_many
from confluent_kafka.admin import AdminClient, NewTopic
from fastavro import writer, parse_schema

from watchlist_step.db.connection import PsqlDatabase
from tests.integration.mocks.mock_alerts import ztf_extra_fields_generator


@pytest.fixture(scope="session")
def docker_compose_command():
    if os.getenv("COMPOSE_VERSION", "") == "V1":
        return "docker-compose"
    return "docker compose"


@pytest.fixture(scope="session")
def docker_compose_file(pytestconfig):
    return os.path.join(
        str(pytestconfig.rootdir), "tests/integration", "docker-compose.yml"
    )


def produce_message(config):
    schema_path = pathlib.Path(
        pathlib.Path(__file__).parent.parent.parent.parent,
        "schemas/sorting_hat_step",
        "output.avsc",
    )
    producer = KafkaSchemalessProducer(
        {
            "PARAMS": {"bootstrap.servers": config["bootstrap.servers"]},
            "TOPIC": config["topic"],
            "SCHEMA_PATH": schema_path,
        }
    )
    schema = load_schema(str(schema_path))
    messages = generate_many(schema, 10)
    producer.set_key_field("oid")

    for message in messages:
        message["sid"] = "ZTF"
        if message["sid"] == "ZTF":
            message["extra_fields"] = ztf_extra_fields_generator()
        producer.produce(message)

def consume_message(config):
    consumer = KafkaConsumer(config)
    for msg in consumer.consume():
        print(msg)


def is_responsive_kafka(url):
    client = AdminClient({"bootstrap.servers": url})
    topics = ["test"]
    new_topics = [NewTopic(topic, num_partitions=1) for topic in topics]
    fs = client.create_topics(new_topics)
    for _, f in fs.items():
        try:
            f.result()
            return True
        except Exception:
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
        "bootstrap.servers": "localhost:9092",
        "topic": "test"        
    }

    produce_message(config)
    return server


def is_responsive_users_database(docker_ip, port):
    try:
        conn = psycopg2.connect(
            dbname="postgres",
            user="postgres",
            host=docker_ip,
            password="password",
            port=port,
        )
        conn.close()
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def users_db(docker_ip, docker_services):
    """Ensure that Kafka service is up and responsive."""
    # `port_for` takes a container port and returns the corresponding host port
    port = docker_services.port_for("users_db", 5432)
    docker_services.wait_until_responsive(
        timeout=30.0,
        pause=0.1,
        check=lambda: is_responsive_users_database(docker_ip, port),
    )
    users_db = PsqlDatabase(
        {
            "HOST": docker_ip,
            "USER": "postgres",
            "PASSWORD": "postgres",
            "PORT": port,
            "DB_NAME": "postgres",
        }
    )
    return users_db
