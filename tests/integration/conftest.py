import os
from apf.producers import KafkaProducer
from apf.consumers import KafkaConsumer
from db_plugins.db.mongo.initialization import init_mongo_database
from pymongo import MongoClient

import pytest
from .schema import SCHEMA


@pytest.fixture(scope="session")
def docker_compose_file(pytestconfig):
    return os.path.join(
        str(pytestconfig.rootdir), "tests/integration", "docker-compose.yaml"
    )


@pytest.fixture(scope="session")
def docker_compose_command():
    v2 = False
    if os.getenv("COMPOSE", "v1") == "v2":
        v2 = True
    return "docker compose" if v2 else "docker-compose"


def consume_messages() -> list:
    config = {
        "PARAMS": {
            "bootstrap.servers": "localhost:9092",
            "group.id": "conftest",
            "auto.offset.reset": "beginning",
            "max.poll.interval.ms": 3600000,
            "enable.partition.eof": True,
        },
        "consume.timeout": 10,
        "consume.messages": 1,
        "TOPICS": ["sorting_hat"],
    }
    consumer = KafkaConsumer(config)
    messages = []
    for message in consumer.consume():
        messages.append(message)
    return messages


def is_responsive_mongo():
    try:
        client = MongoClient(
            "localhost", 27017, username="root", password="root", authSource="admin"
        )
        client.server_info()  # check connection
        # Create test test_user and test_db
        db = client.test_db
        db.command(
            "createUser",
            "test_user",
            pwd="test_password",
            roles=["dbOwner", "readWrite"],
        )
        # put credentials to init database (create collections and indexes)
        settings = {
            "HOST": "localhost",
            "USERNAME": "test_user",
            "PASSWORD": "test_password",
            "PORT": 27017,
            "DATABASE": "test_db",
            "AUTH_SOURCE": "test_db",
        }
        init_mongo_database(settings)
        return True
    except Exception as e:
        print(e)
        return False


@pytest.fixture(scope="session")
def mongo_service(docker_ip, docker_services):
    """Ensure that mongo service is up and responsive."""
    port = docker_services.port_for("mongo", 27017)
    server = "{}:{}".format(docker_ip, port)
    docker_services.wait_until_responsive(
        timeout=30.0, pause=0.1, check=lambda: is_responsive_mongo()
    )
    return server


def is_responsive_kafka(url):
    try:
        generate_messages()
        messages = consume_messages()
        assert len(messages) == 1
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
        timeout=30.0, pause=0.1, check=lambda: is_responsive_kafka(server)
    )
    return server


def get_binary(filename: str):
    with open(f"tests/data/{filename}.fits.gz", "rb") as f:
        data = f.read()
        return data


def generate_messages():
    alert = {
        "oid": "oid",
        "tid": "tid",
        "pid": 1,
        "candid": "123",
        "mjd": 1,
        "fid": 1,
        "ra": 1,
        "dec": 1,
        "rb": 1,
        "rbversion": "a",
        "mag": 1,
        "e_mag": 1,
        "rfid": 1,
        "isdiffpos": 1,
        "e_ra": 1,
        "e_dec": 1,
        "extra_fields": {},
        "aid": "aid",
        "stamps": {
            "science": get_binary("science"),
            "template": None,
            "difference": get_binary("difference"),
        },
    }
    producer = KafkaProducer(
        {
            "TOPIC": "sorting_hat",
            "PARAMS": {
                "bootstrap.servers": "localhost:9092",
            },
            "SCHEMA": SCHEMA,
        }
    )
    producer.produce(alert)
