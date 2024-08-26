import os
from io import BytesIO

import psycopg2
import pytest
from apf.consumers import KafkaConsumer
from confluent_kafka import Producer
from confluent_kafka.admin import AdminClient, NewTopic
from fastavro import writer, parse_schema

from watchlist_step.db.connection import PsqlDatabase


@pytest.fixture(scope="session")
def docker_compose_command():
    return "docker compose"


@pytest.fixture(scope="session")
def docker_compose_file(pytestconfig):
    return os.path.join(
        str(pytestconfig.rootdir), "tests/integration", "docker-compose.yml"
    )


def produce_message(config):
    schema = {
        "doc": "Multi stream alert of any telescope/survey",
        "name": "alerce.alert",
        "type": "record",
        "fields": [
            {"name": "oid", "type": "string"},
            {"name": "candid", "type": "long"},
            {"name": "pid", "type": "long"},
            {"name": "mjd", "type": "double"},
            {"name": "fid", "type": "string"},
            {"name": "ra", "type": "double"},
            {"name": "dec", "type": "double"},
            {"name": "mag", "type": "float"},
            {"name": "e_mag", "type": "float"},
            {"name": "isdiffpos", "type": "int"},
        ],
    }
    parsed_schema = parse_schema(schema)
    records = [
        {
            "oid": "ZTF19aaapkto",
            "candid": 1000151433015015013,
            "pid": 0.5,
            "mjd": 123,
            "fid": "1",
            "ra": 252.6788662886394,
            "dec": 53.34521158573315,
            "mag": 0.5,
            "e_mag": 0.5,
            "isdiffpos": 1,
        },
        {
            "oid": "ZTF19aaapktoBAD",
            "candid": 1000151433015015014,
            "pid": 0.5,
            "mjd": 123,
            "fid": "1",
            "ra": 252.6788662886394,
            "dec": 53.34521158573315,
            "mag": 10.5,
            "e_mag": 0.5,
            "isdiffpos": 1,
        },
    ] * 10
    producer = Producer(config)
    topics = ["test"]
    fo = BytesIO()
    for record in records:
        for topic in topics:
            writer(fo, parsed_schema, [record], "null", 160000, None, None, None, None)
            fo.seek(0)
            producer.produce(topic, value=fo.read())
            fo.seek(0)
    producer.flush()


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
    port = docker_services.port_for("kafka", 9094)
    server = "{}:{}".format(docker_ip, port)
    docker_services.wait_until_responsive(
        timeout=30.0, pause=0.1, check=lambda: is_responsive_kafka(server)
    )
    config = {"bootstrap.servers": "localhost:9094"}

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
            "PASSWORD": "password",
            "PORT": port,
            "DB_NAME": "postgres",
        }
    )
    return users_db
