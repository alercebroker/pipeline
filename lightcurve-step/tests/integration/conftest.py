import os
import pytest
from confluent_kafka.admin import AdminClient, NewTopic
from apf.producers import KafkaProducer
import uuid
from fastavro.utils import generate_many
from fastavro.schema import load_schema
from fastavro.repository.base import SchemaRepositoryError
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import psycopg2
from db_plugins.db.sql._connection import PsqlDatabase
from .utils import generate_message

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


def is_responsive_kafka(url):
    client = AdminClient({"bootstrap.servers": url})
    topics = ["correction", "lightcurve"]
    new_topics = [NewTopic(topic, num_partitions=1) for topic in topics]
    fs = client.create_topics(new_topics)
    for topic, f in fs.items():
        try:
            f.result()
        except Exception as e:
            print(f"Can't create topic {topic}")
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
    return docker_ip, port


@pytest.fixture
def env_variables():
    random_string = uuid.uuid4().hex
    env_variables_dict = {
        "CONSUMER_SERVER": "localhost:9092",
        "CONSUMER_TOPICS": "correction",
        "CONSUMER_GROUP_ID": random_string,
        "METRICS_SERVER": "localhost:9092",
        "PRODUCER_SERVER": "localhost:9092",
        "PRODUCER_TOPIC": "lightcurve",
        "ENABLE_PARTITION_EOF": "True",
        "MONGODB_SECRET_NAME": "mongo_secret",
        "SQL_SECRET_NAME": "sql_secret",
    }
    for key in env_variables_dict:
        os.environ[key] = env_variables_dict[key]

    return env_variables_dict

@pytest.fixture()
def produce_messages(kafka_service):
    try:
        schema = load_schema("tests/integration/input_schema.avsc")
    except SchemaRepositoryError:
        schema = load_schema("lightcurve-step/tests/integration/input_schema.avsc")
    producer = KafkaProducer(
        {
            "PARAMS": {"bootstrap.servers": "localhost:9092"},
            "TOPIC": "correction",
            "SCHEMA": schema,
        }
    )
    messages = generate_message(schema, 10)
    producer.set_key_field("aid")

    for message in messages:
        producer.produce(message)
    producer.producer.flush()
    return

def is_responsive_psql(url):
    try:
        conn = psycopg2.connect(
            "dbname='postgres' user='postgres' host=localhost password='postgres'"
        )
        conn.close()
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def psql_service(docker_ip, docker_services):
    """Ensure that Kafka service is up and responsive."""
    # `port_for` takes a container port and returns the corresponding host port
    port = docker_services.port_for("postgres", 5432)
    server = "{}:{}".format(docker_ip, port)
    docker_services.wait_until_responsive(
        timeout=30.0, pause=0.1, check=lambda: is_responsive_psql(server)
    )
    return server


@pytest.fixture()
def psql_conn(psql_service):
    config = {
        "USER": "postgres",
        "PASSWORD": "postgres",
        "HOST": "localhost",
        "PORT": "5432",
        "DB_NAME": "postgres",
    }
    psql_conn = PsqlDatabase(config)
    psql_conn.create_db()
    yield psql_conn
    psql_conn.drop_db()
