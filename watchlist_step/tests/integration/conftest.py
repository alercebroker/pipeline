import pytest
from confluent_kafka.admin import AdminClient, NewTopic
from confluent_kafka import Producer, Consumer
from db_plugins.db.sql import SQLConnection
from db_plugins.db.sql.models import Detection, Object, Probability
from apf.consumers import KafkaConsumer
from confluent_kafka import KafkaError, KafkaException, Consumer
from fastavro import writer, parse_schema, reader
import glob
import os
import psycopg2
from io import BytesIO


@pytest.fixture(scope="session")
def docker_compose_file(pytestconfig):
    return os.path.join(
        str(pytestconfig.rootdir), "tests/integration", "docker-compose.yml"
    )


def produce_message(config):
    schema = {
        "doc": "Late Classification",
        "name": "probabilities_and_features",
        "type": "record",
        "fields": [
            {"name": "oid", "type": "string"},
            {"name": "candid", "type": "long"},
        ],
    }
    parsed_schema = parse_schema(schema)
    records = [
        {"oid": "ZTF19aaapkto", "candid": 1000151433015015013},
    ]
    producer = Producer(config)
    topics = ["test"]
    fo = BytesIO()
    writer(fo, parsed_schema, records)
    fo.seek(0)
    try:
        for topic in topics:
            producer.produce(topic, value=fo.read())
            producer.flush(30)
            fo.seek(0)
            producer.produce(topic, value=fo.read())
            producer.flush(30)
            fo.seek(0)
            print(f"produced to {topic} {fo.read()}")
    except Exception as e:
        print(f"failed to produce to topic: {e}")


def consume_message(config):
    consumer = KafkaConsumer(config)
    messages = 0
    max_messages = 1
    for msg in consumer.consume():
        print(msg)


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
            return False


@pytest.fixture(scope="session")
def kafka_service(docker_ip, docker_services):
    """Ensure that Kafka service is up and responsive."""
    topics = ["test"]
    # `port_for` takes a container port and returns the corresponding host port
    port = docker_services.port_for("kafka", 9094)
    server = "{}:{}".format(docker_ip, port)
    docker_services.wait_until_responsive(
        timeout=30.0, pause=0.1, check=lambda: is_responsive_kafka(server)
    )
    config = {"bootstrap.servers": "localhost:9094"}
    produce_message(config)
    return server


def is_responsive_alerts_database(url):
    try:
        conn = psycopg2.connect(
            dbname="postgres",
            user="postgres",
            host="localhost",
            password="postgres",
        )
        conn.close()
        return True
    except:
        return False


def init_db(insert: bool, config: dict):
    db = SQLConnection()
    db.connect(config)
    db.create_db()
    if insert:
        obj = Object(oid="ZTF19aaapkto", firstmjd=100000000)
        db.session.add(obj)
        det = Detection(
            candid=1000151433015015013,
            oid="ZTF19aaapkto",
            mjd=123,
            fid=1,
            pid=0.5,
            isdiffpos=1,
            ra=252.6788662886394,
            dec=53.34521158573315,
            magpsf=0.5,
            sigmapsf=0.5,
            corrected=False,
            dubious=False,
            has_stamp=False,
            step_id_corr="test",
        )
        db.session.add(det)
        db.session.commit()


def remove_db(config: dict):
    db = SQLConnection()
    db.connect(config)
    db.drop_db()


@pytest.fixture(scope="session")
def alerts_database(docker_ip, docker_services):
    """Ensure that Kafka service is up and responsive."""
    # `port_for` takes a container port and returns the corresponding host port
    port = docker_services.port_for("alerts_db", 5432)
    server = "{}:{}".format(docker_ip, port)
    docker_services.wait_until_responsive(
        timeout=30.0, pause=0.1, check=lambda: is_responsive_alerts_database(server)
    )
    config = {
        "SQL": {
            "ENGINE": "postgresql",
            "HOST": "localhost",
            "USER": "postgres",
            "PASSWORD": "postgres",
            "PORT": 5432,  # postgresql tipically runs on port 5432. Notice that we use an int here.
            "DB_NAME": "postgres",
        },
        "SQLALCHEMY_DATABASE_URL": "postgresql://postgres:postgres@localhost:5432/postgres",
    }
    init_db(insert=True, config=config)
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
    except:
        return False


@pytest.fixture(scope="session")
def users_database(docker_ip, docker_services):
    """Ensure that Kafka service is up and responsive."""
    # `port_for` takes a container port and returns the corresponding host port
    port = docker_services.port_for("users_db", 5432)
    server = "{}:{}".format(docker_ip, port)
    docker_services.wait_until_responsive(
        timeout=30.0,
        pause=0.1,
        check=lambda: is_responsive_users_database(docker_ip, port),
    )
    return server
