import os
import pathlib
import logging
import random
import uuid

import pytest
from apf.producers import KafkaProducer
from apf.consumers import KafkaConsumer
from confluent_kafka.admin import AdminClient, NewTopic
from fastavro.utils import generate_many
from fastavro.schema import load_schema
from tests.utils import atlas_extra_fields, lsst_extra_fields, ztf_extra_fields


@pytest.fixture(scope="session")
def docker_compose_command():
    version = os.getenv("COMPOSE", "v2")
    return "docker compose" if version == "v2" else "docker-compose"


@pytest.fixture(scope="session")
def docker_compose_file(pytestconfig):
    return (pathlib.Path(pytestconfig.rootdir) / "tests/integration/docker-compose.yaml").absolute()


def is_responsive_kafka(url):
    client = AdminClient({"bootstrap.servers": url})
    futures = client.create_topics([NewTopic("lightcurve", num_partitions=1)])
    for topic, future in futures.items():
        try:
            future.result()
        except Exception as e:
            logging.error(f"Can't create topic {topic}: {e}")
            return False
    produce_messages("lightcurve")
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


@pytest.fixture
def env_variables():
    random_string = uuid.uuid4().hex
    env_variables_dict = {
        "PRODUCER_SCHEMA_PATH": str(
            pathlib.Path(
                pathlib.Path(__file__).parent.parent.parent.parent,
                "schemas/correction_step",
                "output.avsc",
            )
        ),
        "CONSUMER_SCHEMA_PATH": "",
        "METRIS_SCHEMA_PATH": str(
            pathlib.Path(
                pathlib.Path(__file__).parent.parent.parent.parent,
                "schemas/correction_step",
                "metrics.json",
            )
        ),
        "SCRIBE_SCHEMA_PATH": str(
            pathlib.Path(
                pathlib.Path(__file__).parent.parent.parent.parent,
                "schemas/scribe_step",
                "scribe.avsc",
            )
        ),
        "CONSUMER_SERVER": "localhost:9092",
        "PRODUCER_SERVER": "localhost:9092",
        "SCRIBE_SERVER": "localhost:9092",
        "METRICS_SERVER": "localhost:9092",
        "CONSUMER_TOPICS": "lightcurve",
        "PRODUCER_TOPIC": "corrections",
        "SCRIBE_TOPIC": "w_detections",
        "CONSUME_MESSAGES": "1",
        "CONSUMER_GROUP_ID": random_string,
        "ENABLE_PARTITION_EOF": "True",
    }
    for key in env_variables_dict:
        os.environ[key] = env_variables_dict[key]

    return env_variables_dict


def produce_messages(topic):
    lightcurve_schema = str(
        pathlib.Path(
            pathlib.Path(__file__).parent.parent.parent.parent,
            "schemas/lightcurve_step",
            "output.avsc",
        )
    )
    producer = KafkaProducer(
        {
            "PARAMS": {"bootstrap.servers": "localhost:9092"},
            "TOPIC": topic,
            "SCHEMA_PATH": lightcurve_schema,
        }
    )
    lightcurve_schema = load_schema(lightcurve_schema)
    messages = generate_many(lightcurve_schema, 15)
    producer.set_key_field("oid")
    random.seed(42)

    for message in messages:
        for detection in message["detections"]:
            detection["oid"] = message["oid"]
            detection["forced"] = random.choice([True, False])
            detection["tid"] = random.choice(["ZTF", "ATLAS", "LSST"])
            detection["sid"] = detection["tid"]
            if detection["sid"] == "ZTF":
                detection["fid"] = random.choice(["g", "r", "i"])
            if detection["sid"] == "ATLAS":
                detection["fid"] = random.choice(["o", "c"])
            if detection["sid"] == "LSST":
                detection["fid"] = random.choice(["u", "g", "r", "i", "z", "y"])
            if str(detection["tid"]).lower() == "ztf":
                detection["extra_fields"] = ztf_extra_fields()
            elif str(detection["tid"]).lower() == "lsst":
                detection["extra_fields"] = lsst_extra_fields()
            elif str(detection["tid"]).lower() == "atlas":
                detection["extra_fields"] = atlas_extra_fields()
        producer.produce(message)


@pytest.fixture(scope="session")
def kafka_consumer():
    consumer = KafkaConsumer(
        {
            "PARAMS": {
                "bootstrap.servers": "localhost:9092",
                "group.id": "test_step",
                "auto.offset.reset": "beginning",
                "enable.partition.eof": True,
            },
            "TOPICS": ["corrections"],
            "TIMEOUT": 0,
        }
    )
    yield consumer
    consumer.consumer.close()


@pytest.fixture(scope="session")
def scribe_consumer():
    consumer = KafkaConsumer(
        {
            "PARAMS": {
                "bootstrap.servers": "localhost:9092",
                "group.id": "test_step_",
                "auto.offset.reset": "beginning",
                "enable.partition.eof": True,
            },
            "TOPICS": ["w_detections"],
            "TIMEOUT": 0,
        }
    )
    yield consumer
    consumer.consumer.close()
