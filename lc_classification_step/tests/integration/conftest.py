import logging
import os
import pathlib
import random
import time
import uuid
from datetime import datetime

import pytest
from apf.consumers import KafkaConsumer
from apf.producers import KafkaProducer
from confluent_kafka.admin import AdminClient, NewTopic
from fastavro.utils import generate_many

from tests.mockdata.extra_felds import generate_extra_fields
from tests.mockdata.input_elasticc import INPUT_SCHEMA as SCHEMA_ELASTICC
from tests.mockdata.input_ztf import INPUT_SCHEMA as SCHEMA_ZTF


def pytest_configure(config):
    config.addinivalue_line("markers", "ztf: mark a test as a ztf test.")
    config.addinivalue_line("markers", "elasticc: mark a test as a elasticc test.")


@pytest.fixture(scope="session")
def docker_compose_command():
    return (
        "docker compose" if not os.getenv("COMPOSE", "v1") == "v1" else "docker-compose"
    )


@pytest.fixture(scope="session")
def docker_compose_file(pytestconfig):
    return (
        pathlib.Path(pytestconfig.rootdir) / "tests/integration/docker-compose.yml"
    ).absolute()


def get_lc_classifier_topic(model: str):
    return f"lc_classifier_{model}{datetime.utcnow().strftime('%Y%m%d')}"


def is_responsive_kafka(url):
    client = AdminClient({"bootstrap.servers": url})
    futures = client.create_topics(
        [
            NewTopic("features_ztf", num_partitions=1),
            NewTopic("features_elasticc", num_partitions=1),
            NewTopic("w_object", num_partitions=1),
            NewTopic(get_lc_classifier_topic("ztf"), num_partitions=1),
            NewTopic(get_lc_classifier_topic("balto"), num_partitions=1),
            NewTopic(get_lc_classifier_topic("messi"), num_partitions=1),
            NewTopic(get_lc_classifier_topic("toretto"), num_partitions=1),
            NewTopic(get_lc_classifier_topic("barney"), num_partitions=1),
            NewTopic("metrics", num_partitions=1),
        ]
    )
    for topic, future in futures.items():
        try:
            future.result()
            if topic == "features_ztf":
                produce_messages("features_ztf", SCHEMA_ZTF)
            elif topic == "features_elasticc":
                produce_messages("features_elasticc", SCHEMA_ELASTICC)
        except Exception as e:
            logging.error(f"Can't create topic {topic}: {e}")
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


@pytest.fixture
def env_variables_ztf():
    def set_env_variables():
        random_string = uuid.uuid4().hex
        env_variables_dict = {
            "CONSUMER_SERVER": "localhost:9092",
            "CONSUMER_TOPICS": "features_ztf",
            "CONSUMER_GROUP_ID": random_string,
            "PRODUCER_SERVER": "localhost:9092",
            "PRODUCER_TOPIC_FORMAT": "lc_classifier_ztf%s",
            "PRODUCER_DATE_FORMAT": "%Y%m%d",
            "PRODUCER_CHANGE_HOUR": "23",
            "PRODUCER_RETENTION_DAYS": "1",
            "SCRIBE_SERVER": "localhost:9092",
            "METRICS_HOST": "localhost:9092",
            "METRICS_TOPIC": "metrics",
            "SCRIBE_TOPIC": "w_object",
            "CONSUME_MESSAGES": "5",
            "ENABLE_PARTITION_EOF": "True",
            "STREAM": "ztf",
        }
        for key, value in env_variables_dict.items():
            os.environ[key] = value

    return set_env_variables


@pytest.fixture
def env_variables_elasticc():
    def set_env_variables(
        model: str,
        model_class: str,
        predictor_parser_class: str,
        extra_env_vars: dict,
    ):
        random_string = uuid.uuid4().hex
        env_variables_dict = {
            "CONSUMER_SERVER": "localhost:9092",
            "CONSUMER_TOPICS": "features_elasticc",
            "CONSUMER_GROUP_ID": random_string,
            "PRODUCER_SERVER": "localhost:9092",
            "PRODUCER_TOPIC_FORMAT": f"lc_classifier_{model}%s",
            "PRODUCER_DATE_FORMAT": "%Y%m%d",
            "PRODUCER_CHANGE_HOUR": "23",
            "PRODUCER_RETENTION_DAYS": "1",
            "SCRIBE_SERVER": "localhost:9092",
            "METRICS_HOST": "localhost:9092",
            "METRICS_TOPIC": "metrics",
            "SCRIBE_TOPIC": "w_object",
            "CONSUME_MESSAGES": "5",
            "ENABLE_PARTITION_EOF": "True",
            "STREAM": "elasticc",
            "MODEL_CLASS": model_class,
            "PREDICTOR_PARSER_CLASS": predictor_parser_class,
            "SCRIBE_PARSER_CLASS": "lc_classification.core.parsers.elasticc_scribe_parser.ElasticcScribeParser",
            "STEP_PARSER_CLASS": "lc_classification.core.parsers.elasticc_parser.ElasticcParser",
        }
        env_variables_dict.update(extra_env_vars)
        print(env_variables_dict)
        for key, value in env_variables_dict.items():
            os.environ[key] = value

    return set_env_variables


def produce_messages(topic, SCHEMA):
    producer = KafkaProducer(
        {
            "PARAMS": {"bootstrap.servers": "localhost:9092"},
            "TOPIC": topic,
            "SCHEMA": SCHEMA,
        }
    )
    random.seed(42)
    messages = generate_many(SCHEMA, 2)
    producer.set_key_field("aid")

    for message in messages:
        for det in message["detections"]:
            det["aid"] = message["aid"]
            det["extra_fields"] = generate_extra_fields()
        message["detections"][0]["new"] = True
        message["detections"][0]["has_stamp"] = True
        producer.produce(message)


@pytest.fixture(scope="session")
def kafka_consumer():
    def factory(stream: str):
        consumer = KafkaConsumer(
            {
                "PARAMS": {
                    "bootstrap.servers": "localhost:9092",
                    "group.id": f"test_steppu{time.time()}",
                    "auto.offset.reset": "beginning",
                    "enable.partition.eof": True,
                },
                "TOPICS": [
                    get_lc_classifier_topic(stream),
                ],
                "TIMEOUT": 0,
            }
        )
        return consumer

    return factory


@pytest.fixture(scope="session")
def scribe_consumer():
    def factory():
        consumer = KafkaConsumer(
            {
                "PARAMS": {
                    "bootstrap.servers": "localhost:9092",
                    "group.id": "test_step_",
                    "auto.offset.reset": "beginning",
                    "enable.partition.eof": True,
                },
                "TOPICS": ["w_object"],
                "TIMEOUT": 0,
            }
        )
        return consumer

    return factory
