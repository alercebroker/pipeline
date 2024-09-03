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
from apf.core import get_class
from confluent_kafka.admin import AdminClient, NewTopic
from fastavro.utils import generate_many
from fastavro.schema import load_schema

from tests.mockdata.extra_felds import generate_extra_fields
from tests.mockdata.features_elasticc import features_elasticc
from tests.mockdata.features_ztf import features_ztf

INPUT_SCHEMA_PATH = pathlib.Path(
    pathlib.Path(__file__).parent.parent.parent.parent,
    "schemas/feature_step",
    "output.avsc",
)


def pytest_configure(config):
    config.addinivalue_line("markers", "ztf: mark a test as a ztf test.")
    config.addinivalue_line(
        "markers", "elasticc: mark a test as a elasticc test."
    )


@pytest.fixture(scope="session")
def docker_compose_command():
    version = os.getenv("COMPOSE", "v2")
    return "docker compose" if version == "v2" else "docker-compose"


@pytest.fixture(scope="session")
def docker_compose_file(pytestconfig):
    return (
        pathlib.Path(pytestconfig.rootdir)
        / "tests/integration/docker-compose.yml"
    ).absolute()


def get_lc_classifier_topic(model: str):
    return f"lc_classifier_{model}{datetime.utcnow().strftime('%Y%m%d')}"


def create_topics(client):
    futures = client.create_topics(
        [
            NewTopic("features_ztf", num_partitions=1),
            NewTopic("features_elasticc", num_partitions=1),
            NewTopic("w_object", num_partitions=1),
            NewTopic(get_lc_classifier_topic("ztf"), num_partitions=1),
            NewTopic(get_lc_classifier_topic("balto"), num_partitions=1),
            NewTopic(
                get_lc_classifier_topic("balto_schemaless"), num_partitions=1
            ),
            NewTopic(get_lc_classifier_topic("messi"), num_partitions=1),
            NewTopic(get_lc_classifier_topic("toretto"), num_partitions=1),
            NewTopic(get_lc_classifier_topic("barney"), num_partitions=1),
            NewTopic(get_lc_classifier_topic("mlp"), num_partitions=1),
            NewTopic("metrics", num_partitions=1),
        ]
    )
    for topic, future in futures.items():
        try:
            future.result()
        except Exception as e:
            logging.error(f"Can't create topic {topic}: {e}")
            return False
    return True


def delete_topics(client: AdminClient):
    futures = client.delete_topics(
        [
            "features_ztf",
            "features_elasticc",
            "w_object",
            "balto",
            "balto_schemaless",
            "messi",
            "toretto",
            "barney",
            "mlp",
            "metrics",
        ],
        operation_timeout=3,
        request_timeout=3,
    )
    for topic, future in futures.items():
        try:
            future.result()
        except Exception as e:
            logging.error(f"Can't delete topic {topic}: {e}")
            return False
    return True


def is_responsive_kafka(url):
    client = AdminClient({"bootstrap.servers": url})
    responsive = delete_topics(client)
    responsive = create_topics(client)
    return responsive


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
    env_copy = os.environ.copy()

    def set_env_variables():
        random_string = uuid.uuid4().hex
        step_schema_path = pathlib.Path(
            pathlib.Path(__file__).parent.parent.parent.parent,
            "schemas/lc_classification_step",
        )
        env_variables_dict = {
            "PRODUCER_SCHEMA_PATH": str(step_schema_path / "output_ztf.avsc"),
            "METRICS_SCHEMA_PATH": str(step_schema_path / "metrics.json"),
            "SCRIBE_SCHEMA_PATH": str(
                step_schema_path / "../scribe_step/scribe.avsc"
            ),
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
            "MODEL_CLASS": "lc_classifier.classifier.models.HierarchicalRandomForest",
            "STREAM": "ztf",
            "SCRIBE_PARSER_CLASS": "lc_classification.core.parsers.scribe_parser.ScribeParser",
            "STEP_PARSER_CLASS": "lc_classification.core.parsers.alerce_parser.AlerceParser",
        }
        for key, value in env_variables_dict.items():
            os.environ[key] = value

    yield set_env_variables
    os.environ = env_copy


@pytest.fixture
def env_variables_anomaly():
    env_copy = os.environ.copy()

    def set_env_variables(
        model: str,
        model_class: str,
        extra_env_vars: dict = {},
    ):
        random_string = uuid.uuid4().hex
        step_schema_path = pathlib.Path(
            pathlib.Path(__file__).parent.parent.parent.parent,
            "schemas/lc_classification_step",
        )
        env_variables_dict = {
            "PRODUCER_SCHEMA_PATH": str(step_schema_path / "output_ztf.avsc"),
            "METRICS_SCHEMA_PATH": str(step_schema_path / "metrics.json"),
            "SCRIBE_SCHEMA_PATH": str(
                step_schema_path / "../scribe_step/scribe.avsc"
            ),
            "CONSUMER_SERVER": "localhost:9092",
            "CONSUMER_TOPICS": "features_anomaly",
            "CONSUMER_GROUP_ID": random_string,
            "PRODUCER_SERVER": "localhost:9092",
            "PRODUCER_TOPIC_FORMAT": f"lc_classifier_{model}%s",
            "PRODUCER_DATE_FORMAT": "%Y%m%d",
            "PRODUCER_CHANGE_HOUR": "23",
            "PRODUCER_RETENTION_DAYS": "1",
            "SCRIBE_SERVER": "localhost:9092",
            "METRICS_HOST": "localhost:9092",
            "METRICS_TOPIC": "metrics",
            "SCRIBE_TOPIC": "w_object_anomaly",
            "CONSUME_MESSAGES": "5",
            "ENABLE_PARTITION_EOF": "True",
            "STREAM": "ztf",
            "MODEL_CLASS": model_class,
            "SCRIBE_PARSER_CLASS": "lc_classification.core.parsers.scribe_parser.ScoreScribeParser",
            "STEP_PARSER_CLASS": "lc_classification.core.parsers.anomaly_parser.AnomalyParser",
        }
        env_variables_dict.update(extra_env_vars)
        for key, value in env_variables_dict.items():
            os.environ[key] = value

    yield set_env_variables
    os.environ = env_copy


@pytest.fixture
def env_variables_squidward():
    env_copy = os.environ.copy()

    def set_env_variables(
        model: str,
        model_class: str,
        extra_env_vars: dict = {},
    ):
        random_string = uuid.uuid4().hex
        step_schema_path = pathlib.Path(
            pathlib.Path(__file__).parent.parent.parent.parent,
            "schemas/lc_classification_step",
        )
        env_variables_dict = {
            "PRODUCER_SCHEMA_PATH": str(step_schema_path / "output_ztf.avsc"),
            "METRICS_SCHEMA_PATH": str(step_schema_path / "metrics.json"),
            "SCRIBE_SCHEMA_PATH": str(
                step_schema_path / "../scribe_step/scribe.avsc"
            ),
            "CONSUMER_SERVER": "localhost:9092",
            "CONSUMER_TOPICS": "features_squidward",
            "CONSUMER_GROUP_ID": random_string,
            "PRODUCER_SERVER": "localhost:9092",
            "PRODUCER_TOPIC_FORMAT": "lc_classifier_squidward%s",
            "PRODUCER_DATE_FORMAT": "%Y%m%d",
            "PRODUCER_CHANGE_HOUR": "23",
            "PRODUCER_RETENTION_DAYS": "1",
            "SCRIBE_SERVER": "localhost:9092",
            "METRICS_HOST": "localhost:9092",
            "METRICS_TOPIC": "metrics",
            "SCRIBE_TOPIC": "w_object_squidward",
            "CONSUME_MESSAGES": "5",
            "ENABLE_PARTITION_EOF": "True",
            "STREAM": "ztf",
            "MODEL_CLASS": model_class,
            "SCRIBE_PARSER_CLASS": "lc_classification.core.parsers.scribe_parser.ScribeParser",
            "STEP_PARSER_CLASS": "lc_classification.core.parsers.anomaly_parser.AnomalyParser",
        }
        env_variables_dict.update(extra_env_vars)
        for key, value in env_variables_dict.items():
            os.environ[key] = value

    yield set_env_variables
    os.environ = env_copy


@pytest.fixture
def env_variables_mbappe():
    env_copy = os.environ.copy()

    def set_env_variables(
        model: str,
        model_class: str,
        extra_env_vars: dict = {},
    ):
        random_string = uuid.uuid4().hex
        step_schema_path = pathlib.Path(
            pathlib.Path(__file__).parent.parent.parent.parent,
            "schemas/lc_classification_step",
        )
        env_variables_dict = {
            "PRODUCER_SCHEMA_PATH": str(step_schema_path / "output_ztf.avsc"),
            "METRICS_SCHEMA_PATH": str(step_schema_path / "metrics.json"),
            "SCRIBE_SCHEMA_PATH": str(
                step_schema_path / "../scribe_step/scribe.avsc"
            ),
            "CONSUMER_SERVER": "localhost:9092",
            "CONSUMER_TOPICS": "features_mbappe",
            "CONSUMER_GROUP_ID": random_string,
            "PRODUCER_SERVER": "localhost:9092",
            "PRODUCER_TOPIC_FORMAT": "lc_classifier_mbappe%s",
            "PRODUCER_DATE_FORMAT": "%Y%m%d",
            "PRODUCER_CHANGE_HOUR": "23",
            "PRODUCER_RETENTION_DAYS": "1",
            "SCRIBE_SERVER": "localhost:9092",
            "METRICS_HOST": "localhost:9092",
            "METRICS_TOPIC": "metrics",
            "SCRIBE_TOPIC": "w_object_mbappe",
            "CONSUME_MESSAGES": "5",
            "ENABLE_PARTITION_EOF": "True",
            "STREAM": "ztf",
            "MODEL_CLASS": model_class,
            "SCRIBE_PARSER_CLASS": "lc_classification.core.parsers.scribe_parser.ScribeParser",
            "STEP_PARSER_CLASS": "lc_classification.core.parsers.anomaly_parser.AnomalyParser",
        }
        env_variables_dict.update(extra_env_vars)
        for key, value in env_variables_dict.items():
            os.environ[key] = value

    yield set_env_variables
    os.environ = env_copy


@pytest.fixture
def env_variables_elasticc():
    env_copy = os.environ.copy()

    def set_env_variables(
        model: str,
        model_class: str,
        extra_env_vars: dict = {},
    ):
        random_string = uuid.uuid4().hex
        step_schema_path = pathlib.Path(
            pathlib.Path(__file__).parent.parent.parent.parent,
            "schemas/lc_classification_step",
        )
        env_variables_dict = {
            "PRODUCER_SCHEMA_PATH": str(
                step_schema_path / "output_elasticc.avsc"
            ),
            "METRICS_SCHEMA_PATH": str(step_schema_path / "metrics.json"),
            "SCRIBE_SCHEMA_PATH": str(
                step_schema_path / "../scribe_step/scribe.avsc"
            ),
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
            "SCRIBE_PARSER_CLASS": "lc_classification.core.parsers.scribe_parser.ScribeParser",
            "STEP_PARSER_CLASS": "lc_classification.core.parsers.elasticc_parser.ElasticcParser",
        }
        env_variables_dict.update(extra_env_vars)
        for key, value in env_variables_dict.items():
            os.environ[key] = value

    yield set_env_variables
    os.environ = env_copy


@pytest.fixture
def produce_messages():
    def func(
        topic,
        force_empty_features=False,
        force_missing_features=False,
        n_forced=5,
    ):
        schema = load_schema(str(INPUT_SCHEMA_PATH))
        schema_path = INPUT_SCHEMA_PATH
        _produce_messages(
            topic,
            schema,
            schema_path,
            force_empty_features,
            force_missing_features,
            n_forced,
        )

    return func


def _produce_messages(
    topic,
    SCHEMA,
    SCHEMA_PATH,
    force_empty_features,
    force_missing_features,
    n_forced: int,
):
    BANDS = ["g", "r"]
    producer = KafkaProducer(
        {
            "PARAMS": {"bootstrap.servers": "localhost:9092"},
            "TOPIC": topic,
            "SCHEMA_PATH": SCHEMA_PATH,
        }
    )
    random.seed(42)
    messages = generate_many(SCHEMA, 2)
    producer.set_key_field("oid")

    for message in messages:
        for i, det in enumerate(message["detections"]):
            det["oid"] = message["oid"]
            det["candid"] = str(random.randint(0, 100000))
            det["extra_fields"] = generate_extra_fields()
            det["fid"] = random.choice(BANDS)
            if i < n_forced:
                det["forced"] = True
            else:
                det["forced"] = False

        message["detections"][0]["new"] = True
        message["detections"][0]["has_stamp"] = True
        if topic == "features_ztf":
            message["features"] = features_ztf(
                force_empty_features, force_missing_features
            )
        elif topic == "features_elasticc":
            message["features"] = features_elasticc(
                force_empty_features, force_missing_features
            )
        producer.produce(message)


@pytest.fixture
def kafka_consumer():
    def factory(
        stream: str,
        consumer_class="apf.consumers.kafka.KafkaConsumer",
        consumer_params={},
    ):
        Consumer = get_class(consumer_class)
        params = {
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
        params.update(consumer_params)
        consumer = Consumer(params)
        return consumer

    return factory


@pytest.fixture
def scribe_consumer():
    def factory(topic=None):
        consumer = KafkaConsumer(
            {
                "PARAMS": {
                    "bootstrap.servers": "localhost:9092",
                    "group.id": f"test_step_{time.time()}",
                    "auto.offset.reset": "beginning",
                    "enable.partition.eof": True,
                },
                "TOPICS": [topic] if topic is not None else ["w_object"],
                "TIMEOUT": 0,
            }
        )
        return consumer

    return factory
