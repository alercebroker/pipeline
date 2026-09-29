"""Local stand-ins for the two things this step cannot run without.

The step consumes the multisurvey feature_step's topic and reads `classifier` /
`taxonomy` at startup, so a local run needs a broker with a populated features
topic and a seeded database. `docker-compose.yml` supplies both; the fixtures
here create the topics, create and seed the two tables, and produce synthetic
messages (see `fake_features`).

Nothing here talks to quimal: the whole harness is localhost, and the seed is
the one in `taxonomy_seed`, not the live one.
"""
import logging
import os
import pathlib
import uuid

import pytest
from confluent_kafka.admin import AdminClient, NewTopic
from sqlalchemy import create_engine, insert

from . import taxonomy_seed
from .fake_features import SCHEMA_PATH
from .fake_features import generate_messages

# Same file the messages are generated from, so the consumer can never decode
# them against a different schema than the producer wrote.
INPUT_SCHEMA_PATH = str(SCHEMA_PATH)

# The prod names, kept as prefixes so a topic here is recognisable, but each
# test gets its own suffixed pair: the assertions count every message on the
# scribe topic, so two tests sharing one topic would count each other's. Deleting
# and recreating between tests is the alternative, and Kafka topic deletion is
# asynchronous enough to make that flaky.
INPUT_TOPIC_PREFIX = "ztf-features"
SCRIBE_TOPIC_PREFIX = "scribe-multisurvey"

SCHEMAS = pathlib.Path(pathlib.Path(__file__).parent.parent.parent.parent, "schemas")
SCRIBE_SCHEMA_PATH = str(SCHEMAS / "scribe_step" / "scribe.avsc")

# Fixed, not `docker_ip:port_for(...)`: the broker advertises itself as
# 127.0.0.1:9092 (KAFKA_CFG_ADVERTISED_LISTENERS), so a client redirected to a
# mapped port would be handed that address anyway and fail. Every step in this
# repo pins 9092 for the same reason. The cost is that a leftover container from
# a crashed run holds the port and the next run cannot bind it -- `docker rm -f
# $(docker ps -aq --filter name=pytest)` clears that.
KAFKA_SERVER = "localhost:9092"

# Matches docker-compose.yml. `SCHEMA` is deliberately absent: the tables are
# created in `public`, so the step must not set a search_path (see db.py).
PSQL_CONFIG = {
    "HOST": "localhost",
    "PORT": 5432,
    "DB_NAME": "postgres",
    "USER": "postgres",
    "PASSWORD": "postgres",
}

DB_URL = "postgresql://postgres:postgres@localhost:5432/postgres"


@pytest.fixture(scope="session")
def docker_compose_file(pytestconfig):
    return os.path.join(
        str(pytestconfig.rootdir), "tests/integration", "docker-compose.yml"
    )


@pytest.fixture(scope="session")
def docker_compose_command():
    version = os.getenv("COMPOSE", "v2")
    return "docker compose" if version == "v2" else "docker-compose"


def _is_kafka_responsive(server: str) -> bool:
    try:
        AdminClient({"bootstrap.servers": server}).list_topics(timeout=5)
        return True
    except Exception as error:
        logging.debug(f"kafka not ready: {error}")
        return False


@pytest.fixture(scope="session")
def kafka_service(docker_ip, docker_services):
    port = docker_services.port_for("kafka", 9092)
    server = f"{docker_ip}:{port}"
    docker_services.wait_until_responsive(
        timeout=60.0, pause=1, check=lambda: _is_kafka_responsive(server)
    )
    return server


@pytest.fixture
def topics(kafka_service):
    """A fresh (input, scribe) topic pair for one test, created up front.

    Created rather than left to auto-creation so the step's consumer never races
    a topic into existence and reports EOF on nothing.
    """
    suffix = uuid.uuid4().hex[:8]
    names = (f"{INPUT_TOPIC_PREFIX}-{suffix}", f"{SCRIBE_TOPIC_PREFIX}-{suffix}")

    client = AdminClient({"bootstrap.servers": KAFKA_SERVER})
    futures = client.create_topics([NewTopic(name, num_partitions=1) for name in names])
    for topic, future in futures.items():
        try:
            future.result()
        except Exception as error:
            pytest.fail(f"Can't create topic {topic}: {error}")

    return names


def _is_psql_responsive() -> bool:
    try:
        create_engine(DB_URL).connect().close()
        return True
    except Exception as error:
        logging.debug(f"psql not ready: {error}")
        return False


@pytest.fixture(scope="session")
def psql_service(docker_ip, docker_services):
    """Postgres, with `classifier` and `taxonomy` created and seeded.

    Only those two tables: they are the only ones the step reads, and it writes
    nothing (the scribe owns the probability upsert). Their DDL comes from the
    db-plugins models rather than hand-written SQL, so the harness cannot drift
    from the real schema.
    """
    port = docker_services.port_for("postgres", 5432)
    docker_services.wait_until_responsive(
        timeout=60.0, pause=1, check=_is_psql_responsive
    )

    from db_plugins.db.sql.models_pipeline import Base, Classifier, Taxonomy

    engine = create_engine(DB_URL)
    Base.metadata.create_all(
        engine, tables=[Classifier.__table__, Taxonomy.__table__]
    )
    with engine.begin() as connection:
        connection.execute(insert(Classifier), taxonomy_seed.classifier_rows())
        connection.execute(insert(Taxonomy), taxonomy_seed.taxonomy_rows())
    engine.dispose()

    return f"{docker_ip}:{port}"


@pytest.fixture
def produce_features(topics):
    """Produce `count` synthetic feature_step messages, return what was sent."""
    from apf.producers import KafkaSchemalessProducer

    def _produce(count: int = 5, **kwargs) -> list:
        messages = generate_messages(count, **kwargs)
        producer = KafkaSchemalessProducer(
            {
                "TOPIC": topics[0],
                "PARAMS": {"bootstrap.servers": KAFKA_SERVER},
                "SCHEMA_PATH": INPUT_SCHEMA_PATH,
            }
        )
        for message in messages:
            producer.produce(message)
        producer.producer.flush()
        return messages

    return _produce


@pytest.fixture
def scribe_consumer(topics):
    """Consumer over the scribe topic that stops at EOF instead of blocking.

    Every consumer handed out is closed at teardown. Not housekeeping: an open
    librdkafka client keeps its broker threads running, and once pytest-docker
    has torn the containers down they spin retrying a broker that no longer
    exists, which wedges the interpreter in `Py_FinalizeEx` -- the run finishes
    and the process never exits.
    """
    from apf.consumers import KafkaSchemalessConsumer

    created = []

    def _consumer(num_messages: int = 500):
        consumer = KafkaSchemalessConsumer(
            {
                "TOPICS": [topics[1]],
                "SCHEMA_PATH": SCRIBE_SCHEMA_PATH,
                "PARAMS": {
                    "bootstrap.servers": KAFKA_SERVER,
                    "group.id": uuid.uuid4().hex,
                    "auto.offset.reset": "beginning",
                    "enable.partition.eof": True,
                },
                "consume.messages": num_messages,
                "consume.timeout": 10,
            }
        )
        created.append(consumer)
        return consumer

    yield _consumer

    for consumer in created:
        # Closing is what stops the broker threads, and it must happen: an open
        # client outlives the containers pytest-docker tears down and wedges the
        # interpreter in `Py_FinalizeEx` retrying a broker that is gone.
        #
        # The cost is a cosmetic `RuntimeError: Consumer closed`: apf's `__del__`
        # calls `teardown()`, which unsubscribes a consumer that is now closed.
        # Shadowing `teardown` here silences it during the session but not at
        # interpreter shutdown, where the instance dict is already cleared and
        # the class method runs regardless. A hang is worse than a warning.
        consumer.teardown = lambda: None
        consumer.consumer.close()


@pytest.fixture
def step_config(topics):
    """The step's config, pointed at the local broker and database.

    `MODEL_PATH` is the one value nothing here can supply: the BHRF 2.1.0 pickle
    is not in this repo and is not downloadable from CI. Tests that need it skip
    when it is unset.
    """

    def _config(**overrides) -> dict:
        config = {
            "CONSUMER_CONFIG": {
                "CLASS": "apf.consumers.KafkaSchemalessConsumer",
                "TOPICS": [topics[0]],
                "SCHEMA_PATH": INPUT_SCHEMA_PATH,
                "PARAMS": {
                    "bootstrap.servers": KAFKA_SERVER,
                    "group.id": uuid.uuid4().hex,
                    "auto.offset.reset": "beginning",
                    "enable.partition.eof": True,
                },
                "consume.messages": 50,
                "consume.timeout": 10,
            },
            "SCRIBE_PRODUCER_CONFIG": {
                "CLASS": "apf.producers.KafkaSchemalessProducer",
                "TOPIC": topics[1],
                "SCHEMA_PATH": SCRIBE_SCHEMA_PATH,
                "PARAMS": {"bootstrap.servers": KAFKA_SERVER},
            },
            "PSQL_CONFIG": dict(PSQL_CONFIG),
            "MODEL_CONFIG": {
                "CLASS": "alerce_classifiers.squidward.model.SquidwardFeaturesClassifier",
                "CLASS_MAPPER": "alerce_classifiers.squidward.mapper.SquidwardMapper",
                "PARAMS": {"model_path": os.getenv("MODEL_PATH")},
                "VERSION": taxonomy_seed.CLASSIFIER_VERSION,
                "CLASSIFIER_NAME": "lc_classifier_BHRF_forced_phot",
                "SID": 0,
                "MIN_DETECTIONS": None,
            },
            # No downstream Avro schema yet (design §9): apf falls back to its
            # DefaultProducer and nothing is produced downstream.
            "PRODUCER_CONFIG": {},
            "METRICS_CONFIG": {},
        }
        config.update(overrides)
        return config

    return _config
