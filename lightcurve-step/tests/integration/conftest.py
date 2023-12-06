import os
import pytest
from apf.producers import KafkaProducer
import uuid
from fastavro.schema import load_schema
from sqlalchemy import text
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


@pytest.fixture
def env_variables():
    envcopy = os.environ.copy()
    random_string = uuid.uuid4().hex
    env_variables_dict = {
        "PRODUCER_SCHEMA_PATH": "../schemas/lightcurve_step/output.avsc",
        "CONSUMER_SCHEMA_PATH": "",
        "METRICS_SCHEMA_PATH": "../schemas/lightcurve_step//metrics.json",
        "SCRIBE_SCHEMA_PATH": "../schemas/scribe.avsc",
        "CONSUMER_SERVER": "localhost:9092",
        "CONSUMER_TOPICS": "correction",
        "CONSUMER_GROUP_ID": random_string,
        "METRICS_SERVER": "localhost:9092",
        "PRODUCER_SERVER": "localhost:9092",
        "PRODUCER_TOPIC": "lightcurve",
        "ENABLE_PARTITION_EOF": "True",
        "MONGO_SECRET_NAME": "mongo_secret",
        "SQL_SECRET_NAME": "sql_secret",
        "CONSUME_MESSAGES": "10",
    }
    for key in env_variables_dict:
        os.environ[key] = env_variables_dict[key]

    yield env_variables_dict
    os.environ = envcopy


@pytest.fixture()
def produce_messages(kafka_service):
    def _produce(topic: str):
        schema = load_schema("tests/integration/input_schema.avsc")
        producer = KafkaProducer(
            {
                "PARAMS": {"bootstrap.servers": "localhost:9092"},
                "TOPIC": topic,
                "SCHEMA_PATH": os.path.join(
                    os.path.dirname(__file__), "input_schema.avsc"
                ),
            }
        )
        messages = generate_message(schema, 10)
        producer.set_key_field("aid")

        for message in messages:
            producer.produce(message)
        producer.producer.flush()

    return _produce


def populate_sql(conn: PsqlDatabase):
    with conn.session() as session:
        session.execute(
            text(
                """
            INSERT INTO object(oid, ndet, firstmjd, g_r_max, g_r_mean_corr, meanra, meandec, step_id_corr, \
                lastmjd, deltajd, ncovhist, ndethist, corrected, stellar)
                VALUES ('ZTF000llmn', 1, 50001, 1.0, 0.9, 45, 45, 'v1', 50001, 0, 1, 1, false, false) ON CONFLICT DO NOTHING
        """
            )
        )
        session.execute(
            text(
                """
            INSERT INTO detection(candid, oid, mjd, fid, pid, diffmaglim, isdiffpos, \
            ra, dec, magpsf, sigmapsf, corrected, dubious, has_stamp, step_id_corr) 
            VALUES (987654321, 'ZTF000llmn', 1, 1, 1, 0.8, -1, 45, 45, 23.1, 0.9, \
                    false, false, false, 'step')
        """
            )
        )
        session.execute(
            text(
                """
            INSERT INTO non_detection(oid, fid, mjd, diffmaglim) VALUES ('ZTF000llmn', 1, 55000, 42.00)
            """
            )
        )
        session.execute(
            text(
                """
            INSERT INTO forced_photometry(oid, mjd, pid, fid, ra, dec, isdiffpos, corrected, dubious, has_stamp)
            VALUES ('ZTF000llmn', 55500, 9182734, 1, 45.0, 45.0, 1, false, false, false)
            """
            )
        )
        session.commit()


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
    populate_sql(psql_conn)
    yield psql_conn
    psql_conn.drop_db()
