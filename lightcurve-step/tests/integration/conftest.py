import os
import pytest
from apf.producers import KafkaProducer
import uuid
from fastavro.schema import load_schema
from sqlalchemy import text
from db_plugins.db.sql._connection import PsqlDatabase
from db_plugins.db.mongo._connection import MongoConnection
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
        "SKIP_MJD_FILTER": "True",
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


def populate_mongo(mongo_database):
    mongo_database["detection"].insert_one(
        {
            "candid": 987654321,
            "oid": "ZTF000llmn",
            "tid": "ztf",
            "sid": "ztf",
            "aid": "AL00XYZ00",
            "pid": 1,
            "mjd": 1,
            "fid": 1,
            "isdiffpos": -1,
            "ra": 45,
            "dec": 45,
            "e_ra": 0.1,
            "e_dec": 0.1,
            "mag": 23.1,
            "e_mag": 0.9,
            "corrected": False,
            "dubious": False,
            "has_stamp": False,
        }
    )
    mongo_database["object"].insert_one(
        {
            "aid": "AL00XYZ00",
            "oid": "ZTF000llmn",
            "ndet": 1,
            "firstmjd": 50001,
            "lastmjd": 50001,
            "deltajd": 0,
            "meanra": 45,
            "meandec": 45,
            "sigmara": 0.1,
            "sigmadec": 0.1,
        }
    )
    mongo_database["non_detection"].insert_one(
        {
            "candid": 987654321,
            "oid": "ZTF000llmn",
            "fid": 1,
            "mjd": 55000,
            "diffmaglim": 42.00,
            "aid": "AL00XYZ00",
            "tid": "ztf",
            "sid": "ztf",
        }
    )
    mongo_database["forced_photometry"].insert_one(
        {
            "_id": "ZTF000llmn_9182734",
            "oid": "ZTF000llmn",
            "mjd": 55500,
            "pid": 9182734,
            "mag": 21,
            "e_mag": 0.1,
            "fid": 1,
            "ra": 45.0,
            "dec": 45.0,
            "isdiffpos": 1,
            "corrected": False,
            "dubious": False,
            "has_stamp": False,
            "aid": "AL00XYZ00",
            "tid": "ztf",
            "sid": "ztf",
        }
    )


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
            INSERT INTO forced_photometry(oid, mjd, pid, mag, e_mag, fid, ra, dec, isdiffpos, corrected, dubious, has_stamp)
            VALUES ('ZTF000llmn', 55500, 9182734, 21, 0.1, 1, 45.0, 45.0, 1, false, false, false)
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


@pytest.fixture()
def mongo_conn(mongo_service):
    config = {
        "HOST": "localhost",
        "USERNAME": "test_user",
        "PASSWORD": "test_password",
        "PORT": 27017,
        "DATABASE": "test_db",
        "AUTH_SOURCE": "test_db",
    }
    conn = MongoConnection(config)
    conn.create_db()
    populate_mongo(conn.database)
    yield conn
    conn.drop_db()
