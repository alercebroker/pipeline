import os
import pytest
from apf.producers import KafkaProducer
from confluent_kafka.admin import AdminClient
import uuid
from fastavro.schema import load_schema
from sqlalchemy.dialects.postgresql import insert
from db_plugins.db.sql._connection import PsqlDatabase
from db_plugins.db.mongo._connection import MongoConnection
from db_plugins.db.sql.models import (
    Object,
    Detection,
    NonDetection,
    ForcedPhotometry,
)
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


# cambiar a los valores del diccionario env_variables_dict a los correspondientes de delight step

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
        "DELIGHT_FILTER_CLASS":
        "DELIGHT_FILTER_CLASSIFIER":
        "DELIGHT_FILTER_PROBABILITY":
        "DELIGHT_CALC_DISPERSION":
        "DELIGHT_CALC_GAL_SIZE":
    }
    for key in env_variables_dict:
        os.environ[key] = env_variables_dict[key]

    yield env_variables_dict
    os.environ = envcopy


@pytest.fixture()
def produce_messages(kafka_service):
    topic_to_delete = ""

    def _produce(topic: str, nmessages: int, detections: list = [], oids=[]):
        nonlocal topic_to_delete
        topic_to_delete = topic
        schema_path = "../schemas/prv_candidate_step/output.avsc" #cambiarlo por el squema de correction 
        producer = KafkaProducer(
            {
                "PARAMS": {"bootstrap.servers": "localhost:9092"},
                "TOPIC": topic,
                "SCHEMA_PATH": schema_path,
            }
        )
        schema = load_schema(schema_path)
        messages = generate_message(schema, nmessages, detections, oids)
        producer.set_key_field("oid")

        for message in messages:
            producer.produce(message)
        producer.producer.flush()
        del producer

    yield _produce
    admin_client = AdminClient({"bootstrap.servers": "localhost:9092"})
    admin_client.delete_topics(
        [topic_to_delete], operation_timeout=30, request_timeout=30
    )[topic_to_delete].result()




#crear funcion para insertar probability 
    
@pytest.fixture
def insert_object():
    def _populate(
        object: dict,
        *,
        sql: PsqlDatabase = None,
        mongo: MongoConnection = None,
    ):
        if sql:
            with sql.session() as session:
                statement = (
                    insert(Object)
                    .values(
                        oid=object["oid"],
                        ndet=1,
                        firstmjd=50001,
                        g_r_max=1.0,
                        g_r_mean_corr=0.9,
                        meanra=45,
                        meandec=45,
                        step_id_corr="v1",
                        lastmjd=50001,
                        deltajd=0,
                        ncovhist=1,
                        ndethist=1,
                        corrected=False,
                        stellar=False,
                    )
                    .on_conflict_do_nothing()
                )
                session.execute(statement)
                session.commit()
        if mongo:
            mongo.database["object"].insert_one(
                {
                    # "aid": "AL00XYZ00",
                    "aid": object["aid"],
                    "oid": object["oid"],
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

    return _populate




@pytest.fixture
def insert_non_detection():
    def _populate(
        non_detection: dict,
        *,
        sql: PsqlDatabase = None,
        mongo: MongoConnection = None,
    ):
        if sql:
            with sql.session() as session:
                statement = insert(NonDetection).values(
                    oid=non_detection["oid"],
                    fid=1,
                    mjd=55000,
                    diffmaglim=42.00,
                )
                session.execute(statement)
                session.commit()
        if mongo:
            mongo.database["non_detection"].insert_one(
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

    return _populate




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
    yield conn
    conn.drop_db()
