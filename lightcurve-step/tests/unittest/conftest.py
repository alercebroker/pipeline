import pytest
import uuid
import os


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
