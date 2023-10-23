from unittest import mock
from unittest.mock import patch
from lightcurve_step.step import LightcurveStep
import json


def mock_get_secret(secret_name):
    if secret_name == "sql_secret":
        return json.dumps(
            {
                "HOST": "localhost",
                "USER": "postgres",
                "PASSWORD": "postgres",
                "PORT": 5432,
                "DB_NAME": "postgres",
            }
        )
    elif secret_name == "mongo_secret":
        return json.dumps(
            {
                "HOST": "localhost",
                "USERNAME": "test_user",
                "PASSWORD": "test_password",
                "PORT": 27017,
                "DATABASE": "test_db",
                "AUTH_SOURCE": "test_db",
            }
        )
    else:
        return None


@mock.patch("credentials.get_secret")
def test_step_start(
    mock_credentials,
    produce_messages,
    mongo_service,
    env_variables,
    psql_conn,
    kafka_consumer,
):
    from scripts.run_step import step_creator

    mock_credentials.side_effect = mock_get_secret

    step_creator().start()

    consumer = kafka_consumer(["lightcurve"])
    for message in consumer.consume():
        print(message)
