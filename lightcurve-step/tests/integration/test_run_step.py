from unittest import mock
from unittest.mock import patch
from lightcurve_step.step import LightcurveStep
import json


@mock.patch("credentials.get_secret")
def test_step_initialization(
    mock_credentials, kafka_service, mongo_service, env_variables
):
    from scripts.run_step import step_creator

    def mock_get_secret(secret_name):
        if secret_name == "sql_secret":
            return json.dumps(
                {
                    "HOST": "sql_host",
                    "USER": "sql_user",
                    "PASSWORD": "sql_password",
                    "PORT": 5432,
                    "DB_NAME": "sql_db",
                }
            )
        elif secret_name == "mongo_secret":
            return json.dumps(
                {
                    "HOST": "mongo_host",
                    "USERNAME": "mongo_user",
                    "PASSWORD": "mongo_password",
                    "PORT": 27017,
                    "DATABASE": "mongo_db",
                }
            )
        else:
            return None

    mock_credentials.side_effect = mock_get_secret

    assert isinstance(step_creator(), LightcurveStep)


@mock.patch("credentials.get_secret")
def test_step_start(mock_credentials, kafka_service, mongo_service, env_variables):
    from scripts.run_step import step_creator

    def mock_get_secret(secret_name):
        if secret_name == "sql_secret":
            return json.dumps(
                {
                    "HOST": "sql_host",
                    "USER": "sql_user",
                    "PASSWORD": "sql_password",
                    "PORT": 5432,
                    "DB_NAME": "sql_db",
                }
            )
        elif secret_name == "mongo_secret":
            return json.dumps(
                {
                    "HOST": "mongo_host",
                    "USERNAME": "mongo_user",
                    "PASSWORD": "mongo_password",
                    "PORT": 27017,
                    "DATABASE": "mongo_db",
                }
            )
        else:
            return None

    mock_credentials.side_effect = mock_get_secret

    step_creator().start()
