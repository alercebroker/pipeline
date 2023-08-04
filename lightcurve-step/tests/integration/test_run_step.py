from unittest import mock
from lightcurve_step.step import LightcurveStep
import json


@mock.patch("credentials.get_secret")
def test_step_initialization(
    mock_credentials, kafka_service, mongo_service, env_variables
):
    from scripts.run_step import step_creator

    secret = json.dumps(
        {
            "HOST": "localhost",
            "USERNAME": "mongo",
            "PASSWORD": "mongo",
            "PORT": 27017,
            "DATABASE": "test",
        }
    )
    mock_credentials.return_value = secret

    assert isinstance(step_creator(), LightcurveStep)


@mock.patch("credentials.get_secret")
def test_step_start(mock_credentials, kafka_service, mongo_service, env_variables):
    from scripts.run_step import step_creator

    secret = json.dumps(
        {
            "HOST": "localhost",
            "USERNAME": "mongo",
            "PASSWORD": "mongo",
            "PORT": 27017,
            "DATABASE": "test",
        }
    )
    mock_credentials.return_value = secret

    step_creator().start()
