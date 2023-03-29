from unittest import mock
from lightcurve_step.step import LightcurveStep


@mock.patch("settings.get_mongodb_credentials")
def test_step_initialization(mock_credentials, kafka_service, mongo_service, env_variables):
    from scripts.run_step import step_creator
    mock_credentials.return_value = {
        "HOST": "localhost",
        "USERNAME": "mongo",
        "PASSWORD": "mongo",
        "PORT": 27017,
        "DATABASE": "test",
    }

    assert isinstance(step_creator(), LightcurveStep)


@mock.patch("settings.get_mongodb_credentials")
def test_step_start(mock_credentials, kafka_service, mongo_service, env_variables):
    from scripts.run_step import step_creator
    mock_credentials.return_value = {
        "HOST": "localhost",
        "USERNAME": "mongo",
        "PASSWORD": "mongo",
        "PORT": 27017,
        "DATABASE": "test",
    }

    step_creator().start()
