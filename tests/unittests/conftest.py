import pytest
import os


@pytest.fixture
def env_variables():
    envs = {
        "INCLUDED_CALCULATORS": "dmdt,ra,dec",
        "CONSUMER_SERVER": "localhost",
        "CONSUMER_GROUP_ID": "consumer1",
        "CONSUMER_TOPICS": "topic1",
        "CONSUMER_CLASS": "unittest.mock.MagicMock",
        "METRICS_SERVER": "localhost",
        "METRICS_TOPIC": "metrics",
        "SCRIBE_PRODUCER_TOPIC": "w_something",
        "SCRIBE_PRODUCER_CLASS": "unittest.mock.MagicMock",
        "SCRIBE_PRODUCER_SERVER": "localhost",
        "PRODUCER_SERVER": "localhost",
    }
    for env, value in envs.items():
        os.environ[env] = value
    yield
    for env in envs:
        os.environ.pop(env)
