import pytest
import os


@pytest.fixture
def env_variables():
    envcopy = os.environ.copy()
    env_variables_dict = {
        "CONSUMER_SERVER": "localhost:9092",
        "CONSUMER_TOPICS": "sorting-hat",
        "CONSUMER_GROUP_ID": "prv-test",
        "METRICS_HOST": "localhost:9092",
        "PRODUCER_SERVER": "localhost:9092",
        "PRODUCER_TOPIC": "prv-candidates",
        "ENABLE_PARTITION_EOF": "True",
        "SCRIBE_PRODUCER_SERVER": "localhost:9092",
        "SCRIBE_PRODUCER_TOPIC": "w_non_detections",
        "SCRIBE_PRODUCER_CLASS": "unittest.mock.MagicMock",
    }
    for key in env_variables_dict:
        os.environ[key] = env_variables_dict[key]

    yield env_variables_dict
    os.environ = envcopy
