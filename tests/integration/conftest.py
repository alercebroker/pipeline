import pytest
import uuid
import os


@pytest.fixture
def env_variables():
    random_string = uuid.uuid4().hex
    env_variables_dict = {
        "CONSUMER_SERVER": "localhost:9092",
        "CONSUMER_TOPICS": "sorting-hat",
        "CONSUMER_GROUP_ID": random_string,
        "METRICS_HOST": "localhost:9092",
        "METRICS_TOPIC": "metrics",
        "PRODUCER_SERVER": "localhost:9092",
        "PRODUCER_TOPIC": "prv-candidates",
        "ENABLE_PARTITION_EOF": "True",
        "SCRIBE_PRODUCER_TOPIC": "w_non_detections",
    }
    for key in env_variables_dict:
        os.environ[key] = env_variables_dict[key]

    return env_variables_dict
