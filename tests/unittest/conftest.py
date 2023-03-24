import os


def pytest_generate_tests(metafunc):
    os.environ["INCLUDED_CALCULATORS"] = "dmdt,ra,dec"
    os.environ["CONSUMER_SERVER"] = "localhost"
    os.environ["CONSUMER_GROUP_ID"] = "consumer1"
    os.environ["CONSUMER_TOPICS"] = "topic1"
    os.environ["METRICS_HOST"] = "localhost"
    os.environ["METRICS_TOPIC"] = "metrics"
    os.environ["SCRIBE_PRODUCER_TOPIC"] = "w_something"
    os.environ["SCRIBE_PRODUCER_CLASS"] = "unittest.mock.MagicMock"
    os.environ["PRODUCER_SERVER"] = "localhost"

