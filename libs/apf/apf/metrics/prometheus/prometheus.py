from prometheus_client import Enum, Summary
from unittest.mock import MagicMock


class PrometheusMetrics:
    def __init__(self):
        self.consumed_messages = Summary(
            "consumed_messages",
            "Current number of messages consumed",
        )
        self.processed_messages = Summary(
            "processed_messages",
            "Current number of messages processed",
        )
        self.execution_time = Summary(
            "execution_time",
            "Execution time of processed batch",
        )
        # self.telescope_id = Enum(
        #     "telescope_id",
        #     "Id of the telescope",
        #     states=["ZTF", "ATLAS"],
        # )


class DefaultPrometheusMetrics(PrometheusMetrics):
    def __init__(self):
        self.consumed_messages = MagicMock()
        self.processed_messages = MagicMock()
        self.execution_time = MagicMock()
        self.telescope_id = MagicMock()
