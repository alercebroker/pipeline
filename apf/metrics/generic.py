import logging
import datetime
import json


class DateTimeEncoder(json.JSONEncoder):
    # Override the default method
    def default(self, obj):
        if isinstance(obj, (datetime.date, datetime.datetime)):
            return obj.isoformat()


class GenericMetricsProducer:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Creating {self.__class__.__name__}")

    def send_metrics(self, metrics):
        """Write metrics into a data store or other metrics system.

        Parameters
        ----------
        metrics : dict
            Metrics to be written.
        """
        pass
