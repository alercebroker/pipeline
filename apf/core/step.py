from abc import abstractmethod

from apf.consumers import GenericConsumer

import time
import logging
import datetime
from elasticsearch import Elasticsearch


class GenericStep():
    """Generic Step for apf.

    Parameters
    ----------
    consumer : :class:`GenericConsumer`
        An object of type GenericConsumer.
    level : logging.level
        Logging level, has to be a logging.LEVEL constant.
    **step_args : dict
        Additional parameters for the step.
    """
    def __init__(self,consumer = None, level = logging.INFO,config=None, **step_args):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Creating {self.__class__.__name__}")
        self.config = config
        self.consumer = GenericConsumer() if consumer is None else consumer
        self.metrics = None
        self.commit = self.config.get("COMMIT", True)

        if "ES_CONFIG" in config:
            logging.getLogger("elasticsearch").setLevel(logging.WARNING)
            self.logger.info("Creating ES Metrics sender")
            self.metrics = Elasticsearch(**config["ES_CONFIG"])

    def send_metrics(self,**metrics):
        if self.metrics:
            date = datetime.datetime.utcnow().strftime("%Y%m%d")
            index_prefix = self.config["ES_CONFIG"].get("INDEX_PREFIX", "pipeline")
            self.index = f"{index_prefix}-{self.__class__.__name__.lower()}-{date}"
            metrics["@timestamp"] = datetime.datetime.utcnow()
            metrics["source"] = self.__class__.__name__
            self.metrics.index(index = self.index,body=metrics)

    @abstractmethod
    def execute(self, message):
        """Execute the logic of the step. This method has to be implemented by
        the instanced class.

        Parameters
        ----------
        message : dict
            Dict-like message to be processed.
        """
        pass

    def start(self):
        """Start running the step.
        """
        for self.message in self.consumer.consume():
            t0 = time.time()
            self.execute(self.message)
            if self.commit:
                self.consumer.commit()
            self.send_metrics(execution_time = time.time()-t0)
