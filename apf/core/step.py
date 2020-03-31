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

        Adding `LOGGING_DEBUG` to `settings.py` set the step's global logging level to debug.

        .. code-block:: python

            #settings.py
            LOGGING_DEBUG = True

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
        """Send Metrics to an Elasticsearch Cluster.

        For this method to work the `ES_CONFIG` variable has to be set in the `STEP_CONFIG`
        variable.

        By default if `ES_CONFIG` is set the step sends the time of the :meth:`execute()` method as `execution_time=float`
        and `source=Class`, this helps to debug the step.

        **Example:**

        Send the compute time for an object.

        .. code-block:: python

            #example_step/step.py
            self.send_metrics(compute_time=compute_time, oid=oid)

        For this to work we need to declare

        .. code-block:: python

            #settings.py
            STEP_CONFIG = {...
                "ES_CONFIG":{ #Can be a empty dictionary
                    #Optional but useful parameter
                    "INDEX_PREFIX": "ztf_pipeline",
                    # Used to generate index index_prefix+class_name+date
                    #Other optional
                }
            }

        The other optional parameters are the one passed to `Elasticsearch <https://elasticsearch-py.readthedocs.io/en/master/api.html#elasticsearch.Elasticsearch>`_ class.

        Parameters
        ----------
        **metrics : dict-like
            Parameters sended to Elasticsearch.

        """
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
            metrics = {
                "execution_time": time.time()-t0
            }
            if "candid" in self.message:
                metrics["candid"] = self.message["candid"]
            self.send_metrics(**metrics)
