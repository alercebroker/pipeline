from abc import abstractmethod

from apf.consumers import GenericConsumer
from apf.metrics import KafkaMetricsProducer
from apf.core import get_class

import time
import logging
import datetime


class GenericStep:
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

    def __init__(self, consumer=None, level=logging.INFO, config=None, **step_args):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Creating {self.__class__.__name__}")
        self.config = config
        self.consumer = GenericConsumer() if consumer is None else consumer
        self.commit = self.config.get("COMMIT", True)
        self.metrics = {}
        if self.config.get("METRICS_CONFIG"):
            Metrics = get_class(self.config["METRICS_CONFIG"].get("CLASS", KafkaMetricsProducer))
        self.metrics_sender = Metrics(self.config["METRICS_CONFIG"]["PARAMS"])

    def send_metrics(self, **metrics):
        """Send Metrics to an Kafka topic.

        For this method to work the `METRICS_CONFIG` variable has to be set in the `STEP_CONFIG`
        variable.

        **Example:**

        Send the compute time for an object.

        .. code-block:: python

            #example_step/step.py
            self.send_metrics(compute_time=compute_time, oid=oid)

        For this to work we need to declare

        .. code-block:: python

            #settings.py
            STEP_CONFIG = {...
                "METRICS_CONFIG":{ #Can be a empty dictionary
                    "CLASS": "apf.metrics.KafkaMetricsProducer",
                    "PARAMS": { # params for the apf.metrics.KafkaMetricsProducer
                        "PARAMS":{
                            ## this producer uses confluent_kafka.Producer, so here we provide
                            ## arguments for that class, like bootstrap.servers
                            bootstrap.servers": "kafka1:9092",
                        },
                        "TOPIC": "metrics_topic" # the topic to store the metrics
                    },
                }
            }

        Parameters
        ----------
        **metrics : dict-like
            Parameters sent to the kafka topic as message.

        """
        self.metrics_sender.send_metrics(metrics)

    @abstractmethod
    def execute(self, message):
        """Execute the logic of the step. This method has to be implemented by
        the instanced class.

        Parameters
        ----------
        message : dict, list
            Dict-like message to be processed or list of dict-like messages
        """
        pass

    def start(self):
        """Start running the step."""
        for self.message in self.consumer.consume():
            self.metrics["timestamp_received"] = datetime.datetime.now(
                datetime.timezone.utc
            )
            self.execute(self.message)
            if self.commit:
                self.consumer.commit()
            self.metrics["timestamp_sent"] = datetime.datetime.now(
                datetime.timezone.utc
            )
            if "candid" in self.message:
                self.metrics["candid"] = str(self.message["candid"])
                self.send_metrics(**self.metrics)
