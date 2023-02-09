from abc import abstractmethod

from apf.consumers import GenericConsumer
from apf.core import get_class

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
        if config:
            self.config = config
        else:
            self.config = {}
        self.consumer = GenericConsumer() if consumer is None else consumer
        self.commit = self.config.get("COMMIT", True)
        self.metrics = {}
        self.metrics_sender = None
        self.extra_metrics = []

        if self.config.get("METRICS_CONFIG"):
            Metrics = get_class(self.config["METRICS_CONFIG"].get("CLASS", "apf.metrics.KafkaMetricsProducer"))
            self.metrics_sender = Metrics(self.config["METRICS_CONFIG"]["PARAMS"])
            self.extra_metrics = self.config["METRICS_CONFIG"].get("EXTRA_METRICS", ["candid"])

    def send_metrics(self, **metrics):
        """Send Metrics with a metrics producer.

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
        if self.metrics_sender:
            metrics["source"] = self.__class__.__name__
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

    def get_value(self, message, params):
        """ Get values from a massage and process it to create a new metric.

        Parameters
        ----------
        message : dict
            Dict-like message to be processed

        params : str, dict
            String of the value key or dict with the following:

            - 'key': str
                Must have parameter, has to be in the message.
            - 'alias': str
                New key returned, this can be used to standarize some message keys.
            - 'format': callable
                Function to be call on the message value.

        Returns
        -------
        new_key, value
            Aliased key and processed value.

        """
        if isinstance(params, str):
            return params, message.get(params)
        elif isinstance(params, dict):
            if "key" not in params:
                raise KeyError("'key' in parameteres not found")

            val = message.get(params['key'])
            if "format" in params:
                if not callable(params["format"]):
                    raise ValueError("'format' parameter must be a calleable.")
                else:
                    val = params["format"](val)
            if "alias" in params:
                if isinstance(params["alias"], str):
                    return params["alias"], val
                else:
                    raise ValueError("'alias' parameter must be a string.")
            else:
                return params["key"], val

    def get_extra_metrics(self, message):
        """Generate extra metrics from the EXTRA_METRICS metrics configuration.

        Parameters
        ----------
        message : dict, list
            Dict-like message to be processed or list of dict-like messages

        Returns
        -------
        dict
            Dictionary with extra metrics from the messages.

        """
        # Is the message is a list then the metrics are
        # added to an array of values.
        if isinstance(message, list):
            extra_metrics = {}
            for msj in message:
                for metric in self.extra_metrics:
                    aliased_metric, value = self.get_value(msj, metric)
                    # Checking if the metric exists
                    if aliased_metric not in extra_metrics:
                        extra_metrics[aliased_metric] = []
                    extra_metrics[aliased_metric].append(value)
            extra_metrics["n_messages"] = len(message)

        # If not they are only added as a single value.
        else:
            extra_metrics = {}
            for metric in self.extra_metrics:
                aliased_metric, value = self.get_value(message, metric)
                extra_metrics[aliased_metric] = value
            extra_metrics["n_messages"] = 1
        return extra_metrics

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
            time_difference = self.metrics["timestamp_sent"] - self.metrics["timestamp_received"]
            self.metrics["execution_time"] = time_difference.total_seconds()
            if self.extra_metrics:
                extra_metrics = self.get_extra_metrics(self.message)
                self.metrics.update(extra_metrics)
            self.send_metrics(**self.metrics)
