from abc import abstractmethod
from typing import Callable
from apf.consumers import GenericConsumer
from apf.metrics.generic import GenericMetricsProducer
from apf.producers import GenericProducer
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

    logger: logging.Logger
    config: dict
    consumer: GenericConsumer
    producer: GenericProducer
    metrics: dict
    extra_metrics: list
    metrics_sender: GenericMetricsProducer
    message: dict

    def __init__(
        self,
        consumer: Callable = GenericConsumer,
        producer: Callable = GenericProducer,
        metrics_sender: Callable = GenericMetricsProducer,
        config: dict = {},
        level: int = logging.INFO,
    ):
        self._set_logger(level)
        self.config = config
        self.consumer = self._get_consumer(consumer)(self.consumer_config)
        self.producer = self._get_producer(producer)(self.producer_config)
        self.metrics_sender = self._get_metrics_sender(metrics_sender)(
            self.metrics_config
        )
        self.metrics = {}
        self.extra_metrics = []
        if self.metrics_config:
            self.extra_metrics = self.metrics_config.get("EXTRA_METRICS", ["candid"])
        self.commit = self.config.get("COMMIT", True)

    @property
    def consumer_config(self):
        return self.config["CONSUMER_CONFIG"]

    @property
    def producer_config(self):
        return self.config.get("PRODUCER_CONFIG", {})

    @property
    def metrics_config(self):
        return self.config.get("METRICS_CONFIG")

    def _set_logger(self, level):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(level)
        self.logger.info(f"Creating {self.__class__.__name__}")

    def _get_consumer(self, default: Callable):
        if self.consumer_config:
            Consumer = default
            if "CLASS" in self.consumer_config:
                Consumer = get_class(self.consumer_config["CLASS"])
            return Consumer
        raise Exception("Could not find CONSUMER_CONFIG in the step config")

    def _get_producer(self, default: Callable):
        Producer = default
        if "CLASS" in self.producer_config:
            Producer = get_class(self.producer_config["CLASS"])
        return Producer

    def _get_metrics_sender(self, default: Callable):
        Metrics = default
        if self.metrics_config:
            if "CLASS" in self.metrics_config:
                Metrics = get_class(self.config["METRICS_CONFIG"].get("CLASS"))
        return Metrics

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

    def _pre_consume(self):
        self.logger.info("Starting step. Begin processing")
        self.pre_consume()

    @abstractmethod
    def pre_consume(self):
        pass

    def _pre_execute(self):
        self.logger.info("Received message. Begin preprocessing")
        self.metrics["timestamp_received"] = datetime.datetime.now(
            datetime.timezone.utc
        )
        preprocessed = self.pre_execute(self.message)
        return preprocessed or self.message

    @abstractmethod
    def pre_execute(self, message):
        pass

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

    def _post_execute(self, result):
        self.logger.info("Processed message. Begin post processing")
        final_result = self.post_execute(result)
        self.metrics["timestamp_sent"] = datetime.datetime.now(datetime.timezone.utc)
        time_difference = (
            self.metrics["timestamp_sent"] - self.metrics["timestamp_received"]
        )
        self.metrics["execution_time"] = time_difference.total_seconds()
        if self.extra_metrics:
            extra_metrics = self.get_extra_metrics(self.message)
            self.metrics.update(extra_metrics)
        self.send_metrics(**self.metrics)
        return final_result

    @abstractmethod
    def post_execute(self, result):
        return result

    def _pre_produce(self, result):
        self.logger.info("Finished all processing. Begin message production")
        message_to_produce = self.pre_produce(result)
        return message_to_produce

    @abstractmethod
    def pre_produce(self, result):
        return result

    def _post_produce(self):
        self.logger.info("Message produced. Begin post production")
        self.post_produce()

    @abstractmethod
    def post_produce(self):
        pass

    def get_value(self, message, params):
        """Get values from a massage and process it to create a new metric.

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

            val = message.get(params["key"])
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
        self._pre_consume()
        for self.message in self.consumer.consume():
            preprocessed_msg = self._pre_execute()
            result = self.execute(preprocessed_msg)
            result = self._post_execute(result)
            result = self._pre_produce(result)
            self.producer.produce(result)
            self._post_produce()
        self._tear_down()

    def _tear_down(self):
        self.logger.info("Processing finished. No more messages. Begin tear down.")
        self.tear_down()
        self._write_success()

    def _write_success(self):
        f = open("__SUCCESS__", "w")
        f.close()

    def tear_down(self):
        pass
