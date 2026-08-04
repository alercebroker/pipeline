import abc
import datetime
import logging
from abc import abstractmethod
from typing import Any, Dict, Iterable, List, Type, Union
import os

from apf.consumers import GenericConsumer
from apf.core import get_class
from apf.metrics.generic import GenericMetricsProducer
from apf.metrics.prometheus import DefaultPrometheusMetrics, PrometheusMetrics
from apf.metrics.pyroscope import profile
from apf.producers import GenericProducer


class DefaultConsumer(GenericConsumer):
    def consume(self):
        yield {}


class DefaultProducer(GenericProducer):
    def produce(self, message, **kwargs):
        pass


class DefaultMetricsProducer(GenericMetricsProducer):
    def send_metrics(self, metrics):
        pass


class GenericStep(abc.ABC):
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

    def __init__(
        self,
        consumer: Type[GenericConsumer] = DefaultConsumer,
        producer: Type[GenericProducer] = DefaultProducer,
        metrics_sender: Type[GenericMetricsProducer] = DefaultMetricsProducer,
        level: int = logging.NOTSET,
        config: dict = {},
        prometheus_metrics: PrometheusMetrics = DefaultPrometheusMetrics(),
    ):
        self._set_logger(level)
        self.config = config
        self.consumer = self._get_consumer(consumer)(self.consumer_config)
        self.producer = self._get_producer(producer)(self.producer_config)
        self.metrics_sender = self._get_metrics_sender(metrics_sender)(
            self.metrics_producer_params
        )
        self.metrics = {}
        self.extra_metrics = []
        if self.metrics_config:
            self.extra_metrics = self.metrics_config.get("EXTRA_METRICS", ["candid"])
        self.commit = self.config.get("COMMIT", True)
        self.prometheus_metrics = prometheus_metrics

    @property
    def consumer_config(self):
        return self.config["CONSUMER_CONFIG"]

    @property
    def producer_config(self):
        return self.config.get("PRODUCER_CONFIG", {})

    @property
    def metrics_config(self):
        return self.config.get("METRICS_CONFIG")

    @property
    def metrics_producer_params(self) -> dict:
        if self.metrics_config:
            return self.metrics_config["PARAMS"]
        return {}

    def _set_logger(self, level):
        self.logger = logging.getLogger(f"alerce.{self.__class__.__name__}")
        if level != logging.NOTSET:
            self.logger.setLevel(level)
        self.logger.info(f"Creating {self.__class__.__name__}")

    def _get_consumer(self, default: Type[GenericConsumer]) -> Type[GenericConsumer]:
        if self.consumer_config:
            Consumer = default
            if "CLASS" in self.consumer_config:
                Consumer = get_class(self.consumer_config["CLASS"])
            return Consumer
        raise Exception("Could not find CONSUMER_CONFIG in the step config")

    def _get_producer(self, default: Type[GenericProducer]) -> Type[GenericProducer]:
        Producer = default
        if "CLASS" in self.producer_config:
            Producer = get_class(self.producer_config["CLASS"])
        return Producer

    def _get_metrics_sender(
        self, default: Type[GenericMetricsProducer]
    ) -> Type[GenericMetricsProducer]:
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
            self.metrics_sender.send_metrics(metrics)

    def _pre_consume(self):
        self.logger.info("Starting step. Begin processing")
        self.pre_consume()

    def pre_consume(self):
        """
        Override this method to perform operations before the first message arrives.
        """
        pass

    def _pre_execute(self, message: Union[dict, List[dict]]):
        self.logger.info("Received message. Begin preprocessing")
        self.metrics["timestamp_received"] = datetime.datetime.now(
            datetime.timezone.utc
        )
        if isinstance(message, dict):
            message = [message]
        self.message = message
        if isinstance(self.message, dict):
            self.prometheus_metrics.consumed_messages.observe(1)
        if isinstance(self.message, list):
            self.prometheus_metrics.consumed_messages.observe(len(self.message))
        try:
            preprocessed = self.pre_execute(self.message)
        except Exception as error:
            self.logger.debug("Error at pre_execute")
            self.logger.debug(f"The message(s) that caused the error: {message}")
            raise error
        return preprocessed

    def pre_execute(self, messages: List[dict]):
        """
        Override this method to perform operations on each batch of messages consumed.

        Typically this method is used for pre processing operations such as parsing,
        formatting and overall preparation for the execute method that handles
        all the complex logic applied to the messages.
        """
        return messages

    @abstractmethod
    def execute(
        self, messages: List[dict]
    ) -> Union[Iterable[Dict[str, Any]], Dict[str, Any]]:
        """Execute the logic of the step. This method has to be implemented by
        the instanced class.

        Parameters
        ----------
        message : dict, list
            Dict-like message to be processed or list of dict-like messages
        """
        pass

    def _post_execute(self, result: Union[Iterable[Dict[str, Any]], Dict[str, Any]]):
        self.logger.info("Processed message. Begin post processing")
        try:
            final_result = self.post_execute(result)
        except Exception as error:
            self.logger.debug("Error at post_execute")
            self.logger.debug(f"The result that caused the error: {result}")
            raise error
        self.metrics["timestamp_sent"] = datetime.datetime.now(datetime.timezone.utc)
        time_difference = (
            self.metrics["timestamp_sent"] - self.metrics["timestamp_received"]
        )
        self.metrics["execution_time"] = time_difference.total_seconds()
        if self.extra_metrics:
            extra_metrics = self.get_extra_metrics(self.message)
            self.metrics.update(extra_metrics)
        if "source" not in self.metrics:
            self.metrics["source"] = os.getenv(
                "METRICS_SOURCE", f"{self.__class__.__name__}"
            )
        if "survey" not in self.metrics:
            self.metrics["survey"] = os.getenv("METRICS_SURVEY")
        self.send_metrics(**self.metrics)
        if isinstance(self.message, dict):
            self.prometheus_metrics.processed_messages.observe(1)
        if isinstance(self.message, list):
            self.prometheus_metrics.processed_messages.observe(len(self.message))
        self.prometheus_metrics.execution_time.observe(time_difference.total_seconds())
        return final_result

    def post_execute(self, result: Union[Iterable[Dict[str, Any]], Dict[str, Any]]):
        """
        Override this method to perform additional operations on
        the processed data coming from :py:func:`apf.core.step.GenericStep.execute`
        method.

        Typically used to do post processing, parsing, output formatting, etc.
        """
        return result

    def _pre_produce(
        self, result: Union[Iterable[Dict[str, Any]], Dict[str, Any]]
    ) -> Union[Iterable[Dict[str, Any]], Dict[str, Any]]:
        self.logger.info("Finished all processing. Begin message production")
        try:
            message_to_produce = self.pre_produce(result)
        except Exception as error:
            self.logger.debug("Error at pre_produce")
            self.logger.debug(f"The result that caused the error: {result}")
            raise error
        return message_to_produce

    def pre_produce(self, result: Union[Iterable[Dict[str, Any]], Dict[str, Any]]):
        """
        Override this method to perform additional operations on
        the processed data coming from :py:func:`apf.core.step.GenericStep.post_execute`
        method.

        Typically used to format data output as described in the step producer's Schema
        """
        return result

    def _post_produce(self):
        self.logger.info("Messages produced. Begin post production")
        try:
            self.post_produce()
            if self.commit:
                self.consumer.commit()
        except Exception as error:
            self.logger.debug("Error at post_produce")
            raise error

    def post_produce(self):
        """
        Override this method to perform operations after data has been
        produced by the producer.

        You can use this lifecycle method to perform cleanup, send additional metrics,
        notifications, etc.
        """
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
            if val is None or "value" in params:
                # returns None, so val will remain None or take the value of params["value"]
                val = params.get("value")

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
        else:
            raise TypeError(f"params must be str or dict, received {type(params)}")

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
                    if isinstance(value, list):
                        extra_metrics[aliased_metric].extend(value)
                    else:
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

    def produce(self, result: Union[Iterable[Dict[str, Any]], Dict[str, Any]]):
        """
        Produces messages using the configured producer class.

        Parameters
        ----------
        result: dict | list[dict]
            The result of the step's execution.
            This parameter can be an iterable or a single message, where the message should be
            a dictionary that matches the output schema of the step.

        NOTE: If you want to produce with a key, use the set_producer_key_field(key_field)
        method somewhere in the lifecycle of the step prior to the produce state.
        """
        n_messages = 0
        if isinstance(result, dict):
            to_produce = [result]
        else:
            to_produce = result
        count = 0
        for prod_message in to_produce:
            count += 1
            flush = count == len(to_produce)
            self.producer.produce(prod_message, flush=flush)
        if not isinstance(self.producer, DefaultProducer):
            self.logger.info(f"Produced {count} messages")

    def set_producer_key_field(self, key_field: str):
        """
        Set the producer key, used in producer.produce(message, key=message[key_field])

        Parameters
        ----------
        key_field : str
            the key of the message which value will be used as key for the producer
        """
        self.producer.set_key_field(key_field)

    @profile
    def start(self):
        logger = logging.getLogger(f"alerce.{self.__class__.__name__}")
        """Start running the step."""
        self._pre_consume()
        for message in self.consumer.consume():
            preprocessed_msg = self._pre_execute(message)
            if len(preprocessed_msg) == 0:
                logger.info("Message of len zero after pre_execute")
                continue
            try:
                result = self.execute(preprocessed_msg)
            except Exception as error:
                logger.debug("Error at execute")
                logger.debug(f"The message(s) that caused the error: {message}")
                raise error
            result = self._post_execute(result)
            result = self._pre_produce(result)
            self.produce(result)
            self._post_produce()
        self._tear_down()

    def _tear_down(self):
        self.logger.info("Processing finished. No more messages. Begin tear down.")
        try:
            self.tear_down()
        except Exception as error:
            self.logger.debug("Error at tear_down")
            raise error
        self._write_success()

    def _write_success(self):
        f = open("__SUCCESS__", "w")
        f.close()

    def tear_down(self):
        """
        Override this method to perform operations after the consumer
        has stopped consuming data.

        This method is called only once after processing messages and right before
        the start method ends.

        You can use this lifecycle method to perform cleanup, send additional metrics,
        notifications, etc.
        """
        pass
