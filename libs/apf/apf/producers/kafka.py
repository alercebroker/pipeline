from apf.producers.generic import GenericProducer
from confluent_kafka import Producer
import fastavro
import io
import importlib


class KafkaProducer(GenericProducer):
    """Kafka Single Topic Producer.

    Parameters
    ----------
    PARAMS: dict
        Parameters passed to :class:`confluent_kafka.Producer`

        The required parameters are:

        - *bootstrap.servers*: comma separated <host:port> :class:`string` to brokers.

    TOPIC: string
        Kafka fixed output topic.

        *Example:*

        Depending on the step configuration the producer config can be passsed in different ways, the recommended one
        is passing it on the `STEP_CONFIG` variable.

        .. code-block:: python

            #settings.py
            PRODUCER_CONFIG = {
                "PARAMS": {
                    "bootstrap.servers": "kafka1:9092, kafka2:9092",
                },
                "TOPIC": "test_topic"
            }

            STEP_CONFIG = { ...
                "PRODUCER_CONFIG": PRODUCER_CONFIG
            }

        If multiple producers are required, the varible inside `STEP_CONFIG`
        can be changed to "PRODUCER1_CONFIG", "PRODUCER2_CONFIG", etc.

    TOPIC_STRATEGY: dict

        Using a topic strategy instead of a fixed topic.
        Similar to the consumers topic strategy, the required parameters are:

        - *CLASS*: `apf.core.topic_management.GenericTopicStrategy` class to be used.
        - *PARAMS*: Parameters passed to *CLASS* object.

        **Example:**

        Produce to a topic that updates on 23 hours UTC every day.

        .. code-block:: python

            #settings.py
            PRODUCER_CONFIG = { ...
                "TOPIC_STRATEGY": {
                    "CLASS": "apf.core.topic_management.DailyTopicStrategy",
                    "PARAMS": {
                        "topic_format": "test_%s",
                        "date_format": "%Y%m%d",
                        "change_hour": 23
                    }
                }
            }

            STEP_CONFIG = { ...
                "PRODUCER_CONFIG": PRODUCER_CONFIG
            }

    SCHEMA: dict
        AVRO Output Schema
        `(AVRO Schema Definition) <https://avro.apache.org/docs/current/gettingstartedpython.html#Defining+a+schema>`_

        **Example:**

        .. code-block:: python

            #settings.py
            PRODUCER_CONFIG = { ...
                "SCHEMA": {
                    "namespace": "example.avro",
                    "type": "record",
                    "name": "User",
                    "fields": [
                        {"name": "name", "type": "string"},
                        {"name": "favorite_number",  "type": ["int", "null"]},
                        {"name": "favorite_color", "type": ["string", "null"]}
                    ]
                }
            }
    """

    def __init__(self, config):
        super().__init__(config=config)
        self.producer = Producer(self.config["PARAMS"])

        self.schema = fastavro.schema.load_schema(config["SCHEMA_PATH"])

        self.dynamic_topic = False
        if self.config.get("TOPIC"):
            self.logger.info(f'Producing to {self.config["TOPIC"]}')
            self.topic = (
                self.config["TOPIC"]
                if type(self.config["TOPIC"]) is list
                else [self.config["TOPIC"]]
            )
        elif self.config.get("TOPIC_STRATEGY"):
            self.dynamic_topic = True
            module_name, class_name = self.config["TOPIC_STRATEGY"]["CLASS"].rsplit(
                ".", 1
            )
            TopicStrategy = getattr(importlib.import_module(module_name), class_name)
            self.topic_strategy = TopicStrategy(
                **self.config["TOPIC_STRATEGY"]["PARAMS"]
            )
            self.topic = self.topic_strategy.get_topics()
            self.logger.info(f'Using {self.config["TOPIC_STRATEGY"]}')
            self.logger.info(f"Producing to {self.topic}")

    def _serialize_message(self, message):
        try:
            out = io.BytesIO()
            fastavro.writer(out, self.schema, [message])
            return out.getvalue()
        except Exception as e:
            self.logger.error(f"Error serializing message: {message}")
            raise e

    def _handle_buffer_error(self, err, topic, msg, key, callback, **kwargs):
        self.logger.error(f"Error producing message: {err}")
        self.logger.error("Calling flush and producing again")
        self.producer.flush(1)
        try:
            self.producer.produce(
                topic, value=msg, key=key, callback=callback, **kwargs
            )
        except BufferError as err:
            self._handle_buffer_error(err, topic, msg, key, callback, **kwargs)

    def produce(self, message=None, **kwargs):
        """Produce Message to a topic.

        Parameters
        ----------
        message: dict | None
            The value of the message to be produced.
            Should match the schema defined in the config["SCHEMA"]

        **kwargs: dict
            Any other keyword argument will be passed as **kwargs to the produce method of the producer.
            Do not specify the key argument, as it will be duplicated.

        NOTE: To set a key for the message, use the set_key_field(key_field) method prior to consuming
        The producer will have a key_field attribute that will be either None, or some specified value,
        and will use that for each produce() call.
        """

        def acked(err, msg):
            if err is not None:
                if isinstance(err, BufferError):
                    self._handle_buffer_error(err, topic, msg, key, acked, **kwargs)
                else:
                    raise err

        key = None
        if message:
            key = message[self.key_field] if self.key_field else kwargs.pop("key", None)
        message = self._serialize_message(message)
        if self.dynamic_topic:
            self.topic = self.topic_strategy.get_topics()
        for topic in self.topic:
            flush = kwargs.pop("flush", False)
            try:
                self.producer.produce(
                    topic, value=message, key=key, callback=acked, **kwargs
                )
            except BufferError as err:
                self._handle_buffer_error(err, topic, message, key, acked, **kwargs)
            if flush:
                self.producer.flush()

    def __del__(self):
        self.logger.info("Waiting to produce last messages")
        self.producer.flush()


class KafkaSchemalessProducer(KafkaProducer):
    def _serialize_message(self, message):
        try:
            strict = self.config.get("STRICT", True)
            out = io.BytesIO()
            fastavro.schemaless_writer(out, self.schema, message, strict=strict)
            return out.getvalue()
        except Exception as e:
            self.logger.error(f"Error serializing message: {message}")
            raise e
