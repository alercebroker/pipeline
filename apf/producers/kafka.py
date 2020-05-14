from apf.producers.generic import GenericProducer
from confluent_kafka import Producer

import fastavro
import io
import importlib

import json


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

        If multiple producers are required, the varible inside `STEP_CONFIG` can be changed to "PRODUCER1_CONFIG", "PRODUCER2_CONFIG", etc.

    TOPIC_STRATEGY: dict

        Using a topic strategy instead of a fixed topic. Similar to the consumers topic strategy, the required parameters are:

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
        AVRO Output Schema `(AVRO Schema Definition) <https://avro.apache.org/docs/current/gettingstartedpython.html#Defining+a+schema>`_

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
        self.schema = self.config["SCHEMA"]

        self.schema = fastavro.parse_schema(self.schema)

        self.dynamic_topic = False
        if self.config.get("TOPIC"):
            self.logger.info(f'Producing to {self.config["TOPIC"]}')
            self.topic = [self.config["TOPIC"]]
        elif self.config.get("TOPIC_STRATEGY"):
            self.dynamic_topic = True
            module_name, class_name = self.config["TOPIC_STRATEGY"]["CLASS"].rsplit(
                ".", 1)
            TopicStrategy = getattr(
                importlib.import_module(module_name), class_name)
            self.topic_strategy = TopicStrategy(
                **self.config["TOPIC_STRATEGY"]["PARAMS"])
            self.topic = self.topic_strategy.get_topic()
            self.logger.info(f'Using {self.config["TOPIC_STRATEGY"]}')
            self.logger.info(f'Producing to {self.topic}')
            self.consumer.subscribe(self.topic)

    def produce(self, message=None):
        """Produce Message to a topic.
        """
        out = io.BytesIO()
        fastavro.writer(out, self.schema, [message])
        message = out.getvalue()
        # message = json.dumps(message)
        for topic in self.topic:
            try:
                self.producer.produce(topic, message)
            except BufferError as e:
                self.logger.debug(f"Error producing message: {e}")
                self.logger.debug("Calling poll to empty queue and producing again")
                self.producer.poll(1)
                self.producer.produce(topic, message)


    def __del__(self):
        self.logger.info("Waiting to produce last messages")
        self.producer.flush()
