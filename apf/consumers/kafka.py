from apf.consumers.generic import GenericConsumer
from confluent_kafka import Consumer

import fastavro
import io
import importlib


class KafkaConsumer(GenericConsumer):
    """Consume from a Kafka Topic.

    By default :class:`KafkaConsumer` uses a manual commit strategy to avoid data loss on errors. This strategy can be disabled
    completly adding `"COMMIT":False` to the `STEP_CONFIG` variable in the step's `settings.py` file.

    **Example:**

    .. code-block:: python

        #settings.py
        STEP_CONFIG = { ...
            "COMMIT": False #Disable commit
            #useful for testing/debugging.
        }

    Parameters
    ----------
    TOPICS: list
            List of topics to consume.

            **Example:**

            Subscribe to a fixed list of topics:

            .. code-block:: python

                #settings.py
                CONSUMER_CONFIG = { ...
                    "TOPICS": ["topic1", "topic2"]
                }

            Using `confluent_kafka` syntax we can subscribe to a pattern

            .. code-block:: python

                #settings.py
                CONSUMER_CONFIG = { ...
                    "TOPICS": ["^topic*"]
                }

            More on pattern subscribe `here <https://docs.confluent.io/current/clients/confluent-kafka-python/#confluent_kafka.Consumer.subscribe>`_

    TOPIC_STRATEGY: dict
            Parameters to configure a topic strategy instead of a fixed topic list.

            The required parameters are:

            - *CLASS*: `apf.core.topic_management.GenericTopicStrategy` class to be used.
            - *PARAMS*: Parameters passed to *CLASS* object.

            **Example:**

            A topic strategy that updates on 23 hours UTC every day.

            .. code-block:: python

                #settings.py
                CONSUMER_CONFIG = { ...
                    "TOPIC_STRATEGY": {
                        "CLASS": "apf.core.topic_management.DailyTopicStrategy",
                        "PARAMS": {
                            "topic_format": ["ztf_%s_programid1", "ztf_%s_programid3"],
                            "date_format": "%Y%m%d",
                            "change_hour": 23
                        }
                    }
                }

    PARAMS: dict
        Parameters passed to :class:`confluent_kafka.Consumer`

        The required parameters are:

        - *bootstrap.servers*: comma separated <host:port> :py:class:`str` to brokers.
        - *group.id*: :py:class:`str` with consumer group name.

        **Example:**

        Configure a Kafka Consumer to a secure Kafka Cluster

        .. code-block:: python

            #settings.py
            CONSUMER_CONFIG = { ...
                "PARAMS": {
                    "bootstrap.servers": "kafka1:9093,kafka2:9093",
                    "group.id": "step_group",
                    'security.protocol': 'SSL',
                    'ssl.ca.location': '<ca-cert path>',
                    'ssl.keystore.location': '<keystore path>',
                    'ssl.keystore.password': '<keystore password>'
                }
            }

    """

    def __init__(self, config):
        super().__init__(config)
        # Disable auto commit
        self.config["PARAMS"]["enable.auto.commit"] = False
        # if "max.poll.interval.ms" not in config["PARAMS"]:
        #     self.config["PARAMS"]["max.poll.interval.ms"] = 10*60*1000

        if "auto.offset.reset" not in self.config:
            self.config["PARAMS"]["auto.offset.reset"] = "beginning"
        # Creating consumer
        self.consumer = Consumer(self.config["PARAMS"])

        self.dynamic_topic = False
        if self.config.get("TOPICS"):
            self.logger.info(f'Subscribing to {self.config["TOPICS"]}')
            self.consumer.subscribe(self.config["TOPICS"])
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
            self.logger.info(f'Subscribing to {self.topic}')
            self.consumer.subscribe(self.topic)
        else:
            raise Exception("No topics o topic strategy set. ")

    def __del__(self):
        self.logger.info("Shutting down Consumer")
        self.consumer.close()

    def _deserialize_message(self, message):
        bytes_io = io.BytesIO(message.value())
        reader = fastavro.reader(bytes_io)
        data = reader.next()
        return data

    def consume(self, num_messages=1, timeout=60):
        """
        Consumes `num_messages` messages from the specified topic.
        
        Will return a dictionary or a list, depending on the number of messages consumed.

        If num_messages > 1 then it returns list.

        If num_messages = 1 then it returns dict.

        Parameters
        --------------
        num_messages: int
            Number of messages to be consumed
        timeout: int
            Seconds to wait when consuming messages. Raises exception if doesn't get the messages after
            specified time
        """
        if "consume.messages" in self.config:
            num_messages = self.config["consume.messages"]
        if "consume.timeout" in self.config:
            timeout = self.config["consume.timeout"]
        messages = []
        while True:
            messages = self.consumer.consume(num_messages=num_messages,timeout=timeout)
            if len(messages) == 0:
                continue
            
            deserialized = []
            for message in messages:
                if message.error():
                    raise Exception(f"Error in kafka stream: {message.error()}")
                
                message = self._deserialize_message(message)
                deserialized.append(message)

            self.messages = messages
            messages = []
            if num_messages == 1:
                yield deserialized[0]
            else:
                yield deserialized
    def commit(self):
        for message in self.messages:
            self.consumer.commit(message)
