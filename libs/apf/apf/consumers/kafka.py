import datetime
from apf.consumers.generic import GenericConsumer
from confluent_kafka import OFFSET_END, Consumer, KafkaException

import fastavro
import io
import importlib
import json


class KafkaConsumer(GenericConsumer):
    """Consume from a Kafka Topic.

    By default :class:`KafkaConsumer` uses a manual commit strategy to avoid data loss on errors.

    This strategy can be disabled completly adding `"COMMIT":False` to the `STEP_CONFIG` variable
    in the step's `settings.py` file, this can be useful for step testing because Kafka doesn't save
    the messages that already were processed.

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

        More on pattern subscribe
        `here <http://docs.confluent.io/current/clients/confluent-kafka-python/#confluent_kafka.Consumer.subscribe>`_


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
              "CLASS": "apf.core.topic_management"+\\
                        "DailyTopicStrategy",
              "PARAMS": {
                "topic_format": [
                    "ztf_%s_programid1",
                    "ztf_%s_programid3"
                    ],
                "date_format": "%Y%m%d",
                "change_hour": 23,
                "retention_days": 8,
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

        all supported `confluent_kafka` parameters can be found
        `on the official library documentation <https://github.com/edenhill/librdkafka/blob/master/CONFIGURATION.md>`_
    """

    consumer: Consumer

    def __init__(self, config: dict):
        super().__init__(config)
        # Disable auto commit
        self.config["PARAMS"]["enable.auto.commit"] = False
        # Creating consumer
        self.consumer = Consumer(self.config["PARAMS"])
        self.add_topic = config.get("ADD_TOPIC", False)
        self.max_retries = int(self.config.get("COMMIT_RETRY", 5))

        self.logger.info(
            f"Creating consumer for {self.config['PARAMS'].get('bootstrap.servers')}"
        )
        self.dynamic_topic = False
        if self.config.get("TOPICS"):
            self.logger.info(f'Subscribing to {self.config["TOPICS"]}')
            self.consumer.subscribe(self.config["TOPICS"], on_assign=self._on_assign)
        elif self.config.get("TOPIC_STRATEGY"):
            self.dynamic_topic = True
            module_name, class_name = self.config["TOPIC_STRATEGY"]["CLASS"].rsplit(
                ".", 1
            )
            TopicStrategy = getattr(importlib.import_module(module_name), class_name)
            self.topic_strategy = TopicStrategy(
                **self.config["TOPIC_STRATEGY"]["PARAMS"]
            )
            self.topics = self.topic_strategy.get_topics()
            self.logger.info(f'Using {self.config["TOPIC_STRATEGY"]}')
            self.logger.info(f"Subscribing to {self.topics}")
            self.consumer.subscribe(self.topics, on_assign=self._on_assign)
        else:
            raise Exception("No topics o topic strategy set. ")

    def teardown(self):
        self.logger.info("Shutting down Consumer")
        if hasattr(self, "consumer"):
            self.consumer.unsubscribe()
            self.consumer.close()

    def __del__(self):
        self.teardown()

    def _deserialize_message(self, message):
        bytes_io = io.BytesIO(message.value())
        reader = fastavro.reader(bytes_io)
        for data in reader:
            return data

    def _check_topics(self):
        """
        Returns true if new topic
        """
        topics = self.topic_strategy.get_topics()
        if topics != self.topics:
            return True
        return False

    def _subscribe_to_new_topics(self):
        """
        Sets current topic to new topic
        """
        self.topics = self.topic_strategy.get_topics()
        self.consumer.unsubscribe()
        self.logger.info(f"Suscribing to {self.topics}")
        self.consumer.subscribe(self.topics, on_assign=self._on_assign)

    def set_basic_config(self, num_messages, timeout):
        if "consume.messages" in self.config:
            num_messages = self.config["consume.messages"]
        elif "NUM_MESSAGES" in self.config:
            num_messages = self.config["NUM_MESSAGES"]

        if "consume.timeout" in self.config:
            timeout = self.config["consume.timeout"]
        elif "TIMEOUT" in self.config:
            timeout = self.config["TIMEOUT"]
        return num_messages, timeout

    def _on_assign(self, consumer, partitions):
        if self.config.get("offsets", {}).get("start", False):
            self.logger.info(f"Running with offsets {self.config['offsets']}")
            for partition in partitions:
                partition.offset = self.config["offsets"]["start"]
            partitions = consumer.offsets_for_times(partitions)
            consumer.assign(partitions)
            self.logger.debug(f"Assigned partitions {partitions}")

    def _post_process(self, parsed, original_message):
        parsed["timestamp"] = original_message.timestamp()[1]
        if self.add_topic:
            parsed["topic"] = original_message.topic()
        return parsed

    def consume(self, num_messages=1, timeout=60):
        """
        Consumes `num_messages` messages from the specified topic.

        Will return a dictionary or a list, depending on the number of messages consumed.

        If num_messages > 1 then it returns list.

        If num_messages = 1 then it returns dict.

        Parameters
        ----------
        num_messages: int
            Number of messages to be consumed
        timeout: int
            Seconds to wait when consuming messages. Raises exception if doesn't get the messages after
            specified time
        """
        num_messages, timeout = self.set_basic_config(num_messages, timeout)

        messages = []
        while True:
            if self.dynamic_topic:
                if self._check_topics():
                    self._subscribe_to_new_topics()

            messages = self.consumer.consume(num_messages=num_messages, timeout=timeout)
            if len(messages) == 0:
                continue

            deserialized = []
            eof = False
            for message in messages:
                if message.error():
                    if message.error().name() == "_PARTITION_EOF":
                        self.logger.info("PARTITION_EOF: No more messages")
                        eof = True
                        continue
                    self.logger.exception(f"Error in kafka stream: {message.error()}")
                    continue
                else:
                    if self.config.get("offsets"):
                        end = self.config["offsets"].get(
                            "end", (datetime.datetime.now().timestamp() + 1) * 1000
                        )
                        if message.timestamp()[1] >= end:
                            self.logger.info(
                                "Reached end offset %s with timestamp %s",
                                end,
                                message.timestamp()[1],
                            )
                            break
                    ds_message = self._deserialize_message(message)
                    ds_message = self._post_process(ds_message, message)
                    deserialized.append(ds_message)

            self.messages = messages
            messages = []
            if len(deserialized) > 0:
                if num_messages == 1:
                    yield deserialized[0]
                else:
                    yield deserialized
            if eof:
                break
        if self.config.get("exit_on_consume", False):
            self.logger.info("Exiting on consume")
            self.teardown()

    def commit(self):
        retries = 0
        commited = False

        while not commited:
            try:
                self.consumer.commit(asynchronous=False)
                commited = True
            except KafkaException as e:
                retries += 1

                # Rasing the same error
                if retries == self.max_retries:
                    raise e


class KafkaJsonConsumer(KafkaConsumer):
    def __init__(self, config):
        super().__init__(config)

    def _deserialize_message(self, message):
        msg_value = message.value()
        data = json.loads(msg_value)
        return data


class KafkaSchemalessConsumer(KafkaConsumer):
    def __init__(self, config: dict):
        super().__init__(config)
        schema_path = config.get("SCHEMA_PATH")
        if schema_path:
            self.schema = fastavro.schema.load_schema(schema_path)
        else:
            raise Exception("No Schema path provided")

        super().__init__(config)

    def _deserialize_message(self, message):
        bytes_io = io.BytesIO(message.value())
        data = fastavro.schemaless_reader(bytes_io, self.schema)

        return data
