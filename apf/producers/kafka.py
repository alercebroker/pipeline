from apf.producers.generic import GenericProducer
from confluent_kafka import Producer

import io
import fastavro


class KafkaProducer(GenericProducer):
    """Kafka Single Topic Producer.

    Parameters
    ----------
    PARAMS: dict
        Parameters passed to :class:`confluent_kafka.Producer`

        The required parameters are:

        - *bootstrap.servers*: comma separated <host:port> :py:class:`str` to brokers.

    TOPIC: string
        Kafka output topic.

    SCHEMA: dict
        AVRO Output Schema `(AVRO Schema Definition) <https://avro.apache.org/docs/current/gettingstartedpython.html#Defining+a+schema>`_

    """
    def __init__(self,config):
        super().__init__(config=config)
        self.producer = Producer(self.config["PARAMS"])
        self.schema = self.config["SCHEMA"]

        self.schema = fastavro.parse_schema(self.schema)

        self.dynamic_topic = False
        if self.config.get("TOPICS"):
            self.logger.info(f'Producing to {self.config["TOPICS"]}')
            self.consumer.subscribe(self.config["TOPICS"])
        elif self.config.get("TOPIC_STRATEGY"):
            self.dynamic_topic = True
            module_name, class_name = self.config["TOPIC_STRATEGY"]["CLASS"].rsplit(".", 1)
            TopicStrategy = getattr(importlib.import_module(module_name), class_name)
            self.topic_strategy = TopicStrategy(**self.config["TOPIC_STRATEGY"]["PARAMS"])
            self.topic = self.topic_strategy.get_topic()
            self.logger.info(f'Using {self.config["TOPIC_STRATEGY"]}')
            self.logger.info(f'Producing to {self.topic}')
            self.consumer.subscribe(self.topic)


    def produce(self,message=None):
        """Produce Message to a topic.
        """
        out = io.BytesIO()
        fastavro.writer(out, self.schema, [message])
        avro_message = out.getvalue()

        if self.dynamic_topic:
            topics = self.topic_strategy.get_topic()
            if self.topic != topics:
                self.topic = topics

        for topic in self.topic:
            self.producer.produce(topic,avro_message)
            self.producer.flush()
