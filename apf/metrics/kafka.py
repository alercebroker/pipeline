from apf.metrics import GenericMetricsProducer
from apf.metrics import DateTimeEncoder
from confluent_kafka import Producer
from apf.core import get_class

import json


class KafkaMetricsProducer(GenericMetricsProducer):
    """Write metrics in a Kafka Topic.

    Useful for high-throughput distributed metrics, a complex architecture can be build using
    Apache Kafka as a queue and writing the metrics inside a time series data store, for example
    Prometheus, InfluxDB or Elasticsearch.

    Parameters
    ----------
    config : dict.
        Parameters passed to the producer.

        - PARAMS: Parameters passed to :class:`apf.producer.KafkaProducer`.
        - TOPIC: List of topics to produce, for example ['metrics'].

    producer : apf.producers.GenericProducer
        An apf producer, by default is :class:`apf.producer.KafkaProducer`.

    """

    def __init__(self, config, producer=None):
        super().__init__(config)
        self.config = config
        if producer is not None:
            self.producer = producer
        else:
            self.producer = Producer(self.config["PARAMS"])
        self.time_encoder = self.config.get("TIME_ENCODER_CLASS", DateTimeEncoder)
        self.dynamic_topic = False
        if self.config.get("TOPIC"):
            self.logger.info(f'Producing metrics to {self.config["TOPIC"]}')
            self.topic = [self.config["TOPIC"]]
        elif self.config.get("TOPIC_STRATEGY"):
            self.dynamic_topic = True
            TopicStrategy = get_class(self.config["TOPIC_STRATEGY"]["CLASS"])
            self.topic_strategy = TopicStrategy(
                **self.config["TOPIC_STRATEGY"]["PARAMS"]
            )
            self.topic = self.topic_strategy.get_topics()
            self.logger.info(f'Using {self.config["TOPIC_STRATEGY"]}')
            self.logger.info(f"Producing to {self.topic}")

    def send_metrics(self, metrics):
        metrics = json.dumps(metrics, cls=self.time_encoder).encode("utf-8")

        if self.config.get("TOPIC_STRATEGY"):
            self.topic = self.topic_strategy.get_topics()
        for topic in self.topic:
            try:
                self.producer.produce(topic, metrics)
            except BufferError as e:
                self.logger.info(f"Error producing metrics: {e}")
                self.logger.info("Calling poll to empty queue and producing again")
                self.producer.poll(1)
                self.producer.produce(topic, metrics)

    def __del__(self):
        self.logger.info("Waiting to produce last messages")
        self.producer.flush()
