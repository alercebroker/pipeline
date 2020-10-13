from confluent_kafka import Producer
import datetime
import json
import importlib
import logging


class DateTimeEncoder(json.JSONEncoder):
    # Override the default method
    def default(self, obj):
        if isinstance(obj, (datetime.date, datetime.datetime)):
            return obj.isoformat()


class GenericMetricsProducer:
    def __init__(self, config):
        self.config = config

    def send_metrics(self, metrics):
        pass

class KafkaMetricsProducer(GenericMetricsProducer):
    def __init__(self, config, producer=None):
        super().__init__(config)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Creating {self.__class__.__name__}")
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
