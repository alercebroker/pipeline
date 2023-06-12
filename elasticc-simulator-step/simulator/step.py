from apf.core import get_class
from apf.core.step import GenericStep
from apf.producers import KafkaProducer

import logging
import time


class Simulator(GenericStep):
    """Simulator Description

    Parameters
    ----------
    consumer : GenericConsumer
        Description of parameter `consumer`.
    **step_args : type
        Other args passed to step (DB connections, API requests, etc.)

    """

    def __init__(self, consumer=None, config: dict = {}, level=logging.INFO):
        super().__init__(consumer, config=config, level=level)
        if "CLASS" in config["PRODUCER_CONFIG"]:
            Producer = get_class(config["PRODUCER_CONFIG"]["CLASS"])
        else:
            Producer = KafkaProducer
        self.producer = Producer(config["PRODUCER_CONFIG"])
        self.n_messages = config["MESSAGES"]
        self.exposure_time = config["EXPOSURE_TIME"]
        self.process_time = config["PROCESS_TIME"]
        self.start_time = time.time()
        self.messages = []

    def produce(self):
        for message in self.messages:
            self.producer.produce(message)

        self.logger.info(f"Message produced, waiting flush.")
        if type(self.producer) is KafkaProducer:
            self.producer.producer.flush(30)
        self.messages = []

    def sleep_for_exposure(self):
        t1 = time.time()
        real_time = max(
            [0, (self.exposure_time + self.process_time) - (t1 - self.start_time)]
        )
        self.logger.info(
            f"Sleeping for Exposure ({self.exposure_time}s), Process ({self.process_time}s) | Real Time {real_time:.3f}s"
        )
        time.sleep(real_time)

    def check_consumer_timeout(self):
        now = time.time()
        elapsed = now - self.start_time
        return elapsed >= self.config["CONSUME_TIMEOUT"]

    def execute(self, message: dict | list):
        self.start_time = time.time()
        self.sleep_for_exposure()
        if isinstance(message, list):
            self.messages = message
        else:
            self.messages.append(message)
        consumed = len(self.messages)
        self.logger.info(f"{consumed} messages consumed. Producing")
        print(f"{consumed} messages consumed. Producing")
        self.produce()
