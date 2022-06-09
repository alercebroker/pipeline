from apf.core import get_class
from apf.core.step import GenericStep
from apf.producers import KafkaProducer

import logging
import time
import threading

class Simulator(GenericStep):
    """Simulator Description

    Parameters
    ----------
    consumer : GenericConsumer
        Description of parameter `consumer`.
    **step_args : type
        Other args passed to step (DB connections, API requests, etc.)

    """
    consumed = 0
    messages = []
    elapsed_time = 0
    start_time = 0
    n_messages = 10000
    process_time=60

    def __init__(self,consumer = None, config = None,level = logging.INFO,**step_args):
        super().__init__(consumer,config=config, level=level)
        if "CLASS" in config["PRODUCER_CONFIG"]:
            Producer = get_class(config["PRODUCER_CONFIG"]["CLASS"])
        else:
            Producer = KafkaProducer
        self.producer = Producer(config["PRODUCER_CONFIG"])
        self.n_messages = int(config.get("MESSAGES", 10000))
        self.exposure_time = 1 #float(config.get("EXPOSURE_TIME", 3))
        self.process_time = 1 #float(config.get("PROCESS_TIME", 1))
        self.key = config.get("KEY", "objectId")
        self.start_time = time.time()

    def produce(self):
        self.logger.info(f"Consumed {self.consumed}, producing")

        for message in self.messages:
            self.producer.produce(message, key=str(message[self.key]))

        self.logger.info(f"Message produced, waiting flush.")
        if type(self.producer) is KafkaProducer:
            self.producer.producer.flush()
        self.consumed = 0
        self.messages = []

        t1 = time.time()
        real_time = max([0,(self.exposure_time+self.process_time)-(t1-self.start_time)])
        self.logger.info(f"Sleeping for Exposure ({self.exposure_time}s), Process ({self.process_time}s) | Real Time {real_time:.3f}s")
        time.sleep(real_time)
        self.start_time = time.time()

    def check_timeout(self):
        now = time.time()
        elapsed = now - self.start_time
        if elapsed >= self.process_time:
            self.logger.info("Consume timeout, producing")
            self.produce()

    def execute(self, message):
        self.check_timeout()
        self.messages.append(message)
        self.consumed += 1
        if self.consumed == self.n_messages:
            self.produce()
