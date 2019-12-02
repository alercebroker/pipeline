from abc import abstractmethod

from apf.consumers import GenericConsumer
from apf.producers import GenericProducer


import logging

class GenericStep():
    def __init__(self,consumer = None, producer = None, level = logging.INFO,**step_args):
        self.consumer = GenericConsumer() if consumer is None else consumer
        self.producer = GenericProducer() if producer is None else producer
        self.message = None

    @abstractmethod
    def execute(self):
        pass

    def start(self):
        for self.message in self.consumer.consume():
            self.execute(self.message)
            if self.producer:
                self.producer.produce()
            self.consumer.commit()
