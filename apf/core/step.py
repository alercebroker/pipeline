from abc import abstractmethod

from apf.consumers import GenericConsumer
from apf.producers import GenericProducer

class GenericStep():
    def __init__(self,consumer,producer = None, **step_args):
        self.consumer = consumer
        self.producer = producer

    @abstractmethod
    def execute(self):
        pass
