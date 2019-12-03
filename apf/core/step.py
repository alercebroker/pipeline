from abc import abstractmethod

from apf.consumers import GenericConsumer

import logging

class GenericStep():
    def __init__(self,consumer = None, level = logging.INFO,**step_args):
        self.consumer = GenericConsumer() if consumer is None else consumer
        self.message = None

    @abstractmethod
    def execute(self):
        pass

    def start(self):
        for self.message in self.consumer.consume():
            self.execute(self.message)
            self.consumer.commit()
