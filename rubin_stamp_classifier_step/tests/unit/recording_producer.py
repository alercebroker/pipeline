"""A producer the step can load by import path that records what it is given.

It is a real apf GenericProducer so the framework discovers it when it
drains every producer before committing the consumer offset.
"""
from apf.producers.generic import GenericProducer


class RecordingProducer(GenericProducer):
    def __init__(self, config=None):
        super().__init__(config)
        self.produced = []
        self.flushes = 0

    def produce(self, message=None, **kwargs):
        self.produced.append((message, kwargs))

    def flush(self):
        self.flushes += 1
