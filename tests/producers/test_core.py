from typing import Type
from apf.producers import GenericProducer
import unittest


class Producer(GenericProducer):
    def produce(self, message, key):
        pass


class GenericProducerTest(unittest.TestCase):
    component: GenericProducer

    def test_produce(self, use: GenericProducer = Producer()):
        self.component = use
        self.component.produce({"key": "test", "int": 1}, key="key")
