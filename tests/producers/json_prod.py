import unittest
from .core import GenericProducerTest
from apf.producers import JSONProducer

class JSONProducerTest(GenericProducerTest, unittest.TestCase):
    component = JSONProducer
    params = {}
