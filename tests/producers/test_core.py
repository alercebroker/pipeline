from apf.producers import GenericProducer
import unittest

class GenericProducerTest(unittest.TestCase):
    component = GenericProducer
    params = {}

    def test_produce(self):
        comp = self.component(self.params)
        msj = comp.produce({'key':'value', 'int':1}, key="test")
