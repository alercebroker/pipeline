from apf.consumers.generic import GenericConsumer

import unittest


class Consumer(GenericConsumer):
    def consume(self):
        yield {}


class GenericConsumerTest(unittest.TestCase):
    component: GenericConsumer

    def test_consume(self, use: GenericConsumer = Consumer()):
        self.component = use
        for msj in self.component.consume():
            self.assertIsInstance(msj, dict)
