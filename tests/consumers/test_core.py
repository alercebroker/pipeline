from apf.consumers.generic import GenericConsumer

import unittest


class Consumer(GenericConsumer):
    def consume(self):
        yield {}


class GenericConsumerTest(unittest.TestCase):
    component: GenericConsumer
    params: dict

    def test_consume(self):
        self.component = Consumer()
        for msj in self.component.consume():
            self.assertIsInstance(msj, dict)
            self.component.commit()
