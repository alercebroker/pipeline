from apf.consumers import GenericConsumer

import unittest

class GenericConsumerTest(unittest.TestCase):
    component = GenericConsumer
    params = {}

    def test_consume(self):
        comp = self.component(self.params)
        for msj in comp.consume():
            self.assertIsInstance(msj, dict)
            comp.commit()
