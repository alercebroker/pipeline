from .test_core import GenericConsumerTest
from apf.consumers import AVROFileConsumer
import unittest

import os

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
EXAMPLES_PATH = os.path.abspath(os.path.join(FILE_PATH, "../examples"))


class AVROFileConsumerTest(GenericConsumerTest, unittest.TestCase):
    params = {
        "DIRECTORY_PATH": os.path.join(EXAMPLES_PATH, "avro_test"),
        "consume.messages": 1,
    }
    component = AVROFileConsumer(params)
    __test__ = True

    def test_consume_left_messages(self):
        params = self.params
        params["consume.messages"] = 5
        self.component = AVROFileConsumer(params)
        total = 0
        loops = 0
        for msgs in self.component.consume():
            loops += 1
            total += len(msgs)
        assert total == 6
        assert loops == 2
