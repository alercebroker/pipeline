from .test_core import GenericConsumerTest
from apf.consumers import JSONConsumer
import unittest

import os

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
EXAMPLES_PATH = os.path.abspath(os.path.join(FILE_PATH, "../examples"))


class JSONConsumerTest(GenericConsumerTest, unittest.TestCase):
    component = JSONConsumer
    params = {"FILE_PATH": os.path.join(EXAMPLES_PATH, "test.json")}
    __test__ = True

    def test_no_path(self):
        self.assertRaises(Exception, self.component, {})
