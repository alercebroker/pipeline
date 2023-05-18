import unittest
from .test_core import GenericProducerTest
from apf.producers import JSONProducer
import os


class JSONProducerTest(GenericProducerTest, unittest.TestCase):
    component = JSONProducer
    params = {"FILE_PATH": "test.json"}

    def setUp(self):
        if os.path.exists(self.params["FILE_PATH"]):
            os.remove(self.params["FILE_PATH"])
