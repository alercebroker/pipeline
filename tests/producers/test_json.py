from .test_core import GenericProducerTest
from apf.producers import JSONProducer
import unittest
import tempfile
import os

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
EXAMPLES_PATH = os.path.abspath(os.path.join(FILE_PATH,"../examples"))

class JSONProducerTest(GenericProducerTest):    
    component = JSONProducer
    file_path = os.path.join(EXAMPLES_PATH,"test_json_producer.json")
    params = {
        "FILE_PATH": file_path
    }

    def tearDown(self):
        self.assertTrue(os.path.exists(self.file_path))
        os.remove(self.file_path)
