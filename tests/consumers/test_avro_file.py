from .test_core import GenericConsumerTest
from apf.consumers import AVROFileConsumer
import unittest

import os
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
EXAMPLES_PATH = os.path.abspath(os.path.join(FILE_PATH,"../examples"))


class AVROFileConsumerTest(GenericConsumerTest,unittest.TestCase):
    component = AVROFileConsumer
    params = {
        "DIRECTORY_PATH": os.path.join(EXAMPLES_PATH,"avro_test")
    }
    __test__ = True
