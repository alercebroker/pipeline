from .test_core import GenericProducerTest
from apf.producers import JSONProducer
import os
import pathlib

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
EXAMPLES_PATH = os.path.abspath(os.path.join(FILE_PATH, "../examples"))


class JSONProducerTest(GenericProducerTest):
    def setUp(self):
        self.params = {"FILE_PATH": EXAMPLES_PATH}
        self.component = JSONProducer(self.params)

    def test_produce(self):
        super().test_produce(self.component)
        self.path = pathlib.Path(EXAMPLES_PATH) / "producer_output0.json"
        self.assertTrue(self.path.exists())

    def tearDown(self):
        os.remove(self.path)
