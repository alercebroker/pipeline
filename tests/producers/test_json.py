from .test_core import GenericProducerTest
from apf.producers import JSONProducer
import os

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
EXAMPLES_PATH = os.path.abspath(os.path.join(FILE_PATH, "../examples"))


class JSONProducerTest(GenericProducerTest):
    def setUp(self):
        self.file_path = os.path.join(EXAMPLES_PATH, "test_json_producer.json")
        self.params = {"FILE_PATH": self.file_path}
        self.component = JSONProducer(self.params)

    def test_produce(self):
        super().test_produce(self.component)

    def tearDown(self):
        path = self.file_path.split(".")[0] + "0.json"
        self.assertTrue(os.path.exists(path))
        os.remove(path)
