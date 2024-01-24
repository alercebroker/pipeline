from .test_core import GenericProducerTest
from apf.producers import CSVProducer

import os

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
EXAMPLES_PATH = os.path.abspath(os.path.join(FILE_PATH, "../examples"))


class CSVProducerTest(GenericProducerTest):
    def setUp(self) -> None:
        self.file_path = os.path.join(EXAMPLES_PATH, "test_csv_producer.csv")
        self.params = {"FILE_PATH": self.file_path}
        self.component = CSVProducer(self.params)

    def test_produce(self):
        super().test_produce(self.component)

    def tearDown(self):
        self.assertTrue(os.path.exists(self.file_path))
        os.remove(self.file_path)
