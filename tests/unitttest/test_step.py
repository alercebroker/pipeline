from operator import le
import unittest
from unittest import mock
import pytest
from cmirrormaker.utils.consumer import RawKafkaConsumer
from cmirrormaker.utils.producer import RawKafkaProducer
from cmirrormaker.step import CustomMirrormaker
from data.datagen import create_data


class StepTestCase(unittest.TestCase):
    def setUp(self):
        self.mock_consumer = mock.create_autospec(RawKafkaConsumer)
        self.mock_producer = mock.create_autospec(RawKafkaProducer)
        self.step = CustomMirrormaker(
            consumer=self.mock_consumer,
            config={"PRODUCER_CONFIG": "cmirrormaker.utiles.producer"},
            level=69,
            producer=self.mock_producer,
        )

    def tearDown(self):
        del self.step

    def test_step(self):
        data_batch = create_data(10)
        self.step.execute(data_batch)
        self.mock_producer.produce.assert_called()
