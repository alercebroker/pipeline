from .test_core import GenericProducerTest
from apf.producers import KafkaProducer
from unittest import mock
import unittest
import datetime

class KafkaProducerMock(unittest.TestCase):
    def __init__(self):
        pass

    def flush(self):
        pass

    def produce(self, topic, msj):
        self.assertIsInstance(topic,str)
        self.assertIsInstance(msj, (str, bytes))

@mock.patch('apf.producers.kafka.Producer', return_value=KafkaProducerMock())
class KafkaProducerTest(GenericProducerTest):
    component = KafkaProducer

    params = {
         "PARAMS": {
            "bootstrap.servers": "kafka1:9092, kafka2:9092"
            },
         "TOPIC": "test_topic",
         "SCHEMA": {
            "namespace": "test.avro",
            "type": "record",
            "name": "test",
            "fields": [
                {"name": "key", "type": "string"},
                {"name": "int",  "type": ["int"]},
            ]
        }
    }

    def test_produce(self, producer_mock):
        super().test_produce()

    def test_topic_strategy(self,producer_mock):
        #TODO: Check if topic is changed at certain hour
        now = datetime.datetime.utcnow()
        tomorrow = now + datetime.timedelta(days=1)
        date_format = "%Y%m%d"
        params = {
            "PARAMS": {
                "bootstrap.servers": "kafka1:9092, kafka2:9092"
            },
            "TOPIC_STRATEGY": {
                 "CLASS": "apf.core.topic_management.DailyTopicStrategy",
                 "PARAMS": {
                     "topic_format": "apf_test_%s",
                     "date_format": date_format,
                     "change_hour": now.hour,
                },
            },
            "SCHEMA": {
                "namespace": "test.avro",
                "type": "record",
                "name": "test",
                "fields": [
                    {"name": "key", "type": "string"},
                    {"name": "int",  "type": ["int"]},
                ]
            }
        }
        comp = self.component(params)
        msj = comp.produce({'key':'value', 'int':1})
