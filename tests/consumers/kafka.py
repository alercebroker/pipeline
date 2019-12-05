from .core import GenericConsumerTest
from apf.consumers import KafkaConsumer

import unittest
from confluent_kafka import Producer,TopicPartition
import fastavro
import random
import io


class KafkaConsumer(GenericConsumerTest,unittest.TestCase):
    component = KafkaConsumer
    params = {
        "TOPICS": ["apf_test"],
        "PARAMS": {
            'bootstrap.servers': '127.0.0.1:9092',
            'group.id': 'apf_test'

        }
    }

    tp = TopicPartition('apf_test',partition=0)



    @classmethod
    def setUpClass(cls):
        p = Producer({"bootstrap.servers": "127.0.0.1:9092"})

        schema = {
            'doc': 'A weather reading.',
            'name': 'Weather',
            'namespace': 'test',
            'type': 'record',
            'fields': [
                {'name': 'id', 'type': 'int'},
                {'name': 'station', 'type': 'string'},
                {'name': 'time', 'type': 'long'},
                {'name': 'temp', 'type': 'int'},
            ],
        }
        schema = fastavro.parse_schema(schema)

        for i in range(100):
            out = io.BytesIO()
            alert = {'id': i, 'station':'test', 'time': random.randint(0,1000000), 'temp':random.randint(0,100)}
            fastavro.writer(out, schema, [alert])
            message = out.getvalue()
            p.produce('apf_test', message)
            p.flush()

    def test_consume(self):
        pass

    def test_1_consume(self):
        n_msjs = 0
        self.comp = self.component(self.params)
        for msj in self.comp.consume():
            self.assertIsInstance(msj, dict)
            n_msjs +=1
            if(msj["id"] == 99):
                break
        self.assertEqual(n_msjs, 100)

    def test_2_commit(self):
        #Loading data without commit
        n_msjs = 0

        self.comp = self.component(self.params)
        first_offset = self.comp.consumer.position([self.tp])[0].offset
        for msj in self.comp.consume():
            self.assertIsInstance(msj, dict)
            n_msjs += 1
            if(msj["id"] == 99):
                break
        self.assertEqual(n_msjs, 100)
        offset_without = self.comp.consumer.position([self.tp])[0].offset
        self.assertTrue(offset_without == 100)

        self.comp = self.component(self.params)
        offset_second = self.comp.consumer.position([self.tp])[0].offset
        n_msjs = 0
        for msj in self.comp.consume():
            self.assertIsInstance(msj, dict)
            self.comp.commit()
            n_msjs +=1
            if(msj["id"] == 99):
                break
        self.assertEqual(n_msjs, 100)
        offset_with = self.comp.consumer.position([self.tp])[0].offset
        self.assertEqual(offset_with,100)
