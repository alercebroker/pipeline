from .core import GenericConsumerTest
from apf.consumers import KafkaConsumer
from confluent_kafka.admin import AdminClient, NewTopic
import unittest
from confluent_kafka import Producer, TopicPartition
import fastavro
import random
import io
import datetime


class TestKafkaConsumer(GenericConsumerTest, unittest.TestCase):
    component = KafkaConsumer
    params = {
        "TOPICS": ["apf_test"],
        "PARAMS": {"bootstrap.servers": "127.0.0.1:9092", "group.id": "apf_test"},
    }

    admin = AdminClient({"bootstrap.servers": "127.0.0.1:9092"})

    def setUp(self):
        self.tp = TopicPartition("apf_test", partition=0)
        p = Producer({"bootstrap.servers": "127.0.0.1:9092"})

        schema = {
            "doc": "A weather reading.",
            "name": "Weather",
            "namespace": "test",
            "type": "record",
            "fields": [
                {"name": "id", "type": "int"},
                {"name": "station", "type": "string"},
                {"name": "time", "type": "long"},
                {"name": "temp", "type": "int"},
            ],
        }
        schema = fastavro.parse_schema(schema)
        for i in range(100):
            out = io.BytesIO()
            alert = {
                "id": i,
                "station": "test",
                "time": random.randint(0, 1000000),
                "temp": random.randint(0, 100),
            }
            fastavro.writer(out, schema, [alert])
            message = out.getvalue()
            p.produce("apf_test", message)
        p.flush()

    def tearDown(self):
        fs = self.admin.delete_topics(["apf_test"])
        for t, f in fs.items():
            try:
                f.result()
            except Exception as e:
                print(f"failed to delete topic {t}: {e}")

    def test_consume(self):
        pass

    def test_1_consume(self):
        n_msjs = 0
        self.comp = self.component(self.params)
        for msj in self.comp.consume():
            self.assertIsInstance(msj, dict)
            n_msjs += 1
            if msj["id"] == 99:
                break
        self.assertEqual(n_msjs, 100)

    def test_consume_batch(self):
        params = self.params
        params["PARAMS"]["group.id"] = "new"
        self.comp = self.component(params)
        n_batches = 0
        for batch in self.comp.consume(10):
            self.assertIsInstance(batch, list)
            self.assertEqual(len(batch), 10)
            self.assertIsInstance(batch[0], dict)
            n_batches += 1
            if batch[-1]["id"] == 99:
                break
        self.assertEqual(n_batches, 10)

    def test_2_commit(self):
        # Loading data without commit
        n_msjs = 0

        self.comp = self.component(self.params)
        first_offset = self.comp.consumer.position([self.tp])[0].offset
        for msj in self.comp.consume():
            self.assertIsInstance(msj, dict)
            n_msjs += 1
            if msj["id"] == 99:
                break
        self.assertEqual(n_msjs, 100)
        offset_without = self.comp.consumer.position([self.tp])[0].offset
        self.assertEqual(offset_without, 100)

        self.comp = self.component(self.params)
        offset_second = self.comp.consumer.position([self.tp])[0].offset
        n_msjs = 0
        for msj in self.comp.consume():
            self.assertIsInstance(msj, dict)
            self.comp.commit()
            n_msjs += 1
            if msj["id"] == 99:
                break
        self.assertEqual(n_msjs, 100)
        offset_with = self.comp.consumer.position([self.tp])[0].offset
        self.assertEqual(offset_with, 100)


class TestKafkaConsumerDynamicTopic(unittest.TestCase):
    component = KafkaConsumer
    now = datetime.datetime.utcnow()
    tomorrow = now + datetime.timedelta(days=1)
    date_format = "%Y%m%d"
    topic1 = "apf_test_" + now.strftime(date_format)
    topic2 = "apf_test_" + tomorrow.strftime(date_format)
    params = {
        "TOPIC_STRATEGY": {
            "CLASS": "apf.core.topic_management.DailyTopicStrategy",
            "PARAMS": {
                "topic_format": "apf_test_%s",
                "date_format": date_format,
                "change_hour": now.hour,
            },
        },
        "PARAMS": {"bootstrap.servers": "127.0.0.1:9092", "group.id": "apf_test"},
    }
    admin = AdminClient({"bootstrap.servers": "127.0.0.1:9092"})

    def setUp(self):
        self.tp = TopicPartition(self.topic1)
        self.tp2 = TopicPartition(self.topic2)
        p = Producer({"bootstrap.servers": "127.0.0.1:9092"})

        schema = {
            "doc": "A weather reading.",
            "name": "Weather",
            "namespace": "test",
            "type": "record",
            "fields": [
                {"name": "id", "type": "int"},
                {"name": "station", "type": "string"},
                {"name": "time", "type": "long"},
                {"name": "temp", "type": "int"},
            ],
        }
        schema = fastavro.parse_schema(schema)
        for i in range(2):
            out = io.BytesIO()
            alert = {
                "id": i,
                "station": "test",
                "time": random.randint(0, 1000000),
                "temp": random.randint(0, 100),
            }
            fastavro.writer(out, schema, [alert])
            message = out.getvalue()
            p.produce(self.topic1, message)
            p.produce(self.topic2, message)
        p.flush()

    def tearDown(self):
        topics = list(self.admin.list_topics().topics.keys())
        topics.remove("__consumer_offsets")
        fs = self.admin.delete_topics(topics)
        for t, f in fs.items():
            try:
                f.result()
            except Exception as e:
                print(f"failed to delete topic {t}: {e}")

    def test_recognizes_dynamic_topic(self):
        self.comp = self.component(self.params)
        self.assertTrue(self.comp.dynamic_topic)

    def test_creates_correct_topic_strategy_class(self):
        from apf.core.topic_management import DailyTopicStrategy
        self.comp = self.component(self.params)
        self.assertTrue(isinstance(self.comp.topic_strategy, DailyTopicStrategy))

    def test_subscribes_to_correct_topic_list(self):
        self.comp = self.component(self.params)
        self.assertEqual(self.comp.topics, [self.topic1, self.topic2])

    def test_detects_new_topic_while_consuming(self):
        import copy
        params = copy.deepcopy(self.params)
        np1 = self.now.hour + 1 if self.now.hour <=24 else 0
        params["TOPIC_STRATEGY"]["PARAMS"]["change_hour"] = np1 
        self.comp = self.component(params)
        self.comp.topic_strategy.change_hour = self.now.hour
        self.assertEqual(self.comp.topics, [self.topic1])
        for msg in self.comp.consume():
            self.comp.commit()
            if msg["id"] == 1:
                break
        self.assertEqual(self.comp.topics, [self.topic1, self.topic2])

    def test_consumes_from_all_topics(self):
        self.comp = self.component(self.params)
        msgs = 0
        for msg in self.comp.consume():
            self.comp.commit()
            msgs += 1
            if msg["id"] == 1:
                break
        self.assertEqual(msgs, 2)
