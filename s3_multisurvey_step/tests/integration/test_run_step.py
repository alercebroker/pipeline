import io
import unittest
import fastavro
import boto3
import pytest
from apf.consumers import KafkaConsumer
from confluent_kafka import Producer
from moto import mock_s3
from s3_step.step import S3Step


CONSUMER_CONFIG = {
    "PARAMS": {
        "bootstrap.servers": "localhost:9092",
        "group.id": "test_consumer",
        "auto.offset.reset": "beginning",
        "enable.partition.eof": True,
    },
    "consume.messages": 1,
    "TOPICS": ["test_topic1", "test_topic2", "test_survey"],
}

STEP_CONFIG = {
    "CONSUMER_CONFIG": CONSUMER_CONFIG,
    "STORAGE": {
        "BUCKET_NAME": {"test_topic": "test_bucket1", "test_survey": "test_bucket2"},
        "REGION_NAME": "us-east1",
    },
}

schema = {
    "type": "record",
    "name": "test",
    "fields": [
        {"name": "objectId", "type": "string"},
        {
            "name": "candidate",
            "type": {
                "type": "record",
                "name": "candidateRecord",
                "fields": [{"name": "candid", "type": "int"}],
            },
        },
    ],
}


def serialize_message(message):
    out = io.BytesIO()
    fastavro.writer(out, schema, [message])
    return out.getvalue()


@mock_s3
@pytest.mark.usefixtures("kafka_service")
class TestIntegration(unittest.TestCase):
    def setUp(self):
        self.conn = boto3.resource("s3")
        self.conn.create_bucket(Bucket="test_bucket1")
        self.conn.create_bucket(Bucket="test_bucket2")
        self.messages = [
            {
                "objectId": "oid1",
                "candidate": {"candid": 123},
            },
            {
                "objectId": "oid2",
                "candidate": {"candid": 124},
            },
            {
                "objectId": "oid3",
                "candidate": {"candid": 125},
            },
        ]
        self.serialized = [serialize_message(message) for message in self.messages]
        self.topics = ["test_topic2", "test_survey", "test_topic1"]

    def tearDown(self):
        self.conn.Bucket("test_bucket1").objects.all().delete()
        self.conn.Bucket("test_bucket2").objects.all().delete()
        del self.conn

    def test_step_execution(self):
        external = Producer({"bootstrap.servers": "localhost:9092"})
        consumer = KafkaConsumer(CONSUMER_CONFIG)
        step = S3Step(
            consumer=consumer, config=STEP_CONFIG, s3_client=boto3.client("s3")
        )

        for msg, topic in zip(self.serialized, self.topics):
            external.produce(topic=topic, value=msg)
        external.flush()
        step.start()
        step.start()
        step.start()

        o123, o125 = list(self.conn.Bucket("test_bucket1").objects.all())
        self.assertEqual(o123.key, "321.avro")
        self.assertEqual(o125.key, "521.avro")

        (o124,) = list(self.conn.Bucket("test_bucket2").objects.all())
        self.assertEqual(o124.key, "421.avro")
