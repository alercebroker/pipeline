import io
import pathlib
import time
import unittest

import fastavro
import pytest
from confluent_kafka import Consumer, Producer

from reflector_step.step import CustomMirrormaker

PATH = pathlib.Path(
    pathlib.Path(__file__).parent.parent.parent.parent,
    "schemas/ztf",
    "alert.avsc",
)

# RawKafkaConsumerBytes reads the writer schema embedded in each message
# (Avro container format), so the payload does not need to match PATH.
ALERT_SCHEMA = fastavro.parse_schema(
    {
        "name": "alert",
        "type": "record",
        "fields": [
            {"name": "objectId", "type": "string"},
            {"name": "message", "type": "string"},
        ],
    }
)

PRODUCER_CONFIG = {
    "CLASS": "reflector_step.utils.RawKafkaProducer",
    "TOPIC": "test",
    "PARAMS": {"bootstrap.servers": "localhost:9093"},
    "SCHEMA_PATH": PATH,
}

CONSUMER_CONFIG = {
    "CLASS": "reflector_step.utils.RawKafkaConsumerBytes",
    "TOPICS": ["test_topic"],
    "PARAMS": {
        "bootstrap.servers": "localhost:9092",
        "group.id": "reflector_step_test",
        "auto.offset.reset": "beginning",
        "enable.partition.eof": True,
    },
    "consume.timeout": 1,
    "consume.messages": 5,
    "SCHEMA_PATH": PATH,
}


def create_avro_alert(object_id):
    out = io.BytesIO()
    fastavro.writer(
        out, ALERT_SCHEMA, [{"objectId": object_id, "message": "alert"}]
    )
    return out.getvalue()


def consume_all(topic, n_messages, timeout=30):
    consumer = Consumer(
        {
            "bootstrap.servers": "localhost:9093",
            "group.id": "test_consumer",
            "auto.offset.reset": "earliest",
        }
    )
    consumer.subscribe([topic])
    messages = []
    deadline = time.time() + timeout
    while len(messages) < n_messages and time.time() < deadline:
        msg = consumer.poll(1)
        if msg is not None and not msg.error():
            messages.append(msg)
    consumer.close()
    return messages


@pytest.mark.usefixtures("kafka_service")
class MyTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.step_config = {
            "PRODUCER_CONFIG": PRODUCER_CONFIG,
            "CONSUMER_CONFIG": CONSUMER_CONFIG,
        }

    def test_keep_timestamp(self):
        n_messages = 10
        external = Producer({"bootstrap.servers": "localhost:9092"})
        step = CustomMirrormaker(
            config=self.step_config,
            keep_original_timestamp=True,
            use_message_topic=False,
            survey="ztf",
        )

        object_ids = [f"ZTF26aaaaa{i:03d}" for i in range(n_messages)]
        payloads = [create_avro_alert(oid) for oid in object_ids]
        for payload in payloads:
            external.produce(topic="test_topic", value=payload, timestamp=123)
        external.flush()
        step.start()

        messages = consume_all("test", n_messages)
        self.assertEqual(len(messages), n_messages)
        for msg, payload, object_id in zip(messages, payloads, object_ids):
            self.assertEqual(msg.timestamp()[1], 123)
            self.assertEqual(msg.value(), payload)
            self.assertEqual(msg.key(), object_id.encode())
