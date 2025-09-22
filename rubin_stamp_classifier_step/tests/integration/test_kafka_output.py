import os
import unittest
import pytest
from rubin_stamp_classifier_step.step import StampClassifierStep
import sys
from confluent_kafka import Consumer
import fastavro
from io import BytesIO

sys.path.append(os.path.dirname(__file__))
from raw_kafka_producer import RawKafkaProducer
from apf.metrics.prometheus import DefaultPrometheusMetrics
import logging


# root directory for the project
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
kafka_bootstrap_servers = "localhost:9092"
input_topic = "lsst"
output_topic = "rubin_stamp_classifier"

# Minimal config for Kafka integration test
step_config = {
    "LOGGING_LEVEL": "DEBUG",
    "DB_CONFIG": {
        "USER": "postgres",
        "PASSWORD": "postgres",
        "HOST": "localhost",
        "PORT": 5432,
        "DB_NAME": "postgres",
        "SCHEMA": "public",
    },
    "CONSUMER_CONFIG": {
        "CLASS": "rubin_stamp_classifier_step.utils.LsstKafkaConsumer",
        "TOPICS": [input_topic],
        "PARAMS": {
            "bootstrap.servers": kafka_bootstrap_servers,
            "group.id": "test-rubin-stamp-classifier",
            "auto.offset.reset": "earliest",
            "enable.partition.eof": False,
        },
        "consume.timeout": 5,
        "consume.messages": 10,
        "SCHEMA_PATH": os.path.join(
            root_dir,
            "schemas",
            "surveys",
            "lsst",
            "v7_4_alert.avsc",
        ),
    },
    "PRODUCER_CONFIG": {
        "CLASS": "apf.producers.kafka.KafkaSchemalessProducer",
        "TOPIC": output_topic,
        "PARAMS": {"bootstrap.servers": kafka_bootstrap_servers},
        "SCHEMA_PATH": os.path.join(
            root_dir,
            "schemas",
            "rubin_stamp_classifier_step",
            "output.avsc",
        ),
    },
    "MODEL_VERSION": "",
    "MODEL_CONFIG": {
        "MODEL_PATH": os.environ["TEST_RUBIN_STAMP_CLASSIFIER_STEP_MODEL_PATH"]
    },
    "FEATURE_FLAGS": {
        "USE_PROFILING": False,
        "PROMETHEUS": False,
    },
}


@pytest.mark.usefixtures("kafka_service")
class TestKafkaOutput(unittest.TestCase):
    """Integration test for Kafka output of Rubin Stamp Classifier Step."""

    @classmethod
    def setUpClass(cls):
        """Set up the test environment once for all test methods."""
        cls.avro_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data", "avro_messages"
        )

        print("Filling Kafka topic with Avro messages from directory:", cls.avro_dir)
        # Produce Avro messages to input topic
        produce_avro_messages_to_kafka(
            kafka_bootstrap_servers,
            input_topic,
            cls.avro_dir,
        )
        print("Finished producing Avro messages to Kafka topic.")

    def test_kafka_output(self):
        step_config["CONSUMER_CONFIG"]["PARAMS"][
            "group.id"
        ] = "test-kafka-output-method"

        print("Starting Kafka output test...")
        step = StampClassifierStep(
            config=step_config,
            level=step_config["LOGGING_LEVEL"],
            prometheus_metrics=DefaultPrometheusMetrics(),
        )

        step._pre_consume()
        n_messages = 0
        execute_results = []
        for message in step.consumer.consume():
            print(len(message))
            if len(message) == 0:
                continue
            preprocessed_msg = step._pre_execute(message)
            if len(preprocessed_msg) == 0:
                continue
            try:
                result = step.execute(preprocessed_msg)
                execute_results.extend(result)
                n_messages += len(result)
            except Exception as error:
                logging.debug("Error at execute")
                logging.debug(f"The message(s) that caused the error: {message}")
                raise error

            # Post-execute is not called because DB interaction is not tested here

            result = step._pre_produce(result)
            step.produce(result)
            step._post_produce()
            if n_messages >= 5:
                break
        step._tear_down()
        print("Number of messages processed:", n_messages)

        # Now consume from the output topic and compare
        consumer = Consumer(
            {
                "bootstrap.servers": kafka_bootstrap_servers,
                "group.id": "test-kafka-output-checker",
                "auto.offset.reset": "earliest",
            }
        )
        consumer.subscribe([output_topic])
        output_schema_path = step_config["PRODUCER_CONFIG"]["SCHEMA_PATH"]

        with open(output_schema_path) as schema_file:
            output_schema = fastavro.schema.load_schema(output_schema_path)
        deserialized_outputs = []
        while len(deserialized_outputs) < n_messages:
            msg = consumer.poll(5.0)
            if msg is None:
                break
            if msg.error():
                print(f"Consumer error: {msg.error()}")
                continue
            bio = BytesIO(msg.value())
            record = fastavro.schemaless_reader(bio, output_schema)
            deserialized_outputs.append(record)
        consumer.close()
        print(
            f"Deserialized {len(deserialized_outputs)} output messages from topic '{output_topic}'."
        )
        self.assertEqual(
            len(deserialized_outputs),
            len(execute_results),
            "Output topic message count does not match execute results.",
        )
        # Order execute_results and deserialized_outputs by a key to ensure comparison is valid
        execute_results = sorted(execute_results, key=lambda x: x["diaSourceId"])
        deserialized_outputs = sorted(deserialized_outputs, key=lambda x: x["diaSourceId"])
        id_fields = [
            "diaObjectId",
            "ssObjectId",
            "diaSourceId",
        ]
        floating_point_fields = [
            "midpointMjdTai",
            "ra",
            "dec",
        ]
        for i in range(len(execute_results)):
            # Compare fields in execute_results and deserialized_outputs
            for field in id_fields + floating_point_fields:
                self.assertIn(
                    field,
                    execute_results[i],
                    f"Field '{field}' not found in execute_results at index {i}.",
                )
                self.assertIn(
                    field,
                    deserialized_outputs[i],
                    f"Field '{field}' not found in deserialized_outputs at index {i}.",
                )
                if field in floating_point_fields:
                    # Allow for small floating point differences
                    self.assertAlmostEqual(
                        execute_results[i][field],
                        deserialized_outputs[i][field],
                        places=5,
                        msg=f"Field '{field}' does not match at index {i}.",
                    )
                else:
                    # For ID fields, check exact match
                    self.assertEqual(
                        execute_results[i][field],
                        deserialized_outputs[i][field],
                        f"Field '{field}' does not match at index {i}.",
                    )

    def test_lsst_topic_has_messages(self):
        """Test that the lsst topic has messages after setup and count them"""
        consumer = Consumer(
            {
                "bootstrap.servers": kafka_bootstrap_servers,
                "group.id": "test-lsst-topic-has-messages-method",
                "auto.offset.reset": "earliest",
            }
        )
        consumer.subscribe([input_topic])

        n_messages = 0
        while True:
            msg = consumer.poll(5.0)
            if msg is None:
                break
            if msg.error():
                print(f"Consumer error: {msg.error()}")
                continue
            n_messages += 1
        consumer.close()

        print(f"Number of messages in topic '{input_topic}': {n_messages}")


def produce_avro_messages_to_kafka(
    kafka_bootstrap_servers: str, topic: str, avro_dir: str
):
    """
    Produce all Avro files in avro_dir to the given Kafka topic as raw bytes using RawKafkaProducer.
    Each file is sent as a single message, preserving the original bytes.
    """
    producer = RawKafkaProducer(
        {
            "PARAMS": {"bootstrap.servers": kafka_bootstrap_servers},
            "TOPIC": topic,
            "SCHEMA_PATH": os.path.join(
                root_dir,
                "schemas",
                "surveys",
                "lsst",
                "v7_4_alert.avsc",
            ),
        }
    )
    for filename in sorted(os.listdir(avro_dir)):
        if filename.endswith(".avro"):
            with open(os.path.join(avro_dir, filename), "rb") as f:
                data = f.read()
                producer.produce(data)
    producer.__del__()
