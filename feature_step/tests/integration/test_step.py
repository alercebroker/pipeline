from features.step import FeaturesComputer
from schema import SCHEMA
from features.utils.selector import selector
from fastavro.schema import load_schema
import pathlib
import os

# SCHEMA PATH RELATIVE TO THE SETTINGS FILE
PRODUCER_SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "../../../schemas/feature_step/output.avsc")
METRICS_SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "../../../schemas/feature_step/metrics.json")
SCRIBE_SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "../../../schemas/scribe.avsc")

CONSUMER_CONFIG = {
    "CLASS": "apf.consumers.kafka.KafkaConsumer",
    "PARAMS": {
        "bootstrap.servers": "localhost:9092",
        "group.id": "group_id",
        "auto.offset.reset": "beginning",
        "enable.partition.eof": True,
    },
    "TOPICS": ["elasticc"],
    "consume.messages": 1,
    "consume.timeout": 0,
}

PRODUCER_CONFIG = {
    "CLASS": "apf.producers.kafka.KafkaProducer",
    "TOPIC": "test_output",
    "PARAMS": {
        "bootstrap.servers": "localhost:9092",
    },
    "SCHEMA_PATH": PRODUCER_SCHEMA_PATH,
}

try:
    schema_path = pathlib.Path("scribe_schema.avsc")
    assert schema_path.exists()
except Exception:
    schema_path = pathlib.Path("feature_step/scribe_schema.avsc")
    assert schema_path.exists()
SCRIBE_PRODUCER_CONFIG = {
    "CLASS": "apf.producers.kafka.KafkaProducer",
    "TOPIC": "test-scribe",
    "PARAMS": {
        "bootstrap.servers": "localhost:9092",
    },
    "SCHEMA_PATH": SCRIBE_SCHEMA_PATH,
}


def test_step_ztf(kafka_service):
    CONSUMER_CONFIG["TOPICS"] = ["ztf"]
    step_config = {
        "PRODUCER_CONFIG": PRODUCER_CONFIG,
        "CONSUMER_CONFIG": CONSUMER_CONFIG,
        "SCRIBE_PRODUCER_CONFIG": SCRIBE_PRODUCER_CONFIG,
    }
    extractor = selector("ztf")
    step = FeaturesComputer(
        extractor,
        config=step_config,
    )
    step.start()


def test_step_elasticc(kafka_service):
    CONSUMER_CONFIG["TOPICS"] = ["elasticc"]
    step_config = {
        "PRODUCER_CONFIG": PRODUCER_CONFIG,
        "CONSUMER_CONFIG": CONSUMER_CONFIG,
        "SCRIBE_PRODUCER_CONFIG": SCRIBE_PRODUCER_CONFIG,
    }
    extractor = selector("elasticc")
    step = FeaturesComputer(
        extractor,
        config=step_config,
    )
    step.start()

