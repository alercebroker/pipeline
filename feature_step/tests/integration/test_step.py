from features.step import FeaturesComputer
from features.utils.selector import selector
import pathlib


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
    "SCHEMA_PATH": pathlib.Path(
        pathlib.Path(__file__).parent.parent.parent.parent,
        "schemas/feature_step",
        "output.avsc",
    ),
}

SCRIBE_PRODUCER_CONFIG = {
    "CLASS": "apf.producers.kafka.KafkaProducer",
    "TOPIC": "test-scribe",
    "PARAMS": {
        "bootstrap.servers": "localhost:9092",
    },
    "SCHEMA_PATH": pathlib.Path(
        pathlib.Path(__file__).parent.parent.parent.parent,
        "schemas/scribe_step",
        "scribe.avsc",
    ),
}


def test_step_ztf(ztf_messages):
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


def test_step_elasticc(elasticc_messages):
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

def test_step_atlas(atlas_messages):
    CONSUMER_CONFIG["TOPICS"] = ["atlas"]
    step_config = {
        "PRODUCER_CONFIG": PRODUCER_CONFIG,
        "CONSUMER_CONFIG": CONSUMER_CONFIG,
        "SCRIBE_PRODUCER_CONFIG": SCRIBE_PRODUCER_CONFIG,
    }
    extractor = selector("atlas")
    step = FeaturesComputer(
        extractor,
        config=step_config,
    )
    step.start()

def test_step_ztf_atlas_messages(atlas_messages_ztf_topic):
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
