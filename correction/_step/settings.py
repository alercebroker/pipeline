import json
import os

from fastavro import schema

SCHEMA_DIR = os.path.join(os.path.dirname(__file__), "schemas")


def get_output_schema() -> dict:
    return schema.load_schema(os.path.join(SCHEMA_DIR, "output.avsc"))


def get_scribe_schema() -> dict:
    return schema.load_schema(os.path.join(SCHEMA_DIR, "scribe.avsc"))


def get_metrics_schema() -> dict:
    path = os.path.join(SCHEMA_DIR, "metrics.json")
    with open(path, "r") as fh:
        return json.load(fh)


def settings_creator():
    # Set the global logging level to debug
    logging_debug = bool(os.getenv("LOGGING_DEBUG"))
    prometheus = bool(os.getenv("USE_PROMETHEUS"))

    # Consumer configuration
    # Each consumer has different parameters and can be found in the documentation
    consumer_config = {
        "CLASS": "apf.consumers.KafkaConsumer",
        "PARAMS": {
            "bootstrap.servers": os.environ["CONSUMER_SERVER"],
            "group.id": os.environ["CONSUMER_GROUP_ID"],
            "auto.offset.reset": "beginning",
            "enable.partition.eof": bool(os.getenv("ENABLE_PARTITION_EOF")),
        },
        "TOPICS": os.environ["CONSUMER_TOPICS"].split(","),
        "consume.messages": int(os.getenv("CONSUME_MESSAGES", 50)),
        "consume.timeout": int(os.getenv("CONSUME_TIMEOUT", 10)),
    }

    producer_config = {
        "CLASS": "apf.producers.KafkaProducer",
        "PARAMS": {
            "bootstrap.servers": os.environ["PRODUCER_SERVER"],
        },
        "TOPIC": os.environ["PRODUCER_TOPIC"],
        "SCHEMA": get_output_schema(),
    }

    scribe_producer_config = {
        "CLASS": "apf.producers.KafkaProducer",
        "PARAMS": {
            "bootstrap.servers": os.environ["SCRIBE_SERVER"],
        },
        "TOPIC": os.environ["SCRIBE_TOPIC"],
        "SCHEMA": get_scribe_schema(),
    }

    metrics_config = {
        "CLASS": "apf.metrics.KafkaMetricsProducer",
        "PARAMS": {
            "PARAMS": {
                "bootstrap.servers": os.getenv("METRICS_SERVER"),
                "auto.offset.reset": "smallest",
            },
            "TOPIC": os.getenv("METRICS_TOPIC", "metrics"),
            "SCHEMA": get_metrics_schema(),
        },
    }

    # Step Configuration
    step_config = {
        "CONSUMER_CONFIG": consumer_config,
        "METRICS_CONFIG": metrics_config,
        "PRODUCER_CONFIG": producer_config,
        "SCRIBE_PRODUCER_CONFIG": scribe_producer_config,
        "LOGGING_DEBUG": logging_debug,
        "PROMETHEUS": prometheus,
    }
    return step_config
