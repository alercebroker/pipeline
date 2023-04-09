import json
import os

from fastavro import schema

SCHEMA_DIR = os.path.join(os.path.dirname(__file__), "schemas")


def load_json(path: str) -> dict:
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
        "SCHEMA": schema.load_schema(os.path.join(SCHEMA_DIR, "output.avsc")),
    }

    scribe_producer_config = {
        "CLASS": "apf.producers.KafkaProducer",
        "PARAMS": {
            "bootstrap.servers": os.environ["SCRIBE_SERVER"],
        },
        "TOPIC": os.environ["SCRIBE_TOPIC"],
        "SCHEMA": schema.load_schema(os.path.join(SCHEMA_DIR, "scribe.avsc")),
    }

    metrics_config = {
        "CLASS": "apf.metrics.KafkaMetricsProducer",
        "PARAMS": {
            "PARAMS": {
                "bootstrap.servers": os.getenv("METRICS_SERVER"),
                "auto.offset.reset": "smallest",
            },
            "TOPIC": os.getenv("METRICS_TOPIC", "metrics"),
            "SCHEMA": load_json(os.path.join(SCHEMA_DIR, "metrics.json")),
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
