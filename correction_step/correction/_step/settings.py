import os
import json

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
        "consume.timeout": int(os.getenv("CONSUME_TIMEOUT", 0)),
    }

    producer_config = {
        "CLASS": "apf.producers.KafkaProducer",
        "PARAMS": {
            "bootstrap.servers": os.environ["PRODUCER_SERVER"],
            "message.max.bytes": int(
                os.getenv("PRODUCER_MESSAGE_MAX_BYTES", 6291456)
            ),
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

    if os.getenv("CONSUMER_KAFKA_USERNAME") and os.getenv(
        "CONSUMER_KAFKA_PASSWORD"
    ):
        consumer_config["PARAMS"]["security.protocol"] = "SASL_SSL"
        consumer_config["PARAMS"]["sasl.mechanism"] = "SCRAM-SHA-512"
        consumer_config["PARAMS"]["sasl.username"] = os.getenv(
            "CONSUMER_KAFKA_USERNAME"
        )
        consumer_config["PARAMS"]["sasl.password"] = os.getenv(
            "CONSUMER_KAFKA_PASSWORD"
        )
    if os.getenv("PRODUCER_KAFKA_USERNAME") and os.getenv(
        "PRODUCER_KAFKA_PASSWORD"
    ):
        producer_config["PARAMS"]["security.protocol"] = "SASL_SSL"
        producer_config["PARAMS"]["sasl.mechanism"] = "SCRAM-SHA-512"
        producer_config["PARAMS"]["sasl.username"] = os.getenv(
            "PRODUCER_KAFKA_USERNAME"
        )
        producer_config["PARAMS"]["sasl.password"] = os.getenv(
            "PRODUCER_KAFKA_PASSWORD"
        )
    if os.getenv("SCRIBE_KAFKA_USERNAME") and os.getenv(
        "SCRIBE_KAFKA_PASSWORD"
    ):
        scribe_producer_config["PARAMS"]["security.protocol"] = "SASL_SSL"
        scribe_producer_config["PARAMS"]["sasl.mechanism"] = "SCRAM-SHA-512"
        scribe_producer_config["PARAMS"]["sasl.username"] = os.getenv(
            "SCRIBE_KAFKA_USERNAME"
        )
        scribe_producer_config["PARAMS"]["sasl.password"] = os.getenv(
            "SCRIBE_KAFKA_PASSWORD"
        )
    if os.getenv("METRICS_KAFKA_USERNAME") and os.getenv(
        "METRICS_KAFKA_PASSWORD"
    ):
        metrics_config["PARAMS"]["PARAMS"]["security.protocol"] = "SASL_SSL"
        metrics_config["PARAMS"]["PARAMS"]["sasl.mechanism"] = "SCRAM-SHA-512"
        metrics_config["PARAMS"]["PARAMS"]["sasl.username"] = os.getenv(
            "METRICS_KAFKA_USERNAME"
        )
        metrics_config["PARAMS"]["PARAMS"]["sasl.password"] = os.getenv(
            "METRICS_KAFKA_PASSWORD"
        )

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
