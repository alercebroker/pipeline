##################################################
#       features   Settings File
##################################################
import os
from schema import SCHEMA
from fastavro import schema

EXTRACTOR = os.environ["FEATURE_EXTRACTOR"]


CONSUMER_CONFIG = {
    "CLASS": "apf.consumers.KafkaConsumer",
    "TOPICS": os.environ["CONSUMER_TOPICS"].strip().split(","),
    "PARAMS": {
        "bootstrap.servers": os.environ["CONSUMER_SERVER"],
        "group.id": os.environ["CONSUMER_GROUP_ID"],
        "auto.offset.reset": "beginning",
        "max.poll.interval.ms": 3600000,
        "enable.partition.eof": os.getenv("ENABLE_PARTITION_EOF", False),
    },
    "consume.timeout": int(os.getenv("CONSUME_TIMEOUT", 10)),
    "consume.messages": int(os.getenv("CONSUME_MESSAGES", 1000)),
}

PRODUCER_CONFIG = {
    "CLASS": "apf.producers.KafkaProducer",
    "TOPIC": os.environ["PRODUCER_TOPIC"],
    "PARAMS": {
        "bootstrap.servers": os.environ["PRODUCER_SERVER"],
        "message.max.bytes": int(
            os.getenv("PRODUCER_MESSAGE_MAX_BYTES", 6291456)
        ),
    },
    "SCHEMA": SCHEMA,
}

SCRIBE_PRODUCER_CONFIG = {
    "CLASS": "apf.producers.KafkaProducer",
    "PARAMS": {
        "bootstrap.servers": os.environ["SCRIBE_SERVER"],
    },
    "TOPIC": os.environ["SCRIBE_TOPIC"],
    "SCHEMA": schema.load_schema("scribe_schema.avsc"),
}


METRICS_CONFIG = {
    "CLASS": "apf.metrics.KafkaMetricsProducer",
    "EXTRA_METRICS": [{"key": "aid", "alias": "aid"}, {"key": "candid"}],
    "PARAMS": {
        "PARAMS": {
            "bootstrap.servers": os.environ["METRICS_HOST"],
        },
        "TOPIC": os.environ["METRICS_TOPIC"],
        "SCHEMA": {
            "$schema": "http://json-schema.org/draft-07/schema",
            "$id": "http://example.com/example.json",
            "type": "object",
            "title": "The root schema",
            "description": "The root schema comprises the entire JSON document.",
            "default": {},
            "examples": [
                {
                    "timestamp_sent": "2020-09-01",
                    "timestamp_received": "2020-09-01",
                }
            ],
            "required": ["timestamp_sent", "timestamp_received"],
            "properties": {
                "timestamp_sent": {
                    "$id": "#/properties/timestamp_sent",
                    "type": "string",
                    "title": "The timestamp_sent schema",
                    "description": "Timestamp sent refers to the time at which a message is sent.",
                    "default": "",
                    "examples": ["2020-09-01"],
                },
                "timestamp_received": {
                    "$id": "#/properties/timestamp_received",
                    "type": "string",
                    "title": "The timestamp_received schema",
                    "description": "Timestamp received refers to the time at which a message is received.",
                    "default": "",
                    "examples": ["2020-09-01"],
                },
            },
            "additionalProperties": True,
        },
    },
}

if os.getenv("KAFKA_USERNAME") and os.getenv("KAFKA_PASSWORD"):
    CONSUMER_CONFIG["PARAMS"]["security.protocol"] = "SASL_SSL"
    CONSUMER_CONFIG["PARAMS"]["sasl.mechanism"] = "SCRAM-SHA-512"
    CONSUMER_CONFIG["PARAMS"]["sasl.username"] = os.getenv("KAFKA_USERNAME")
    CONSUMER_CONFIG["PARAMS"]["sasl.password"] = os.getenv("KAFKA_PASSWORD")
    PRODUCER_CONFIG["PARAMS"]["security.protocol"] = "SASL_SSL"
    PRODUCER_CONFIG["PARAMS"]["sasl.mechanism"] = "SCRAM-SHA-512"
    PRODUCER_CONFIG["PARAMS"]["sasl.username"] = os.getenv("KAFKA_USERNAME")
    PRODUCER_CONFIG["PARAMS"]["sasl.password"] = os.getenv("KAFKA_PASSWORD")

    SCRIBE_PRODUCER_CONFIG["PARAMS"]["security.protocol"] = "SASL_SSL"
    SCRIBE_PRODUCER_CONFIG["PARAMS"]["sasl.mechanism"] = "SCRAM-SHA-512"
    SCRIBE_PRODUCER_CONFIG["PARAMS"]["sasl.username"] = os.getenv(
        "KAFKA_USERNAME"
    )
    SCRIBE_PRODUCER_CONFIG["PARAMS"]["sasl.password"] = os.getenv(
        "KAFKA_PASSWORD"
    )

    METRICS_CONFIG["PARAMS"]["PARAMS"]["security.protocol"] = "SASL_SSL"
    METRICS_CONFIG["PARAMS"]["PARAMS"]["sasl.mechanism"] = "SCRAM-SHA-512"
    METRICS_CONFIG["PARAMS"]["PARAMS"]["sasl.username"] = os.getenv(
        "KAFKA_USERNAME"
    )
    METRICS_CONFIG["PARAMS"]["PARAMS"]["sasl.password"] = os.getenv(
        "KAFKA_PASSWORD"
    )


use_profiling = bool(os.getenv("USE_PROFILING", True))
pyroscope_server = os.getenv(
    "PYROSCOPE_SERVER", "http://pyroscope.pyroscope:4040"
)

STEP_CONFIG = {
    "CONSUMER_CONFIG": CONSUMER_CONFIG,
    "PRODUCER_CONFIG": PRODUCER_CONFIG,
    "SCRIBE_PRODUCER_CONFIG": SCRIBE_PRODUCER_CONFIG,
    "METRICS_CONFIG": METRICS_CONFIG,
    "USE_PROFILING": use_profiling,
    "PYROSCOPE_SERVER": pyroscope_server,
}
