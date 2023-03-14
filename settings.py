##################################################
#       features   Settings File
##################################################
import os
from schema import SCHEMA

FEATURE_VERSION = os.environ["FEATURE_VERSION"]
STEP_VERSION = os.environ["STEP_VERSION"]

DB_CONFIG = {
    "SQL": {
        "ENGINE": os.environ["DB_ENGINE"],
        "HOST": os.environ["DB_HOST"],
        "USER": os.environ["DB_USER"],
        "PASSWORD": os.environ["DB_PASSWORD"],
        "PORT": int(os.environ["DB_PORT"]),
        "DB_NAME": os.environ["DB_NAME"],
    }
}

CONSUMER_CONFIG = {
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
    "TOPIC": os.environ["PRODUCER_TOPIC"],
    "PARAMS": {
        "bootstrap.servers": os.environ["PRODUCER_SERVER"],
    },
    "SCHEMA": SCHEMA,
}

METRICS_CONFIG = {
    "CLASS": "apf.metrics.KafkaMetricsProducer",
    "EXTRA_METRICS": [
        {"key": "candid", "format": lambda x: str(x)},
        {"key": "oid", "alias": "oid"},
        {"key": "aid", "alias": "aid"},
        {"key": "tid", "format": lambda x: str(x)},
    ],
    "PARAMS": {
        "PARAMS": {
            "bootstrap.servers": os.environ["METRICS_HOST"],
            "auto.offset.reset": "smallest",
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
                {"timestamp_sent": "2020-09-01", "timestamp_received": "2020-09-01"}
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
    METRICS_CONFIG["PARAMS"]["PARAMS"]["security.protocol"] = "SASL_SSL"
    METRICS_CONFIG["PARAMS"]["PARAMS"]["sasl.mechanism"] = "SCRAM-SHA-512"
    METRICS_CONFIG["PARAMS"]["PARAMS"]["sasl.username"] = os.getenv("KAFKA_USERNAME")
    METRICS_CONFIG["PARAMS"]["PARAMS"]["sasl.password"] = os.getenv("KAFKA_PASSWORD")

STEP_METADATA = {
    "STEP_VERSION": os.getenv("STEP_VERSION", "dev"),
    "STEP_ID": os.getenv("STEP_ID", "features"),
    "STEP_NAME": os.getenv("STEP_NAME", "features"),
    "STEP_COMMENTS": os.getenv("STEP_COMMENTS", ""),
    "FEATURE_VERSION": os.getenv("FEATURE_VERSION", "dev"),
}

STEP_CONFIG = {
    "CONSUMER_CONFIG": CONSUMER_CONFIG,
    "DB_CONFIG": DB_CONFIG,
    "PRODUCER_CONFIG": PRODUCER_CONFIG,
    "METRICS_CONFIG": METRICS_CONFIG,
    "STEP_METADATA": STEP_METADATA,
}
