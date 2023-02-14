##################################################
#       atlas_stamp_classifier_step   Settings File
##################################################

import os
from schema import SCHEMA


CONSUMER_CONFIG = {
    "CLASS": os.getenv("CONSUMER_CLASS", "apf.consumers.KafkaConsumer"),
    "TOPICS": os.environ["CONSUMER_TOPICS"].strip().split(","),
    "PARAMS": {
        "bootstrap.servers": os.environ["CONSUMER_SERVER"],
        "group.id": os.environ["CONSUMER_GROUP_ID"],
        "auto.offset.reset": "smallest",
    },
    "consume.timeout": int(os.getenv("CONSUME_TIMEOUT", 10)),
    "consume.messages": int(os.getenv("CONSUME_MESSAGES", 50)),
}

OUTPUT_PRODUCER_CONFIG = {
    "TOPIC_STRATEGY": {
        "PARAMS": {
            "topic_format": os.environ["PRODUCER_TOPIC_FORMAT"],
            "date_format": os.environ["PRODUCER_DATE_FORMAT"],
            "change_hour": int(os.environ["PRODUCER_CHANGE_HOUR"]),
            "retention_days": int(os.environ["PRODUCER_RETENTION_DAYS"]),
        },
        "CLASS": os.getenv(
            "PRODUCER_TOPIC_STRATEGY_CLASS",
            "apf.core.topic_management.DailyTopicStrategy",
        ),
    },
    "PARAMS": {
        "bootstrap.servers": os.environ["PRODUCER_SERVER"],
        "security.protocol": os.getenv("PRODUCER_SECURITY_PROTOCOL", "SASL_PLAINTEXT"),
        "sasl.mechanism": os.getenv("PRODUCER_SASL_MECHANISM", "SCRAM-SHA-256"),
        "sasl.username": os.environ["PRODUCER_SASL_USERNAME"],
        "sasl.password": os.environ["PRODUCER_SASL_PASSWORD"],
    },
    "SCHEMA": SCHEMA,
}

SCRIBE_PRODUCER_CONFIG = {
    "TOPIC": os.environ["SCRIBE_TOPIC"],
    "PARAMS": {
        "bootstrap.servers": os.environ["SCRIBE_SERVER"],
    },
}

METRICS_CONFIG = {
    "CLASS": "apf.metrics.KafkaMetricsProducer",
    "EXTRA_METRICS": [{"key": "candid", "format": lambda x: str(x)}, "oid"],
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


STEP_CONFIG = {
    "PRODUCER_CONFIG": OUTPUT_PRODUCER_CONFIG,
    "SCRIBE_CONFIG": SCRIBE_PRODUCER_CONFIG,
    "MODEL_NAME": os.getenv("MODEL_NAME", "atlas_stamp_classifier"),
    "MODEL_VERSION": os.getenv("MODEL_VERSION", "1.0.0"),
    "METRICS_CONFIG": METRICS_CONFIG,
    "MODEL_PATH": os.environ["MODEL_PATH"],
}
