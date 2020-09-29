##################################################
#       early_classifier   Settings File
##################################################
import os

DB_CONFIG = {
    "SQL": {
        "ENGINE": "postgres",
        "HOST": os.environ["DB_HOST"],
        "USER": os.environ["DB_USER"],
        "PASSWORD": os.environ["DB_PASSWORD"],
        "PORT": int(os.environ["DB_PORT"]),
        "DB_NAME": os.environ["DB_NAME"],
    }
}

METRICS_CONFIG = {
    "CLASS": "apf.metrics.KafkaMetricsProducer",
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

CONSUMER_CONFIG = {
    "TOPICS": os.environ["CONSUMER_TOPICS"].strip().split(","),
    "PARAMS": {
        "bootstrap.servers": os.environ["CONSUMER_SERVER"],
        "group.id": os.environ["CONSUMER_GROUP_ID"],
        "auto.offset.reset": "smallest",
        "enable.partition.eof": os.getenv("ENABLE_PARTITION_EOF", False),
    }
}

STEP_METADATA = {
    "STEP_VERSION": os.getenv("STEP_VERSION", "dev"),
    "STEP_ID": os.getenv("STEP_ID", "stamp_classification"),
    "STEP_NAME": os.getenv("STEP_NAME", "stamp_classification"),
    "STEP_COMMENTS": os.getenv("STEP_COMMENTS", ""),
    "CLASSIFIER_NAME": os.getenv("CLASSIFIER_NAME", "stamp_classifier"),
    "CLASSIFIER_VERSION": os.getenv("CLASSIFIER_VERSION", "0.0.0")
}

STEP_CONFIG = {
    "DB_CONFIG": DB_CONFIG,
    "METRICS_CONFIG": METRICS_CONFIG,
    "STEP_METADATA": STEP_METADATA,
    "N_RETRY": os.getenv("N_RETRY", 5),
}

LOGGING_DEBUG = os.getenv("LOGGING_DEBUG", False)
