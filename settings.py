##################################################
#       s3_step   Settings File
##################################################
import os

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
        "auto.offset.reset":"smallest",
        "enable.partition.eof": os.getenv("ENABLE_PARTITION_EOF", False),
    },
}


METRICS_CONFIG = {
    "CLASS": "apf.metrics.KafkaMetricsProducer",
    "PARAMS": {
        "PARAMS": {
            "bootstrap.servers": os.environ["METRICS_HOST"],
            "auto.offset.reset":"smallest"},
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

STEP_METADATA = {
    "STEP_VERSION": os.getenv("STEP_VERSION", "dev"),
    "STEP_ID": os.getenv("STEP_ID", "features"),
    "STEP_NAME": os.getenv("STEP_NAME", "features"),
    "STEP_COMMENTS": os.getenv("STEP_COMMENTS", ""),
    "FEATURE_VERSION": os.getenv("FEATURE_VERSION", "dev")
}

STORAGE_CONFIG = {"BUCKET_NAME": os.environ["BUCKET_NAME"], "REGION_NAME": os.environ["REGION_NAME"]}

LOGGING_DEBUG = os.getenv("LOGGING_DEBUG", False)

STEP_CONFIG = {
    "STORAGE": STORAGE_CONFIG,
    "CONSUMER_CONFIG": CONSUMER_CONFIG,
    "DB_CONFIG": DB_CONFIG,
    "METRICS_CONFIG": METRICS_CONFIG,
    "STEP_METADATA": STEP_METADATA,
}
