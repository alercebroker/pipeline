##################################################
#       early_classifier   Settings File
##################################################
import os
import json

# SCHEMA PATH RELATIVE TO THE SETTINGS FILE
PRODUCER_SCHEMA_PATH = os.path.join(os.path.dirname(__file__), os.getenv("PRODUCER_SCHEMA_PATH"))
with open(PRODUCER_SCHEMA_PATH, "r") as f:
    PRODUCER_SCHEMA = json.load(f)

METRICS_SCHEMA_PATH = os.path.join(os.path.dirname(__file__), os.getenv("METRICS_SCHEMA_PATH"))
with open(METRICS_SCHEMA_PATH, "r") as f:
    METRICS_SCHEMA = json.load(f)

DB_CONFIG = {
    "SQL": {
        "ENGINE": "postgresql",
        "HOST": os.environ["DB_HOST"],
        "USER": os.environ["DB_USER"],
        "PASSWORD": os.environ["DB_PASSWORD"],
        "PORT": int(os.environ["DB_PORT"]),
        "DB_NAME": os.environ["DB_NAME"],
    }
}

METRICS_CONFIG = {
    "CLASS": "apf.metrics.KafkaMetricsProducer",
    "EXTRA_METRICS": [
        {"key": "candid", "format": lambda x: str(x)},
        {"key": "objectId", "alias": "oid"},
    ],
    "PARAMS": {
        "PARAMS": {
            "bootstrap.servers": os.environ["METRICS_HOST"],
            "auto.offset.reset": "smallest",
        },
        "TOPIC": os.environ["METRICS_TOPIC"],
        "SCHEMA_PATH": METRICS_SCHEMA_PATH,
        "SCHEMA": METRICS_SCHEMA
    },
}

CONSUMER_CONFIG = {
    "PARAMS": {
        "bootstrap.servers": os.environ["CONSUMER_SERVER"],
        "group.id": os.environ["CONSUMER_GROUP_ID"],
        "auto.offset.reset": "smallest",
        "max.poll.interval.ms": 3600000,
        "enable.partition.eof": os.getenv("ENABLE_PARTITION_EOF", False),
    },
}

if os.getenv("TOPIC_STRATEGY_FORMAT"):
    CONSUMER_CONFIG["TOPIC_STRATEGY"] = {
        "CLASS": "apf.core.topic_management.DailyTopicStrategy",
        "PARAMS": {
            "topic_format": os.environ["TOPIC_STRATEGY_FORMAT"].strip().split(","),
            "date_format": "%Y%m%d",
            "change_hour": 23,
        },
    }
elif os.getenv("CONSUMER_TOPICS"):
    CONSUMER_CONFIG["TOPICS"] = os.environ["CONSUMER_TOPICS"].strip().split(",")
else:
    raise Exception("Add TOPIC_STRATEGY or CONSUMER_TOPICS")

PRODUCER_CONFIG = {
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
    },
    "SCHEMA_PATH": PRODUCER_SCHEMA_PATH,
    "SCHEMA": PRODUCER_SCHEMA
}

if os.getenv("CONSUMER_KAFKA_USERNAME") and os.getenv("CONSUMER_KAFKA_PASSWORD"):
    CONSUMER_CONFIG["PARAMS"]["security.protocol"] = os.getenv("CONSUMER_SECURITY_PROTOCOL", "SASL_SSL")
    CONSUMER_CONFIG["PARAMS"]["sasl.mechanism"] = os.getenv("CONSUMER_SASL_MECHANISM", "SCRAM-SHA-512")
    CONSUMER_CONFIG["PARAMS"]["sasl.username"] = os.getenv("CONSUMER_KAFKA_USERNAME")
    CONSUMER_CONFIG["PARAMS"]["sasl.password"] = os.getenv("CONSUMER_KAFKA_PASSWORD")
if os.getenv("PRODUCER_KAFKA_USERNAME") and os.getenv("PRODUCER_KAFKA_PASSWORD"):
    PRODUCER_CONFIG["PARAMS"]["security.protocol"] = os.getenv(
        "PRODUCER_SECURITY_PROTOCOL", "SASL_PLAINTEXT"
    )
    PRODUCER_CONFIG["PARAMS"]["sasl.mechanism"] = os.getenv(
        "PRODUCER_SASL_MECHANISM", "SCRAM-SHA-256"
    )
    PRODUCER_CONFIG["PARAMS"]["sasl.username"] = os.getenv("PRODUCER_KAFKA_USERNAME")
    PRODUCER_CONFIG["PARAMS"]["sasl.password"] = os.getenv("PRODUCER_KAFKA_PASSWORD")
if os.getenv("METRICS_KAFKA_USERNAME") and os.getenv("METRICS_KAFKA_PASSWORD"):
    METRICS_CONFIG["PARAMS"]["PARAMS"]["security.protocol"] = os.getenv("METRICS_SECURITY_PROTOCOL", "SASL_SSL")
    METRICS_CONFIG["PARAMS"]["PARAMS"]["sasl.mechanism"] = os.getenv("METRICS_SASL_MECHANISM", "SCRAM-SHA-512")
    METRICS_CONFIG["PARAMS"]["PARAMS"]["sasl.username"] = os.getenv(
        "METRICS_KAFKA_USERNAME"
    )
    METRICS_CONFIG["PARAMS"]["PARAMS"]["sasl.password"] = os.getenv(
        "METRICS_KAFKA_PASSWORD"
    )

STEP_METADATA = {
    "STEP_VERSION": os.getenv("STEP_VERSION", "dev"),
    "STEP_ID": os.getenv("STEP_ID", "stamp_classification"),
    "STEP_NAME": os.getenv("STEP_NAME", "stamp_classification"),
    "STEP_COMMENTS": os.getenv("STEP_COMMENTS", ""),
    "CLASSIFIER_NAME": os.getenv("CLASSIFIER_NAME", "stamp_classifier"),
    "CLASSIFIER_VERSION": os.getenv("CLASSIFIER_VERSION", "0.0.0"),
}

STEP_CONFIG = {
    "DB_CONFIG": DB_CONFIG,
    "METRICS_CONFIG": METRICS_CONFIG,
    "STEP_METADATA": STEP_METADATA,
    "N_RETRY": os.getenv("N_RETRY", 5),
    "PRODUCER_CONFIG": PRODUCER_CONFIG,
}

LOGGING_DEBUG = os.getenv("LOGGING_DEBUG", False)
