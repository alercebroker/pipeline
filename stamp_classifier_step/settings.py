##################################################
#       stamp_classifier_step   Settings File
##################################################

import os
from schema import SCHEMA, SCRIBE_SCHEMA

# SCHEMA PATH RELATIVE TO THE SETTINGS FILE
PRODUCER_SCHEMA_PATH = os.path.join(os.path.dirname(__file__), os.getenv("PRODUCER_SCHEMA_PATH"))
METRICS_SCHEMA_PATH = os.path.join(os.path.dirname(__file__), os.getenv("METRIS_SCHEMA_PATH"))
SCRIBE_SCHEMA_PATH =  os.path.join(os.path.dirname(__file__), os.getenv("SCRIBE_SCHEMA_PATH"))

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
    },
    "SCHEMA_PATH": PRODUCER_SCHEMA_PATH,
}

SCRIBE_PRODUCER_CONFIG = {
    "TOPIC": os.environ["SCRIBE_TOPIC"],
    "PARAMS": {
        "bootstrap.servers": os.environ["SCRIBE_SERVER"],
    },
    "SCHEMA_PATH": SCRIBE_SCHEMA_PATH,
}

CLASSIFIER_STRATEGY = os.environ["CLASSIFIER_STRATEGY"]

METRICS_CONFIG = {
    "CLASS": "apf.metrics.KafkaMetricsProducer",
    "EXTRA_METRICS": [{"key": "candid", "format": lambda x: str(x)}, "oid"],
    "PARAMS": {
        "PARAMS": {
            "bootstrap.servers": os.environ["METRICS_HOST"],
            "auto.offset.reset": "smallest",
        },
        "TOPIC": os.environ["METRICS_TOPIC"],
        "SCHEMA_PATH": METRICS_SCHEMA_PATH,
    },
}

if os.getenv("CONSUMER_KAFKA_USERNAME") and os.getenv("CONSUMER_KAFKA_PASSWORD"):
    CONSUMER_CONFIG["PARAMS"]["security.protocol"] = "SASL_SSL"
    CONSUMER_CONFIG["PARAMS"]["sasl.mechanism"] = "SCRAM-SHA-512"
    CONSUMER_CONFIG["PARAMS"]["sasl.username"] = os.getenv("CONSUMER_KAFKA_USERNAME")
    CONSUMER_CONFIG["PARAMS"]["sasl.password"] = os.getenv("CONSUMER_KAFKA_PASSWORD")
if os.getenv("PRODUCER_KAFKA_USERNAME") and os.getenv("PRODUCER_KAFKA_PASSWORD"):
    OUTPUT_PRODUCER_CONFIG["PARAMS"]["security.protocol"] = os.getenv(
        "PRODUCER_SECURITY_PROTOCOL", "SASL_PLAINTEXT"
    )
    OUTPUT_PRODUCER_CONFIG["PARAMS"]["sasl.mechanism"] = os.getenv(
        "PRODUCER_SASL_MECHANISM", "SCRAM-SHA-256"
    )
    OUTPUT_PRODUCER_CONFIG["PARAMS"]["sasl.username"] = os.getenv(
        "PRODUCER_KAFKA_USERNAME"
    )
    OUTPUT_PRODUCER_CONFIG["PARAMS"]["sasl.password"] = os.getenv(
        "PRODUCER_KAFKA_PASSWORD"
    )
if os.getenv("SCRIBE_KAFKA_USERNAME") and os.getenv("SCRIBE_KAFKA_PASSWORD"):
    SCRIBE_PRODUCER_CONFIG["PARAMS"]["security.protocol"] = "SASL_SSL"
    SCRIBE_PRODUCER_CONFIG["PARAMS"]["sasl.mechanism"] = "SCRAM-SHA-512"
    SCRIBE_PRODUCER_CONFIG["PARAMS"]["sasl.username"] = os.getenv(
        "SCRIBE_KAFKA_USERNAME"
    )
    SCRIBE_PRODUCER_CONFIG["PARAMS"]["sasl.password"] = os.getenv(
        "SCRIBE_KAFKA_PASSWORD"
    )
if os.getenv("METRICS_KAFKA_USERNAME") and os.getenv("METRICS_KAFKA_PASSWORD"):
    METRICS_CONFIG["PARAMS"]["PARAMS"]["security.protocol"] = "SASL_SSL"
    METRICS_CONFIG["PARAMS"]["PARAMS"]["sasl.mechanism"] = "SCRAM-SHA-512"
    METRICS_CONFIG["PARAMS"]["PARAMS"]["sasl.username"] = os.getenv(
        "METRICS_KAFKA_USERNAME"
    )
    METRICS_CONFIG["PARAMS"]["PARAMS"]["sasl.password"] = os.getenv(
        "METRICS_KAFKA_PASSWORD"
    )

STEP_CONFIG = {
    "PRODUCER_CONFIG": OUTPUT_PRODUCER_CONFIG,
    "SCRIBE_CONFIG": SCRIBE_PRODUCER_CONFIG,
    "MODEL_NAME": os.getenv("MODEL_NAME"),
    "MODEL_VERSION": os.getenv("MODEL_VERSION"),
    "METRICS_CONFIG": METRICS_CONFIG,
}
