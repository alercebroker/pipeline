import os
import pathlib

metrics_schema_path = pathlib.Path(pathlib.Path(__file__).parent.parent, "schemas/metadata_step", "metrics.json")
## Set the global logging level to debug
LOGGING_DEBUG = os.getenv("LOGGING_DEBUG", False)


## Consumer configuration
### Each consumer has different parameters and can be found in the documentation
CONSUMER_CONFIG = {
    "CLASS": os.getenv("CONSUMER_CLASS", "apf.consumers.KafkaConsumer"),
    "PARAMS": {
        "bootstrap.servers": os.environ["CONSUMER_SERVER"],
        "group.id": os.environ["CONSUMER_GROUP_ID"],
        "auto.offset.reset": "beginning",
        "enable.partition.eof": bool(os.getenv("ENABLE_PARTITION_EOF")),
    },
    "TOPICS": os.environ["CONSUMER_TOPICS"].split(","),
    "consume.messages": int(os.getenv("CONSUME_MESSAGES", 250)),
    "consume.timeout": int(os.getenv("CONSUME_TIMEOUT", 10)),
}

METRICS_CONFIG = {
    "CLASS": "apf.metrics.KafkaMetricsProducer",
    "EXTRA_METRICS": [
        {"key": "aid"},
        {"key": "candid"},
    ],
    "PARAMS": {
        "PARAMS": {
            "bootstrap.servers": os.getenv("METRICS_SERVER"),
        },
        "TOPIC": os.getenv("METRICS_TOPIC", "metrics"),
        "SCHEMA_PATH": os.getenv("METRICS_SCHEMA_PATH", metrics_schema_path),
    },
}

if os.getenv("CONSUMER_KAFKA_USERNAME") and os.getenv("CONSUMER_KAFKA_PASSWORD"):
    CONSUMER_CONFIG["PARAMS"]["security.protocol"] = "SASL_SSL"
    CONSUMER_CONFIG["PARAMS"]["sasl.mechanism"] = "SCRAM-SHA-512"
    CONSUMER_CONFIG["PARAMS"]["sasl.username"] = os.getenv("CONSUMER_KAFKA_USERNAME")
    CONSUMER_CONFIG["PARAMS"]["sasl.password"] = os.getenv("CONSUMER_KAFKA_PASSWORD")
if os.getenv("METRICS_KAFKA_USERNAME") and os.getenv("METRICS_KAFKA_PASSWORD"):
    METRICS_CONFIG["PARAMS"]["PARAMS"]["security.protocol"] = "SASL_SSL"
    METRICS_CONFIG["PARAMS"]["PARAMS"]["sasl.mechanism"] = "SCRAM-SHA-512"
    METRICS_CONFIG["PARAMS"]["PARAMS"]["sasl.username"] = os.getenv("METRICS_KAFKA_USERNAME")
    METRICS_CONFIG["PARAMS"]["PARAMS"]["sasl.password"] = os.getenv("METRICS_KAFKA_PASSWORD")

## Step Configuration
STEP_CONFIG = {
    "CONSUMER_CONFIG": CONSUMER_CONFIG,
    "METRICS_CONFIG": METRICS_CONFIG,
    "LOGGING_DEBUG": LOGGING_DEBUG,
    "DATABASE_SECRET_NAME": os.environ["SQL_SECRET_NAME"],
}
