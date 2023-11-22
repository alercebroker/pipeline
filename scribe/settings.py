import os
import pathlib

##################################################
#       mongo_scribe   Settings File
##################################################

metrics_schema_path = pathlib.Path(
    pathlib.Path(__file__).parent.parent, "schemas/scribe_step", "metrics.json"
)

## Consumer configuration
### Each consumer has different parameters and can be found in the documentation
CONSUMER_CONFIG = {
    "CLASS": os.getenv("CONSUMER_CLASS", "apf.consumers.KafkaConsumer"),
    "PARAMS": {
        "bootstrap.servers": os.environ["CONSUMER_SERVER"],
        "group.id": os.environ["CONSUMER_GROUP_ID"],
        "auto.offset.reset": "beginning",
    },
    "TOPICS": os.environ["TOPICS"].strip().split(","),
    "NUM_MESSAGES": int(os.getenv("NUM_MESSAGES", "50")),
    "TIMEOUT": int(os.getenv("TIMEOUT", "10")),
}

if os.getenv("KAFKA_USERNAME") and os.getenv("KAFKA_PASSWORD"):
    CONSUMER_CONFIG["PARAMS"]["security.protocol"] = "SASL_SSL"
    CONSUMER_CONFIG["PARAMS"]["sasl.mechanism"] = "SCRAM-SHA-512"
    CONSUMER_CONFIG["PARAMS"]["sasl.username"] = os.getenv("KAFKA_USERNAME")
    CONSUMER_CONFIG["PARAMS"]["sasl.password"] = os.getenv("KAFKA_PASSWORD")


METRICS_CONFIG = {
    "CLASS": "apf.metrics.KafkaMetricsProducer",
    "PARAMS": {
        "PARAMS": {
            "bootstrap.servers": os.environ["METRICS_HOST"],
        },
        "TOPIC": os.environ["METRICS_TOPIC"],
        "SCHEMA_PATH": os.getenv("METRICS_SCHEMA_PATH", metrics_schema_path),
    },
}

if os.getenv("METRICS_KAFKA_USERNAME") and os.getenv("METRICS_KAFKA_PASSWORD"):
    METRICS_CONFIG["PARAMS"]["PARAMS"]["security.protocol"] = "SASL_SSL"
    METRICS_CONFIG["PARAMS"]["PARAMS"]["sasl.mechanism"] = "SCRAM-SHA-512"
    METRICS_CONFIG["PARAMS"]["PARAMS"]["sasl.username"] = os.getenv(
        "METRICS_KAFKA_USERNAME"
    )
    METRICS_CONFIG["PARAMS"]["PARAMS"]["sasl.password"] = os.getenv(
        "METRICS_KAFKA_PASSWORD"
    )

## Step Configuration
STEP_CONFIG = {
    "DB_TYPE": os.getenv("DB_ENGINE", "mongo"),
    "CONSUMER_CONFIG": CONSUMER_CONFIG,
    "METRICS_CONFIG": METRICS_CONFIG,
    "FEATURE_FLAGS": {
        "PROMETHEUS": bool(os.getenv("USE_PROMETHEUS", "True")),
        "USE_PROFILING": bool(os.getenv("USE_PROFILING", True)),
    },
    "PYROSCOPE_SERVER": os.getenv(
        "PYROSCOPE_SERVER", "http://pyroscope.pyroscope:4040"
    ),
}
