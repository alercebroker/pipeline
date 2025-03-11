##################################################
#       sorting_hat_step   Settings File
##################################################
import os
import pathlib
from typing import Any, Callable, TypedDict

producer_schema_path = pathlib.Path(
    pathlib.Path(__file__).parent.parent, "schemas/sorting_hat_step", "output.avsc"
)
metrics_schema_path = pathlib.Path(
    pathlib.Path(__file__).parent.parent, "schemas/sorting_hat_step", "metrics.json"
)
scribe_schema_path = pathlib.Path(
    pathlib.Path(__file__).parent.parent, "schemas/scribe_step", "scribe.avsc"
)

# Set the global logging level to debug
LOGGING_DEBUG = bool(os.getenv("LOGGING_DEBUG", False))

# Export prometheus metrics
PROMETHEUS = True


# Consumer configuration
# Each consumer has different parameters and can be found in the documentation
TopicStrategy = TypedDict(
    "TopicStrategy",
    {
        "CLASS": str,
        "PARAMS": dict[str, Any],
    },
)

ConsumerConfig = TypedDict(
    "ConsumerConfig",
    {
        "CLASS": str,
        "PARAMS": dict[str, Any],
        "TOPIC_STRATEGY": TopicStrategy | None,
        "TOPICS": list[str] | None,
        "SCHEMA_PATH": str | None,
        "consume.timeout": int,
        "consume.messages": int,
    },
)


CONSUMER_CONFIG: ConsumerConfig = {
    "CLASS": os.getenv("CONSUMER_CLASS", "apf.consumers.KafkaConsumer"),
    "PARAMS": {
        "bootstrap.servers": os.environ["CONSUMER_SERVER"],
        "group.id": os.environ["CONSUMER_GROUP_ID"],
        "auto.offset.reset": "beginning",
        "max.poll.interval.ms": 3600000,
    },
    "TOPIC_STRATEGY": None,
    "TOPICS": None,
    "SCHEMA_PATH": None,
    "consume.timeout": int(os.getenv("CONSUME_TIMEOUT", 10)),
    "consume.messages": int(os.getenv("CONSUME_MESSAGES", 100)),
}

if os.getenv("TOPIC_STRATEGY_TOPIC_FORMAT"):
    CONSUMER_CONFIG["TOPIC_STRATEGY"] = {
        "CLASS": "apf.core.topic_management.DailyTopicStrategy",
        "PARAMS": {
            "topic_format": os.environ["TOPIC_STRATEGY_TOPIC_FORMAT"]
            .strip()
            .split(","),
            "date_format": os.getenv("TOPIC_STRATEGY_DATE_FORMAT", "%Y%m%d"),
            "change_hour": int(os.getenv("TOPIC_STRATEGY_CHANGE_HOUR", 23)),
        },
    }
elif os.getenv("CONSUMER_TOPICS"):
    CONSUMER_CONFIG["TOPICS"] = os.environ["CONSUMER_TOPICS"].strip().split(",")
else:
    raise Exception("Add TOPIC_STRATEGY or CONSUMER_TOPICS")

if os.getenv("CONSUMER_CLASS") == "apf.consumers.KafkaSchemalessConsumer":
    try:
        CONSUMER_CONFIG["SCHEMA_PATH"] = os.path.join(
            os.path.dirname(__file__), "schemas/elasticc/elasticc.v0_9_1.alert.avsc"
        )
        assert os.path.exists(CONSUMER_CONFIG["SCHEMA_PATH"])
    except AssertionError:
        CONSUMER_CONFIG["SCHEMA_PATH"] = os.path.join(
            os.path.dirname(__file__),
            "sorting_hat_step/schemas/elasticc/elasticc.v0_9_1.alert.avsc",
        )
        assert os.path.exists(CONSUMER_CONFIG["SCHEMA_PATH"])

# Producer configuration
ProducerConfig = TypedDict(
    "ProducerConfig",
    {
        "CLASS": str,
        "PARAMS": dict[str, Any],
        "TOPIC": str,
        "SCHEMA_PATH": str,
    },
)
PRODUCER_CONFIG: ProducerConfig = {
    "CLASS": os.getenv("PRODUCER_CLASS", "apf.producers.KafkaProducer"),
    "TOPIC": os.environ["PRODUCER_TOPIC"],
    "PARAMS": {
        "bootstrap.servers": os.environ["PRODUCER_SERVER"],
        "message.max.bytes": int(os.getenv("PRODUCER_MESSAGE_MAX_BYTES", 6291456)),
    },
    "SCHEMA_PATH": os.getenv("PRODUCER_SCHEMA_PATH", str(producer_schema_path)),
}

ExtraMetric = TypedDict("ExtraMetric", {"key": str, "format": Callable[[Any], str]})
MetricConfig = TypedDict(
    "MetricConfig",
    {"CLASS": str, "PARAMS": dict[str, Any], "EXTRA_METRICS": list[ExtraMetric]},
)
METRICS_CONFIG: MetricConfig = {
    "CLASS": os.getenv("METRICS_CLASS", "apf.metrics.KafkaMetricsProducer"),
    "EXTRA_METRICS": [
        {"key": "candid", "format": str},
    ],
    "PARAMS": {
        "PARAMS": {
            "bootstrap.servers": os.getenv("METRICS_HOST"),
        },
        "TOPIC": os.getenv("METRICS_TOPIC", "metrics"),
        "SCHEMA_PATH": os.getenv("METRICS_SCHEMA_PATH", str(metrics_schema_path)),
    },
}

if os.getenv("CONSUMER_KAFKA_USERNAME") and os.getenv("CONSUMER_KAFKA_PASSWORD"):
    CONSUMER_CONFIG["PARAMS"]["security.protocol"] = "SASL_SSL"
    CONSUMER_CONFIG["PARAMS"]["sasl.mechanism"] = "SCRAM-SHA-512"
    CONSUMER_CONFIG["PARAMS"]["sasl.username"] = os.getenv("CONSUMER_KAFKA_USERNAME")
    CONSUMER_CONFIG["PARAMS"]["sasl.password"] = os.getenv("CONSUMER_KAFKA_PASSWORD")
if os.getenv("PRODUCER_KAFKA_USERNAME") and os.getenv("PRODUCER_KAFKA_PASSWORD"):
    PRODUCER_CONFIG["PARAMS"]["security.protocol"] = "SASL_SSL"
    PRODUCER_CONFIG["PARAMS"]["sasl.mechanism"] = "SCRAM-SHA-512"
    PRODUCER_CONFIG["PARAMS"]["sasl.username"] = os.getenv("PRODUCER_KAFKA_USERNAME")
    PRODUCER_CONFIG["PARAMS"]["sasl.password"] = os.getenv("PRODUCER_KAFKA_PASSWORD")
if os.getenv("METRICS_KAFKA_USERNAME") and os.getenv("METRICS_KAFKA_PASSWORD"):
    METRICS_CONFIG["PARAMS"]["PARAMS"]["security.protocol"] = "SASL_SSL"
    METRICS_CONFIG["PARAMS"]["PARAMS"]["sasl.mechanism"] = "SCRAM-SHA-512"
    METRICS_CONFIG["PARAMS"]["PARAMS"]["sasl.username"] = os.getenv(
        "METRICS_KAFKA_USERNAME"
    )
    METRICS_CONFIG["PARAMS"]["PARAMS"]["sasl.password"] = os.getenv(
        "METRICS_KAFKA_PASSWORD"
    )


# Step Configuration
StepConfig = TypedDict(
    "StepConfig",
    {
        "FEATURE_FLAGS": dict[str, bool],
        "PSQL_CONFIG": dict[str, str | int | None],
        "PSQL_SECRET_NAME": str | None,
        "SURVEY_STRATEGY": str | None,
        "CONSUMER_CONFIG": ConsumerConfig,
        "PRODUCER_CONFIG": ProducerConfig,
        "METRICS_CONFIG": MetricConfig,
        "LOGGING_DEBUG": bool,
        "PYROSCOPE_SERVER": str,
    },
)

STEP_CONFIG: StepConfig = {
    "FEATURE_FLAGS": {
        "USE_PROFILING": bool(os.getenv("USE_PROFILING", True)),
        "PROMETHEUS": PROMETHEUS,
    },
    "PSQL_CONFIG": {
        "ENGINE": "postgres",
        "HOST": os.getenv("PSQL_HOST"),
        "USERNAME": os.getenv("PSQL_USERNAME"),
        "PASSWORD": os.getenv("PSQL_PASSWORD"),
        "PORT": int(os.getenv("PSQL_PORT", 5432)),
        "DBNAME": os.getenv("PSQL_DATABASE"),
    },
    "PSQL_SECRET_NAME": os.getenv("PSQL_SECRET_NAME"),
    "SURVEY_STRATEGY": "None",
    "CONSUMER_CONFIG": CONSUMER_CONFIG,
    "PRODUCER_CONFIG": PRODUCER_CONFIG,
    "METRICS_CONFIG": METRICS_CONFIG,
    "LOGGING_DEBUG": LOGGING_DEBUG,
    "PYROSCOPE_SERVER": os.getenv("PYROSCOPE_SERVER", ""),
}
