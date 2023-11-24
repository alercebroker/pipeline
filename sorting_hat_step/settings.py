##################################################
#       sorting_hat_step   Settings File
##################################################
import os
import pathlib
from credentials import get_credentials

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
LOGGING_DEBUG = os.getenv("LOGGING_DEBUG", False)

# Export prometheus metrics
PROMETHEUS = True

# Consumer configuration
# Each consumer has different parameters and can be found in the documentation
CONSUMER_CONFIG = {
    "CLASS": os.getenv("CONSUMER_CLASS", "apf.consumers.KafkaConsumer"),
    "PARAMS": {
        "bootstrap.servers": os.environ["CONSUMER_SERVER"],
        "group.id": os.environ["CONSUMER_GROUP_ID"],
        "auto.offset.reset": "beginning",
        "max.poll.interval.ms": 3600000,
    },
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
PRODUCER_CONFIG = {
    "CLASS": os.getenv("PRODUCER_CLASS", "apf.producers.KafkaProducer"),
    "TOPIC": os.environ["PRODUCER_TOPIC"],
    "PARAMS": {
        "bootstrap.servers": os.environ["PRODUCER_SERVER"],
        "message.max.bytes": int(os.getenv("PRODUCER_MESSAGE_MAX_BYTES", 6291456)),
    },
    "SCHEMA_PATH": os.getenv("PRODUCER_SCHEMA_PATH", producer_schema_path),
}


METRICS_CONFIG = {
    "CLASS": os.getenv("METRICS_CLASS", "apf.metrics.KafkaMetricsProducer"),
    "EXTRA_METRICS": [
        {"key": "candid", "format": lambda x: str(x)},
    ],
    "PARAMS": {
        "PARAMS": {
            "bootstrap.servers": os.getenv("METRICS_HOST"),
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
STEP_CONFIG = {
    "FEATURE_FLAGS": {
        "RUN_CONESEARCH": os.getenv("RUN_CONESEARCH", True),
        "USE_PSQL": os.getenv("USE_PSQL", False),
        "USE_PROFILING": os.getenv("USE_PROFILING", "True"),
        "PROMETHEUS": PROMETHEUS,
    },
    "MONGO_CONFIG": {
        "HOST": os.getenv("MONGO_HOST"),
        "USERNAME": os.getenv("MONGO_USERNAME"),
        "PASSWORD": os.getenv("MONGO_PASSWORD"),
        "PORT": int(os.getenv("MONGO_PORT", 27017)),
        "DATABASE": os.getenv("MONGO_DATABASE"),
    },
    "PSQL_CONFIG": {
        "ENGINE": "postgres",
        "HOST": os.getenv("PSQL_HOST"),
        "USERNAME": os.getenv("PSQL_USERNAME"),
        "PASSWORD": os.getenv("PSQL_PASSWORD"),
        "PORT": int(os.getenv("PSQL_PORT", 5432)),
        "DBNAME": os.getenv("PSQL_DATABASE"),
    },
    "MONGO_SECRET_NAME": os.getenv("MONGO_SECRET_NAME"),
    "PSQL_SECRET_NAME": os.getenv("PSQL_SECRET_NAME"),
    "CONSUMER_CONFIG": CONSUMER_CONFIG,
    "PRODUCER_CONFIG": PRODUCER_CONFIG,
    "METRICS_CONFIG": METRICS_CONFIG,
    "LOGGING_DEBUG": LOGGING_DEBUG,
    "PYROSCOPE_SERVER": os.getenv("PYROSCOPE_SERVER", ""),
}
