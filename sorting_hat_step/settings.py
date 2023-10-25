##################################################
#       sorting_hat_step   Settings File
##################################################
import os
from credentials import get_credentials
from schemas.output_schema import SCHEMA
from schemas.scribe_schema import SCRIBE_SCHEMA
from fastavro.repository.base import SchemaRepositoryError

# Set the global logging level to debug
LOGGING_DEBUG = os.getenv("LOGGING_DEBUG", False)

# Export prometheus metrics
PROMETHEUS = True

MONGO_CONFIG = get_credentials(os.environ["MONGODB_SECRET_NAME"], secret_type="mongo")
PSQL_CONFIG = {}

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
    "SCHEMA": SCHEMA,
}

SCRIBE_PRODUCER_CONFIG = {
    "CLASS": os.getenv("SCRIBE_PRODUCER_CLASS", "apf.producers.KafkaProducer"),
    "PARAMS": {
        "bootstrap.servers": os.environ["SCRIBE_PRODUCER_SERVER"],
    },
    "TOPIC": os.environ["SCRIBE_PRODUCER_TOPIC"], #w_metadata
    "SCHEMA": SCRIBE_SCHEMA,
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
if os.getenv("SCRIBE_KAFKA_USERNAME") and os.getenv("SCRIBE_KAFKA_PASSWORD"):
    SCRIBE_PRODUCER_CONFIG["PARAMS"]["security.protocol"] = "SASL_SSL"
    SCRIBE_PRODUCER_CONFIG["PARAMS"]["sasl.mechanism"] = "SCRAM-SHA-512"
    SCRIBE_PRODUCER_CONFIG["PARAMS"]["sasl.username"] = os.getenv(
        "SCRIBE_KAFKA_USERNAME"
    )
    SCRIBE_PRODUCER_CONFIG["PARAMS"]["sasl.password"] = os.getenv(
        "SCRIBE_KAFKA_PASSWORD"
    )

## Feature flags
RUN_CONESEARCH = os.getenv("RUN_CONESEARCH", "True")
USE_PSQL = os.getenv("USE_PSQL", "False")

if USE_PSQL.lower() == "true":
    PSQL_CONFIG = get_credentials(os.environ["PSQL_SECRET_NAME"], secret_type="psql")
# Step Configuration
STEP_CONFIG = {
    "PROMETHEUS": PROMETHEUS,
    "MONGO_CONFIG": MONGO_CONFIG,
    "PSQL_CONFIG": PSQL_CONFIG,
    "CONSUMER_CONFIG": CONSUMER_CONFIG,
    "PRODUCER_CONFIG": PRODUCER_CONFIG,
    "SCRIBE_PRODUCER_CONFIG": SCRIBE_PRODUCER_CONFIG,
    "METRICS_CONFIG": METRICS_CONFIG,
    "RUN_CONESEARCH": RUN_CONESEARCH,
    "USE_PSQL": USE_PSQL,
}
