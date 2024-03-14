import os

##################################################
#       xmatch_step   Settings File
##################################################

# SCHEMA PATH RELATIVE TO THE SETTINGS FILE
PRODUCER_SCHEMA_PATH = os.path.join(
    os.path.dirname(__file__), os.getenv("PRODUCER_SCHEMA_PATH")
)
METRICS_SCHEMA_PATH = os.path.join(
    os.path.dirname(__file__), os.getenv("METRIS_SCHEMA_PATH")
)
SCRIBE_SCHEMA_PATH = os.path.join(
    os.path.dirname(__file__), os.getenv("SCRIBE_SCHEMA_PATH")
)

# Set the global logging level to debug

LOGGING_DEBUG = os.getenv("LOGGING_DEBUG", False)

CONSUMER_CONFIG = {
    "CLASS": "apf.consumers.KafkaConsumer",
    "PARAMS": {
        "bootstrap.servers": os.environ["CONSUMER_SERVER"],
        "group.id": os.environ["CONSUMER_GROUP_ID"],
        "auto.offset.reset": "beginning",
        "max.poll.interval.ms": 3600000,
    },
    "TOPICS": os.environ["CONSUMER_TOPICS"].split(","),
    "consume.timeout": int(os.getenv("CONSUME_TIMEOUT", 10)),
    "consume.messages": int(os.getenv("CONSUME_MESSAGES", 1000)),
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

# Producer Configuration

PRODUCER_CONFIG = {
    "CLASS": "apf.producers.KafkaProducer",
    "TOPIC": os.environ["PRODUCER_TOPIC"],
    "PARAMS": {
        "bootstrap.servers": os.environ["PRODUCER_SERVER"],
        "message.max.bytes": int(os.getenv("PRODUCER_MESSAGE_MAX_BYTES", 6291456)),
    },
    "SCHEMA_PATH": PRODUCER_SCHEMA_PATH,
}

# Xmatch Configuration

XMATCH_CONFIG = {
    "CATALOG": {
        "name": "allwise",
        "columns": [
            "AllWISE",
            "RAJ2000",
            "DEJ2000",
            "W1mag",
            "W2mag",
            "W3mag",
            "W4mag",
            "e_W1mag",
            "e_W2mag",
            "e_W3mag",
            "e_W4mag",
            "Jmag",
            "e_Jmag",
            "Hmag",
            "e_Hmag",
            "Kmag",
            "e_Kmag",
        ],
    }
}

SCRIBE_PRODUCER_CONFIG = {
    "CLASS": "apf.producers.KafkaProducer",
    "PARAMS": {
        "bootstrap.servers": os.environ["SCRIBE_SERVER"],
    },
    "TOPIC": os.environ["SCRIBE_TOPIC"],
    "SCHEMA_PATH": SCRIBE_SCHEMA_PATH,
}

METRICS_CONFIG = {
    "CLASS": "apf.metrics.KafkaMetricsProducer",
    "EXTRA_METRICS": [
        {"key": "candid"},
        {"key": "oid", "alias": "oid"},
        {"key": "aid", "alias": "aid"},
        {"key": "tid", "format": lambda x: str(x)},
    ],
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

# Step Configuration
STEP_CONFIG = {
    "FEATURE_FLAGS": {
        "PROMETHEUS": bool(os.getenv("USE_PROMETHEUS", True)),
    },
    "CONSUMER_CONFIG": CONSUMER_CONFIG,
    "PRODUCER_CONFIG": PRODUCER_CONFIG,
    "XMATCH_CONFIG": XMATCH_CONFIG,
    "METRICS_CONFIG": METRICS_CONFIG,
    "RETRIES": int(os.getenv("RETRIES", 3)),
    "RETRY_INTERVAL": int(os.getenv("RETRY_INTERVAL", 1)),
    "SCRIBE_PRODUCER_CONFIG": SCRIBE_PRODUCER_CONFIG,
    # "COMMIT": False,           #Disables commit, useful to debug KafkaConsumer
}
