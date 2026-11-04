import os

##################################################
#       delight_step   Settings File
##################################################

# SCHEMA PATH RELATIVE TO THE SETTINGS FILE
CONSUMER_SCHEMA_PATH = os.path.join(
    os.path.dirname(__file__), os.getenv("CONSUMER_SCHEMA_PATH")
)
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
    "CLASS": "apf.consumers.KafkaSchemalessConsumer",
    "SCHEMA_PATH": CONSUMER_SCHEMA_PATH,
    "PARAMS": {
      "bootstrap.servers": os.environ["CONSUMER_SERVER"],
      "group.id": "delight",
    },
    "TOPICS": os.environ["CONSUMER_TOPIC"],
    "consume.timeout": 10,
    "consume.messages": 200,
}

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

# Delight Configuration

DELIGHT_CONFIG = {
    "filter": {
        "class": os.getenv("DELIGHT_FILTER_CLASS", ""),
        "clasifier": os.getenv("DELIGHT_FILTER_CLASSIFIER", ""),
        "prob": os.getenv("DELIGHT_FILTER_PROBABILITY", ""),
    },
    "calc_dispersion": True if os.getenv("DELIGHT_CALC_DISPERSION", "False") == "True" else False,
    "calc_galaxy_size": True if os.getenv("DELIGHT_CALC_GAL_SIZE", "FALSE") == "True" else False,
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

# agregar psql config. Ejemplo en lightcurve step
PSQL_CONFIG = {
            "ENGINE": "postgres",
            "HOST": os.getenv("PSQL_HOST"),
            "USER": os.getenv("PSQL_USERNAME"),
            "PASSWORD": os.getenv("PSQL_PASSWORD"),
            "PORT": int(os.getenv("PSQL_PORT", 5432)),
            "DB_NAME": os.getenv("PSQL_DATABASE"),
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
    "DELIGHT_CONFIG": DELIGHT_CONFIG,
    "METRICS_CONFIG": METRICS_CONFIG,
    "SCRIBE_PRODUCER_CONFIG": SCRIBE_PRODUCER_CONFIG,
    "PSQL_CONFIG": PSQL_CONFIG
}