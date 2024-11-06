##################################################
#       features   Settings File
##################################################
import os
import pathlib

EXTRACTOR = os.environ["FEATURE_EXTRACTOR"]

output_schema_path = str(
    pathlib.Path(
        pathlib.Path(__file__).parent.parent,
        "schemas/feature_step",
        "output.avsc",
    )
)
scribe_schema_path = str(
    pathlib.Path(
        pathlib.Path(__file__).parent.parent,
        "schemas/scribe_step",
        "scribe.avsc",
    )
)
metrics_schema_path = str(
    pathlib.Path(
        pathlib.Path(__file__).parent.parent,
        "schemas/feature_step",
        "metrics.json",
    )
)

CONSUMER_CONFIG = {
    "CLASS": os.getenv("CONSUMER_CLASS", "apf.consumers.KafkaConsumer"),
    "TOPICS": os.environ["CONSUMER_TOPICS"].strip().split(","),
    "PARAMS": {
        "bootstrap.servers": os.environ["CONSUMER_SERVER"],
        "group.id": os.environ["CONSUMER_GROUP_ID"],
        "auto.offset.reset": "beginning",
        "max.poll.interval.ms": 3600000,
        "enable.partition.eof": os.getenv("ENABLE_PARTITION_EOF", False),
    },
    "consume.timeout": int(os.getenv("CONSUME_TIMEOUT", 10)),
    "consume.messages": int(os.getenv("CONSUME_MESSAGES", 1000)),
}

PRODUCER_CONFIG = {
    "CLASS": os.getenv("PRODUCER_CLASS", "apf.producers.KafkaProducer"),
    "TOPIC": os.environ["PRODUCER_TOPIC"],
    "PARAMS": {
        "bootstrap.servers": os.environ["PRODUCER_SERVER"],
        "message.max.bytes": int(os.getenv("PRODUCER_MESSAGE_MAX_BYTES", 6291456)),
    },
    "SCHEMA_PATH": os.getenv("PRODUCER_SCHEMA_PATH", output_schema_path),
}

SCRIBE_PRODUCER_CONFIG = {
    "CLASS": os.getenv("SCRIBE_PRODUCER_CLASS", "apf.producers.KafkaProducer"),
    "PARAMS": {
        "bootstrap.servers": os.environ["SCRIBE_SERVER"],
    },
    "TOPIC": os.environ["SCRIBE_TOPIC"],
    "SCHEMA_PATH": os.getenv("SCRIBE_SCHEMA_PATH", scribe_schema_path),
}


METRICS_CONFIG = {
    "CLASS": "apf.metrics.KafkaMetricsProducer",
    "EXTRA_METRICS": [{"key": "aid", "alias": "aid"}, {"key": "candid"}],
    "PARAMS": {
        "PARAMS": {
            "bootstrap.servers": os.environ["METRICS_HOST"],
        },
        "TOPIC": os.environ["METRICS_TOPIC"],
        "SCHEMA_PATH": os.getenv("METRICS_SCHEMA_PATH", metrics_schema_path),
    },
}

SQL_SECRET_NAME = {
    "SQL_SECRET_NAME": os.getenv("SQL_SECRET_NAME"),
}

PSQL_CONFIG = {
    "ENGINE": "postgres",
    "HOST": os.getenv("PSQL_HOST"),
    "USER": os.getenv("PSQL_USERNAME"),
    "PASSWORD": os.getenv("PSQL_PASSWORD"),
    "PORT": int(os.getenv("PSQL_PORT", 5432)),
    "DB_NAME": os.getenv("PSQL_DATABASE"),
}

if os.getenv("KAFKA_USERNAME") and os.getenv("KAFKA_PASSWORD"):
    CONSUMER_CONFIG["PARAMS"]["security.protocol"] = "SASL_SSL"
    CONSUMER_CONFIG["PARAMS"]["sasl.mechanism"] = "SCRAM-SHA-512"
    CONSUMER_CONFIG["PARAMS"]["sasl.username"] = os.getenv("KAFKA_USERNAME")
    CONSUMER_CONFIG["PARAMS"]["sasl.password"] = os.getenv("KAFKA_PASSWORD")
    PRODUCER_CONFIG["PARAMS"]["security.protocol"] = "SASL_SSL"
    PRODUCER_CONFIG["PARAMS"]["sasl.mechanism"] = "SCRAM-SHA-512"
    PRODUCER_CONFIG["PARAMS"]["sasl.username"] = os.getenv("KAFKA_USERNAME")
    PRODUCER_CONFIG["PARAMS"]["sasl.password"] = os.getenv("KAFKA_PASSWORD")

    SCRIBE_PRODUCER_CONFIG["PARAMS"]["security.protocol"] = "SASL_SSL"
    SCRIBE_PRODUCER_CONFIG["PARAMS"]["sasl.mechanism"] = "SCRAM-SHA-512"
    SCRIBE_PRODUCER_CONFIG["PARAMS"]["sasl.username"] = os.getenv("KAFKA_USERNAME")
    SCRIBE_PRODUCER_CONFIG["PARAMS"]["sasl.password"] = os.getenv("KAFKA_PASSWORD")

    METRICS_CONFIG["PARAMS"]["PARAMS"]["security.protocol"] = "SASL_SSL"
    METRICS_CONFIG["PARAMS"]["PARAMS"]["sasl.mechanism"] = "SCRAM-SHA-512"
    METRICS_CONFIG["PARAMS"]["PARAMS"]["sasl.username"] = os.getenv("KAFKA_USERNAME")
    METRICS_CONFIG["PARAMS"]["PARAMS"]["sasl.password"] = os.getenv("KAFKA_PASSWORD")


use_profiling = bool(os.getenv("USE_PROFILING", True))
pyroscope_server = os.getenv("PYROSCOPE_SERVER", "http://pyroscope.pyroscope:4040")

STEP_CONFIG = {
    "EXTRACTOR": EXTRACTOR,
    "CONSUMER_CONFIG": CONSUMER_CONFIG,
    "PRODUCER_CONFIG": PRODUCER_CONFIG,
    "SCRIBE_PRODUCER_CONFIG": SCRIBE_PRODUCER_CONFIG,
    "METRICS_CONFIG": METRICS_CONFIG,
    "FEATURE_FLAGS": {
        "USE_PROFILING": use_profiling,
    },
    "PYROSCOPE_SERVER": pyroscope_server,
    "SQL_SECRET_NAME": os.getenv("SQL_SECRET_NAME"),
    "PSQL_CONFIG": PSQL_CONFIG,
}
