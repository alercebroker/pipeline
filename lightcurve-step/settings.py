import os
import pathlib
from fastavro import schema
from fastavro.repository.base import SchemaRepositoryError
from credentials import get_credentials

##################################################
#       lightcurve_step   Settings File
##################################################

# SCHEMA PATH RELATIVE TO THE SETTINGS FILE
producer_schema_path = pathlib.Path(
    pathlib.Path(__file__).parent.parent, "schemas/lightcurve_step", "output.avsc"
)
metrics_schema_path = pathlib.Path(
    pathlib.Path(__file__).parent.parent, "schemas/lightcurve_step", "metrics.json"
)
scribe_schema_path = pathlib.Path(
    pathlib.Path(__file__).parent.parent, "schemas/scribe_step", "scribe.avsc"
)


def settings_creator():
    # Set the global logging level to debug
    logging_debug = bool(os.getenv("LOGGING_DEBUG"))

    # Consumer configuration
    # Each consumer has different parameters and can be found in the documentation
    consumer_config = {
        "CLASS": os.getenv("CONSUMER_CLASS", "apf.consumers.KafkaConsumer"),
        "PARAMS": {
            "bootstrap.servers": os.environ["CONSUMER_SERVER"],
            "group.id": os.environ["CONSUMER_GROUP_ID"],
            "auto.offset.reset": "beginning",
            "enable.partition.eof": True
            if os.getenv("ENABLE_PARTITION_EOF")
            else False,
        },
        "TOPICS": os.environ["CONSUMER_TOPICS"].split(","),
        "consume.messages": int(os.getenv("CONSUME_MESSAGES", 50)),
        "consume.timeout": int(os.getenv("CONSUME_TIMEOUT", 15)),
    }

    producer_config = {
        "CLASS": os.getenv("PRODUCER_CLASS", "apf.producers.KafkaProducer"),
        "PARAMS": {
            "bootstrap.servers": os.environ["PRODUCER_SERVER"],
            "message.max.bytes": int(os.getenv("PRODUCER_MESSAGE_MAX_BYTES", 6291456)),
        },
        "TOPIC": os.environ["PRODUCER_TOPIC"],
        "SCHEMA_PATH": os.getenv("PRODUCER_SCHEMA_PATH", producer_schema_path),
    }

    metrics_config = {
        "CLASS": "apf.metrics.KafkaMetricsProducer",
        "EXTRA_METRICS": [
            {"key": "aid", "format": lambda x: str(x)},
            {"key": "candid"},
        ],
        "PARAMS": {
            "PARAMS": {
                "bootstrap.servers": os.environ["METRICS_SERVER"],
            },
            "TOPIC": os.getenv("METRICS_TOPIC", "metrics"),
            "SCHEMA_PATH": os.getenv("METRICS_SCHEMA_PATH", metrics_schema_path),
        },
    }

    if os.getenv("CONSUMER_KAFKA_USERNAME") and os.getenv("CONSUMER_KAFKA_PASSWORD"):
        consumer_config["PARAMS"]["security.protocol"] = "SASL_SSL"
        consumer_config["PARAMS"]["sasl.mechanism"] = "SCRAM-SHA-512"
        consumer_config["PARAMS"]["sasl.username"] = os.getenv(
            "CONSUMER_KAFKA_USERNAME"
        )
        consumer_config["PARAMS"]["sasl.password"] = os.getenv(
            "CONSUMER_KAFKA_PASSWORD"
        )
    if os.getenv("PRODUCER_KAFKA_USERNAME") and os.getenv("PRODUCER_KAFKA_PASSWORD"):
        producer_config["PARAMS"]["security.protocol"] = "SASL_SSL"
        producer_config["PARAMS"]["sasl.mechanism"] = "SCRAM-SHA-512"
        producer_config["PARAMS"]["sasl.username"] = os.getenv(
            "PRODUCER_KAFKA_USERNAME"
        )
        producer_config["PARAMS"]["sasl.password"] = os.getenv(
            "PRODUCER_KAFKA_PASSWORD"
        )
    if os.getenv("METRICS_KAFKA_USERNAME") and os.getenv("METRICS_KAFKA_PASSWORD"):
        metrics_config["PARAMS"]["PARAMS"]["security.protocol"] = "SASL_SSL"
        metrics_config["PARAMS"]["PARAMS"]["sasl.mechanism"] = "SCRAM-SHA-512"
        metrics_config["PARAMS"]["PARAMS"]["sasl.username"] = os.getenv(
            "METRICS_KAFKA_USERNAME"
        )
        metrics_config["PARAMS"]["PARAMS"]["sasl.password"] = os.getenv(
            "METRICS_KAFKA_PASSWORD"
        )

    prometheus = os.getenv("USE_PROMETHEUS", False)
    use_profiling = bool(os.getenv("USE_PROFILING", True))
    pyroscope_server = os.getenv("PYROSCOPE_SERVER", "http://pyroscope.pyroscope:4040")

    # Step Configuration
    step_config = {
        "CONSUMER_CONFIG": consumer_config,
        "PRODUCER_CONFIG": producer_config,
        "METRICS_CONFIG": metrics_config,
        "MONGO_SECRET_NAME": os.getenv("MONGO_SECRET_NAME"),
        "SQL_SECRET_NAME": os.getenv("SQL_SECRET_NAME"),
        "MONGO_CONFIG": {
            "host": os.getenv("MONGO_HOST"),
            "username": os.getenv("MONGO_USERNAME"),
            "password": os.getenv("MONGO_PASSWORD"),
            "port": int(os.getenv("MONGO_PORT", 27017)),
            "database": os.getenv("MONGO_DATABASE"),
            "authSource": os.getenv("MONGO_AUTH_SOURCE"),
        },
        "PSQL_CONFIG": {
            "ENGINE": "postgres",
            "HOST": os.getenv("PSQL_HOST"),
            "USER": os.getenv("PSQL_USERNAME"),
            "PASSWORD": os.getenv("PSQL_PASSWORD"),
            "PORT": int(os.getenv("PSQL_PORT", 5432)),
            "DB_NAME": os.getenv("PSQL_DATABASE"),
        },
        "LOGGING_DEBUG": logging_debug,
        "PYROSCOPE_SERVER": pyroscope_server,
        "FEATURE_FLAGS": {
            "USE_PROFILING": use_profiling,
            "PROMETHEUS": prometheus,
            "USE_SQL": bool(os.getenv("USE_SQL", True)),
            "USE_MONGO": bool(os.getenv("USE_MONGO", True)),
            "SKIP_MJD_FILTER": bool(os.getenv("SKIP_MJD_FILTER", False)),
        },
    }
    return step_config
