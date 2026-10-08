##################################################
#   LC Classification Multisurvey Settings File
##################################################
import os
import pathlib

from models_settings import configurator


def model_config_factory():
    return configurator(os.environ["MODEL_CLASS"])


def config():
    CONSUMER_CONFIG = {
        "CLASS": os.getenv("CONSUMER_CLASS", "apf.consumers.KafkaConsumer"),
        "TOPICS": os.environ["CONSUMER_TOPICS"].strip().split(","),
        "PARAMS": {
            "bootstrap.servers": os.environ["CONSUMER_SERVER"],
            "group.id": os.environ["CONSUMER_GROUP_ID"],
            "auto.offset.reset": "beginning",
            "enable.partition.eof": bool(os.getenv("ENABLE_PARTITION_EOF", None)),
        },
        "consume.timeout": int(os.getenv("CONSUME_TIMEOUT", 10)),
        "consume.messages": int(os.getenv("CONSUME_MESSAGES", 100)),
    }

    scribe_schema_path = str(
        pathlib.Path(
            pathlib.Path(__file__).parent.parent, "schemas/scribe_step", "scribe.avsc"
        )
    )
    SCRIBE_PRODUCER_CONFIG = {
        "CLASS": os.getenv("SCRIBE_PRODUCER_CLASS", "apf.producers.KafkaProducer"),
        "PARAMS": {"bootstrap.servers": os.environ["SCRIBE_SERVER"]},
        "TOPIC": os.environ["SCRIBE_TOPIC"],
        "SCHEMA_PATH": os.getenv("SCRIBE_SCHEMA_PATH", scribe_schema_path),
    }

    # Optional daily topic output, same schema as the legacy lc_classification_step.
    # Without PRODUCER_SERVER nothing is produced.
    PRODUCER_CONFIG = {}
    if os.getenv("PRODUCER_SERVER"):
        producer_schema_path = str(
            pathlib.Path(
                pathlib.Path(__file__).parent.parent,
                "schemas/lc_classification_step",
                "output_ztf.avsc",
            )
        )
        PRODUCER_CONFIG = {
            "CLASS": os.getenv("PRODUCER_CLASS", "apf.producers.kafka.KafkaProducer"),
            "TOPIC_STRATEGY": {
                "CLASS": os.getenv(
                    "PRODUCER_TOPIC_STRATEGY_CLASS",
                    "lc_classification_multisurvey_step.topic_strategy.CurrentDailyTopicStrategy",
                ),
                "PARAMS": {
                    "topic_format": os.environ["PRODUCER_TOPIC_FORMAT"],
                    "date_format": os.getenv("PRODUCER_DATE_FORMAT", "%Y%m%d"),
                    "change_hour": int(os.getenv("PRODUCER_CHANGE_HOUR", 23)),
                    "retention_days": int(os.getenv("PRODUCER_RETENTION_DAYS", 1)),
                },
            },
            "PARAMS": {"bootstrap.servers": os.environ["PRODUCER_SERVER"]},
            "SCHEMA_PATH": os.getenv("PRODUCER_SCHEMA_PATH", producer_schema_path),
        }

    METRICS_CONFIG = {}
    if os.getenv("METRICS_HOST"):
        metrics_schema_path = str(
            pathlib.Path(
                pathlib.Path(__file__).parent.parent,
                "schemas/lc_classification_step",
                "metrics.json",
            )
        )
        METRICS_CONFIG = {
            "CLASS": "apf.metrics.KafkaMetricsProducer",
            "PARAMS": {
                "PARAMS": {"bootstrap.servers": os.environ["METRICS_HOST"]},
                "TOPIC": os.environ["METRICS_TOPIC"],
                "SCHEMA_PATH": os.getenv("METRICS_SCHEMA_PATH", metrics_schema_path),
            },
        }

    PSQL_CONFIG = {
        "HOST": os.environ["PSQL_HOST"],
        "USER": os.environ["PSQL_USER"],
        "PASSWORD": os.environ["PSQL_PASSWORD"],
        "PORT": int(os.getenv("PSQL_PORT", 5432)),
        "DB_NAME": os.environ["PSQL_DATABASE"],
        "SCHEMA": os.getenv("PSQL_SCHEMA", "multisurvey_ztf"),
    }

    if os.getenv("CONSUMER_KAFKA_USERNAME") and os.getenv("CONSUMER_KAFKA_PASSWORD"):
        CONSUMER_CONFIG["PARAMS"]["security.protocol"] = "SASL_SSL"
        CONSUMER_CONFIG["PARAMS"]["sasl.mechanism"] = "SCRAM-SHA-512"
        CONSUMER_CONFIG["PARAMS"]["sasl.username"] = os.getenv("CONSUMER_KAFKA_USERNAME")
        CONSUMER_CONFIG["PARAMS"]["sasl.password"] = os.getenv("CONSUMER_KAFKA_PASSWORD")
    if PRODUCER_CONFIG and os.getenv("PRODUCER_KAFKA_USERNAME") and os.getenv("PRODUCER_KAFKA_PASSWORD"):
        PRODUCER_CONFIG["PARAMS"]["security.protocol"] = "SASL_SSL"
        PRODUCER_CONFIG["PARAMS"]["sasl.mechanism"] = "SCRAM-SHA-512"
        PRODUCER_CONFIG["PARAMS"]["sasl.username"] = os.getenv("PRODUCER_KAFKA_USERNAME")
        PRODUCER_CONFIG["PARAMS"]["sasl.password"] = os.getenv("PRODUCER_KAFKA_PASSWORD")
    if os.getenv("SCRIBE_KAFKA_USERNAME") and os.getenv("SCRIBE_KAFKA_PASSWORD"):
        SCRIBE_PRODUCER_CONFIG["PARAMS"]["security.protocol"] = "SASL_SSL"
        SCRIBE_PRODUCER_CONFIG["PARAMS"]["sasl.mechanism"] = "SCRAM-SHA-512"
        SCRIBE_PRODUCER_CONFIG["PARAMS"]["sasl.username"] = os.getenv("SCRIBE_KAFKA_USERNAME")
        SCRIBE_PRODUCER_CONFIG["PARAMS"]["sasl.password"] = os.getenv("SCRIBE_KAFKA_PASSWORD")

    return {
        "CONSUMER_CONFIG": CONSUMER_CONFIG,
        "PRODUCER_CONFIG": PRODUCER_CONFIG,
        "SCRIBE_PRODUCER_CONFIG": SCRIBE_PRODUCER_CONFIG,
        "METRICS_CONFIG": METRICS_CONFIG,
        "PSQL_CONFIG": PSQL_CONFIG,
        "MODEL_CONFIG": model_config_factory(),
        "FEATURE_FLAGS": {
            "PROMETHEUS": bool(os.getenv("USE_PROMETHEUS", False)),
        },
        "LOGGING_DEBUG": bool(os.getenv("LOGGING_DEBUG", False)),
    }
