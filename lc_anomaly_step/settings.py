##################################################
#       Late Classifier Settings File
##################################################
import os
import pathlib
from models_settings import configurator


def model_config_factory():
    modelclass = os.getenv("MODEL_CLASS", "")
    config = configurator(modelclass)
    return config


def config():
    CONSUMER_CONFIG = {
        "CLASS": os.getenv("CONSUMER_CLASS", "apf.consumers.KafkaConsumer"),
        "TOPICS": os.environ["CONSUMER_TOPICS"].strip().split(","),
        "PARAMS": {
            "bootstrap.servers": os.environ["CONSUMER_SERVER"],
            "group.id": os.environ["CONSUMER_GROUP_ID"],
            "auto.offset.reset": "beginning",
            "enable.partition.eof": bool(
                os.getenv("ENABLE_PARTITION_EOF", None)
            ),
        },
        "consume.timeout": int(os.getenv("CONSUME_TIMEOUT", 10)),
        "consume.messages": int(os.getenv("CONSUME_MESSAGES", 1000)),
    }

    producer_schema_path = str(
        pathlib.Path(
            pathlib.Path(__file__).parent.parent,
            "schemas/lc_classification_step",
            "output_ztf.avsc",
        )
    )
    PRODUCER_CONFIG = {
        "TOPIC_STRATEGY": {
            "PARAMS": {
                "topic_format": os.environ["PRODUCER_TOPIC_FORMAT"],
                "date_format": os.environ["PRODUCER_DATE_FORMAT"],
                "change_hour": int(os.environ["PRODUCER_CHANGE_HOUR"]),
                "retention_days": int(os.environ["PRODUCER_RETENTION_DAYS"]),
            },
            "CLASS": os.getenv(
                "PRODUCER_TOPIC_STRATEGY_CLASS",
                "apf.core.topic_management.DailyTopicStrategy",
            ),
        },
        "PARAMS": {
            "bootstrap.servers": os.environ["PRODUCER_SERVER"],
        },
        "CLASS": os.getenv(
            "PRODUCER_CLASS", "apf.producers.kafka.KafkaProducer"
        ),
        "SCHEMA_PATH": os.getenv("PRODUCER_SCHEMA_PATH", producer_schema_path),
    }

    scribe_schema_path = str(
        pathlib.Path(
            pathlib.Path(__file__).parent.parent,
            "schemas/scribe_step",
            "scribe.avsc",
        )
    )
    SCRIBE_PRODUCER_CONFIG = {
        "CLASS": "apf.producers.KafkaProducer",
        "PARAMS": {
            "bootstrap.servers": os.environ["SCRIBE_SERVER"],
        },
        "TOPIC": os.environ["SCRIBE_TOPIC"],
        "SCHEMA_PATH": os.getenv("SCRIBE_SCHEMA_PATH", scribe_schema_path),
    }

    metrics_schema_path = str(
        pathlib.Path(
            pathlib.Path(__file__).parent.parent,
            "schemas/lc_classification_step",
            "metrics.json",
        )
    )
    METRICS_CONFIG = {
        "CLASS": "apf.metrics.KafkaMetricsProducer",
        "EXTRA_METRICS": [{"key": "aid", "alias": "aid"}, {"key": "candid"}],
        "PARAMS": {
            "PARAMS": {
                "bootstrap.servers": os.environ["METRICS_HOST"],
            },
            "TOPIC": os.environ["METRICS_TOPIC"],
            "SCHEMA_PATH": os.getenv(
                "METRICS_SCHEMA_PATH", metrics_schema_path
            ),
        },
    }

    if os.getenv("CONSUMER_KAFKA_USERNAME") and os.getenv(
        "CONSUMER_KAFKA_PASSWORD"
    ):
        CONSUMER_CONFIG["PARAMS"]["security.protocol"] = "SASL_SSL"
        CONSUMER_CONFIG["PARAMS"]["sasl.mechanism"] = "SCRAM-SHA-512"
        CONSUMER_CONFIG["PARAMS"]["sasl.username"] = os.getenv(
            "CONSUMER_KAFKA_USERNAME"
        )
        CONSUMER_CONFIG["PARAMS"]["sasl.password"] = os.getenv(
            "CONSUMER_KAFKA_PASSWORD"
        )
    if os.getenv("PRODUCER_KAFKA_USERNAME") and os.getenv(
        "PRODUCER_KAFKA_PASSWORD"
    ):
        PRODUCER_CONFIG["PARAMS"]["security.protocol"] = os.getenv(
            "PRODUCER_SECURITY_PROTOCOL", "SASL_PLAINTEXT"
        )
        PRODUCER_CONFIG["PARAMS"]["sasl.mechanism"] = os.getenv(
            "PRODUCER_SASL_MECHANISM", "SCRAM-SHA-256"
        )
        PRODUCER_CONFIG["PARAMS"]["sasl.username"] = os.getenv(
            "PRODUCER_KAFKA_USERNAME"
        )
        PRODUCER_CONFIG["PARAMS"]["sasl.password"] = os.getenv(
            "PRODUCER_KAFKA_PASSWORD"
        )
    if os.getenv("SCRIBE_KAFKA_USERNAME") and os.getenv(
        "SCRIBE_KAFKA_PASSWORD"
    ):
        SCRIBE_PRODUCER_CONFIG["PARAMS"]["security.protocol"] = "SASL_SSL"
        SCRIBE_PRODUCER_CONFIG["PARAMS"]["sasl.mechanism"] = "SCRAM-SHA-512"
        SCRIBE_PRODUCER_CONFIG["PARAMS"]["sasl.username"] = os.getenv(
            "SCRIBE_KAFKA_USERNAME"
        )
        SCRIBE_PRODUCER_CONFIG["PARAMS"]["sasl.password"] = os.getenv(
            "SCRIBE_KAFKA_PASSWORD"
        )
    if os.getenv("METRICS_KAFKA_USERNAME") and os.getenv(
        "METRICS_KAFKA_PASSWORD"
    ):
        METRICS_CONFIG["PARAMS"]["PARAMS"]["security.protocol"] = "SASL_SSL"
        METRICS_CONFIG["PARAMS"]["PARAMS"]["sasl.mechanism"] = "SCRAM-SHA-512"
        METRICS_CONFIG["PARAMS"]["PARAMS"]["sasl.username"] = os.getenv(
            "METRICS_KAFKA_USERNAME"
        )
        METRICS_CONFIG["PARAMS"]["PARAMS"]["sasl.password"] = os.getenv(
            "METRICS_KAFKA_PASSWORD"
        )

    return {
        "SCRIBE_PRODUCER_CONFIG": SCRIBE_PRODUCER_CONFIG,
        "CONSUMER_CONFIG": CONSUMER_CONFIG,
        "PRODUCER_CONFIG": PRODUCER_CONFIG,
        "METRICS_CONFIG": METRICS_CONFIG,
        "MODEL_VERSION": os.getenv("MODEL_VERSION", "dev"),
        "MODEL_CONFIG": model_config_factory(),
        "SCRIBE_PARSER_CLASS": os.getenv("SCRIBE_PARSER_CLASS"),
        "STEP_PARSER_CLASS": os.getenv("STEP_PARSER_CLASS"),
        "FEATURE_FLAGS": {
            "PROMETHEUS": bool(os.getenv("USE_PROMETHEUS", True)),
            "USE_PROFILING": bool(os.getenv("USE_PROFILING", False)),
            "LOG_CLASS_DISTRIBUTION": bool(
                os.getenv("LOG_CLASS_DISTRIBUTION", False)
            ),
        },
        "MIN_DETECTIONS": os.getenv("MIN_DETECTIONS", None),
    }
