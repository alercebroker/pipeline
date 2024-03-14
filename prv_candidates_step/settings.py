import pathlib
import os

##################################################
#       prv_candidates_step   Settings File
##################################################

# SCHEMA PATH RELATIVE TO THE SETTINGS FILE
PRODUCER_SCHEMA_PATH = pathlib.Path(
    pathlib.Path(__file__).parent.parent, "schemas/prv_candidate_step", "output.avsc"
)
METRICS_SCHEMA_PATH = pathlib.Path(
    pathlib.Path(__file__).parent.parent, "schemas/prv_candidate_step", "metrics.json"
)
SCRIBE_SCHEMA_PATH = pathlib.Path(
    pathlib.Path(__file__).parent.parent, "schemas/scribe_step", "scribe.avsc"
)


def settings_creator():
    # Set the global logging level to debug
    logging_debug = False
    prometheus = os.getenv("USE_PROMETHEUS", False)

    # Consumer configuration
    # Each consumer has different parameters and can be found in the documentation
    consumer_config = {
        "CLASS": "apf.consumers.KafkaConsumer",
        "PARAMS": {
            "bootstrap.servers": os.environ["CONSUMER_SERVER"],
            "group.id": os.environ["CONSUMER_GROUP_ID"],
            "auto.offset.reset": "beginning",
            "enable.partition.eof": bool(os.getenv("ENABLE_PARTITION_EOF", False)),
        },
        "TOPICS": os.environ["CONSUMER_TOPICS"].split(","),
        "consume.messages": int(os.getenv("CONSUME_MESSAGES", "1")),
        "consume.timeout": int(os.getenv("CONSUME_TIMEOUT", "10")),
    }

    producer_config = {
        "CLASS": os.getenv("PRODUCER_CLASS", "apf.producers.KafkaProducer"),
        "PARAMS": {
            "bootstrap.servers": os.environ["PRODUCER_SERVER"],
            "message.max.bytes": int(os.getenv("PRODUCER_MESSAGE_MAX_BYTES", 6291456)),
        },
        "TOPIC": os.environ["PRODUCER_TOPIC"],
        "SCHEMA_PATH": os.getenv("PRODUCER_SCHEMA_PATH ", PRODUCER_SCHEMA_PATH),
    }

    scribe_producer_config = {
        "CLASS": os.getenv("SCRIBE_PRODUCER_CLASS", "apf.producers.KafkaProducer"),
        "PARAMS": {
            "bootstrap.servers": os.environ["SCRIBE_PRODUCER_SERVER"],
        },
        "TOPIC": os.environ["SCRIBE_PRODUCER_TOPIC"],
        "SCHEMA_PATH": os.getenv("SCRIBE_SCHEMA_PATH", SCRIBE_SCHEMA_PATH),
    }

    metrics_config = {
        "CLASS": os.getenv("METRICS_CLASS", "apf.metrics.KafkaMetricsProducer"),
        "EXTRA_METRICS": [
            {"key": "candid", "format": lambda x: str(x)},
            {"key": "oid"},
            {"key": "aid"},
        ],
        "PARAMS": {
            "PARAMS": {
                "bootstrap.servers": os.getenv("METRICS_HOST"),
                "auto.offset.reset": "smallest",
            },
            "TOPIC": os.getenv("METRICS_TOPIC", "metrics"),
            "SCHEMA_PATH": os.getenv("METRICS_SCHEMA_PATH", METRICS_SCHEMA_PATH),
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
    if os.getenv("SCRIBE_KAFKA_USERNAME") and os.getenv("SCRIBE_KAFKA_PASSWORD"):
        scribe_producer_config["PARAMS"]["security.protocol"] = "SASL_SSL"
        scribe_producer_config["PARAMS"]["sasl.mechanism"] = "SCRAM-SHA-512"
        scribe_producer_config["PARAMS"]["sasl.username"] = os.getenv(
            "SCRIBE_KAFKA_USERNAME"
        )
        scribe_producer_config["PARAMS"]["sasl.password"] = os.getenv(
            "SCRIBE_KAFKA_PASSWORD"
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

    # Step Configuration
    return {
        "CONSUMER_CONFIG": consumer_config,
        "METRICS_CONFIG": metrics_config,
        "PRODUCER_CONFIG": producer_config,
        "SCRIBE_PRODUCER_CONFIG": scribe_producer_config,
        "LOGGING_DEBUG": logging_debug,
        "USE_PROMETHEUS": prometheus,
        "USE_PROFILING": os.getenv("USE_PROFILING", False),
        "PYROSCOPE_SERVER": os.getenv("PYROSCOPE_SERVER"),
    }
