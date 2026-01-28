import os
import pathlib

# SCHEMA PATH RELATIVE TO THE SETTINGS FILE
producer_schema_path = pathlib.Path(pathlib.Path(__file__).parent.parent, "schemas/magstats_step", "output.avsc")
metrics_schema_path = pathlib.Path(pathlib.Path(__file__).parent.parent, "schemas/magstats_step", "metrics.json")
scribe_schema_path = pathlib.Path(pathlib.Path(__file__).parent.parent, "schemas/scribe_step", "scribe.avsc")


def settings_factory():
    # Set the global logging level to debug
    logging_debug = os.getenv("LOGGING_DEBUG", False)

    excluded_calculators = os.getenv("EXCLUDED_CALCULATORS", "").strip().split(",")
    # Consumer configuration
    # Each consumer has different parameters and can be found in the documentation
    consumer_config = {
        "CLASS": os.getenv("CONSUMER_CLASS", "apf.consumers.KafkaConsumer"),
        "PARAMS": {
            "bootstrap.servers": os.environ["CONSUMER_SERVER"],
            "group.id": os.environ["CONSUMER_GROUP_ID"],
            "auto.offset.reset": "beginning",
            "enable.partition.eof": bool(os.getenv("ENABLE_PARTITION_EOF")),
        },
        "TOPICS": os.environ["CONSUMER_TOPICS"].split(","),
        "consume.messages": int(os.getenv("CONSUME_MESSAGES", 50)),
        "consume.timeout": int(os.getenv("CONSUME_TIMEOUT", 0)),
    }

    scribe_producer_config = {
        "CLASS": os.getenv("SCRIBE_PRODUCER_CLASS", "apf.producers.KafkaProducer"),
        "PARAMS": {
            "bootstrap.servers": os.environ["SCRIBE_PRODUCER_SERVER"],
        },
        "TOPIC": os.environ["SCRIBE_PRODUCER_TOPIC"],
        "SCHEMA_PATH": os.getenv("SCRIBE_SCHEMA_PATH", scribe_schema_path),
    }

    metrics_config = {
        "CLASS": "apf.metrics.KafkaMetricsProducer",
        "EXTRA_METRICS": [{"key": "aid"}, {"key": "candid"}],
        "PARAMS": {
            "PARAMS": {
                "bootstrap.servers": os.getenv("METRICS_SERVER"),
            },
            "TOPIC": os.getenv("METRICS_TOPIC", "metrics"),
            "SCHEMA_PATH": os.getenv("METRICS_SCHEMA_PATH", metrics_schema_path),
        },
    }

    if os.getenv("CONSUMER_KAFKA_USERNAME") and os.getenv("CONSUMER_KAFKA_PASSWORD"):
        consumer_config["PARAMS"]["security.protocol"] = "SASL_SSL"
        consumer_config["PARAMS"]["sasl.mechanism"] = "SCRAM-SHA-512"
        consumer_config["PARAMS"]["sasl.username"] = os.getenv("CONSUMER_KAFKA_USERNAME")
        consumer_config["PARAMS"]["sasl.password"] = os.getenv("CONSUMER_KAFKA_PASSWORD")
    if os.getenv("METRICS_KAFKA_USERNAME") and os.getenv("METRICS_KAFKA_PASSWORD"):
        metrics_config["PARAMS"]["PARAMS"]["security.protocol"] = "SASL_SSL"
        metrics_config["PARAMS"]["PARAMS"]["sasl.mechanism"] = "SCRAM-SHA-512"
        metrics_config["PARAMS"]["PARAMS"]["sasl.username"] = os.getenv("METRICS_KAFKA_USERNAME")
        metrics_config["PARAMS"]["PARAMS"]["sasl.password"] = os.getenv("METRICS_KAFKA_PASSWORD")
    if os.getenv("SCRIBE_KAFKA_USERNAME") and os.getenv("SCRIBE_KAFKA_PASSWORD"):
        scribe_producer_config["PARAMS"]["security.protocol"] = "SASL_SSL"
        scribe_producer_config["PARAMS"]["sasl.mechanism"] = "SCRAM-SHA-512"
        scribe_producer_config["PARAMS"]["sasl.username"] = os.getenv("SCRIBE_KAFKA_USERNAME")
        scribe_producer_config["PARAMS"]["sasl.password"] = os.getenv("SCRIBE_KAFKA_PASSWORD")
    # Step Configuration
    step_config = {
        "CONSUMER_CONFIG": consumer_config,
        "METRICS_CONFIG": metrics_config,
        "LOGGING_DEBUG": logging_debug,
        "SCRIBE_PRODUCER_CONFIG": scribe_producer_config,
        "EXCLUDED_CALCULATORS": excluded_calculators,
    }

    return step_config
