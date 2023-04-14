import os
from fastavro import schema

from credentials import get_mongodb_credentials

##################################################
#       lightcurve_step   Settings File
##################################################


def settings_creator():
    # Set the global logging level to debug
    logging_debug = False

    db_config = get_mongodb_credentials(os.environ["MONGODB_SECRET_NAME"])

    # Consumer configuration
    # Each consumer has different parameters and can be found in the documentation
    consumer_config = {
        "CLASS": "apf.consumers.KafkaConsumer",
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
        "consume.timeout": int(os.getenv("CONSUME_TIMEOUT", 10)),
    }

    producer_config = {
        "CLASS": "apf.producers.KafkaProducer",
        "PARAMS": {
            "bootstrap.servers": os.environ["PRODUCER_SERVER"],
        },
        "TOPIC": os.environ["PRODUCER_TOPIC"],
        "SCHEMA": schema.load_schema("schema.avsc"),
    }

    metrics_config = {
        "CLASS": "apf.metrics.KafkaMetricsProducer",
        "EXTRA_METRICS": [
            {"key": "aid", "format": lambda x: str(x)},
        ],
        "PARAMS": {
            "PARAMS": {
                "bootstrap.servers": os.environ["METRICS_SERVER"],
                "auto.offset.reset": "smallest",
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
        consumer_config["PARAMS"]["security.protocol"] = "SASL_SSL"
        consumer_config["PARAMS"]["sasl.mechanism"] = "SCRAM-SHA-512"
        consumer_config["PARAMS"]["sasl.username"] = os.getenv(
            "CONSUMER_KAFKA_USERNAME"
        )
        consumer_config["PARAMS"]["sasl.password"] = os.getenv(
            "CONSUMER_KAFKA_PASSWORD"
        )
    if os.getenv("PRODUCER_KAFKA_USERNAME") and os.getenv("PRODUCER_KAFKA_PASSWORD"):
        producer_config["PARAMS"]["security.protocol"] = os.getenv(
            "PRODUCER_SECURITY_PROTOCOL", "SASL_PLAINTEXT"
        )
        producer_config["PARAMS"]["sasl.mechanism"] = os.getenv(
            "PRODUCER_SASL_MECHANISM", "SCRAM-SHA-256"
        )
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

    # Step Configuration
    step_config = {
        "consumer_config": consumer_config,
        "producer_config": producer_config,
        "metrics_config": metrics_config,
        "PROMETHEUS": prometheus,
        "DB_CONFIG": db_config,
        "LOGGING_DEBUG": logging_debug,
    }
    return step_config
