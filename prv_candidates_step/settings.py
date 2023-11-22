import os
from fastavro import schema
from fastavro.repository.base import SchemaRepositoryError

##################################################
#       prv_candidates_step   Settings File
##################################################


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

    try:
        the_schema = schema.load_schema("schema.avsc")
    except SchemaRepositoryError:
        the_schema = schema.load_schema("prv_candidates_step/schema.avsc")
    producer_config = {
        "CLASS": "apf.producers.KafkaProducer",
        "PARAMS": {
            "bootstrap.servers": os.environ["PRODUCER_SERVER"],
            "message.max.bytes": int(os.getenv("PRODUCER_MESSAGE_MAX_BYTES", 6291456)),
        },
        "TOPIC": os.environ["PRODUCER_TOPIC"],
        "SCHEMA": the_schema,
    }

    try:
        the_schema = schema.load_schema("scribe_schema.avsc")
    except SchemaRepositoryError:
        the_schema = schema.load_schema("prv_candidates_step/scribe_schema.avsc")
    scribe_producer_config = {
        "CLASS": os.getenv("SCRIBE_PRODUCER_CLASS", "apf.producers.KafkaProducer"),
        "PARAMS": {
            "bootstrap.servers": os.environ["SCRIBE_PRODUCER_SERVER"],
        },
        "TOPIC": os.environ["SCRIBE_PRODUCER_TOPIC"],
        "SCHEMA": the_schema,
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
