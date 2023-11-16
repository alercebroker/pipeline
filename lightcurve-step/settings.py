import os
from fastavro import schema
from fastavro.repository.base import SchemaRepositoryError
from credentials import get_credentials

##################################################
#       lightcurve_step   Settings File
##################################################


def settings_creator():
    # Set the global logging level to debug
    logging_debug = bool(os.getenv("LOGGING_DEBUG"))

    # db_config_mongo = get_credentials(
    #     os.environ["MONGODB_SECRET_NAME"], db_name="mongo"
    # )
    db_config_mongo = {
        "username": "root",
        "password": "root",
        "host": "mongo",
        "port": 27017,
        "database": "alerts",
        "authSource": "admin",
    }
    if os.getenv("SQL_SECRET_NAME"):
        db_config_sql = get_credentials(
            os.environ.get("SQL_SECRET_NAME"), db_name="sql"
        )
    else:
        db_config_sql = {
            "USER": "postgres",
            "PASSWORD": "postgres",
            "HOST": "postgres",
            "PORT": 5432,
            "DB_NAME": "postgres",
        }

    # Consumer configuration
    # Each consumer has different parameters and can be found in the documentation
    consumer_config = {
        "CLASS": "apf.consumers.KafkaConsumer",
        "PARAMS": {
            "bootstrap.servers": os.environ["CONSUMER_SERVER"],
            "group.id": os.environ["CONSUMER_GROUP_ID"],
            "auto.offset.reset": "beginning",
            "enable.partition.eof": True,
        },
        "TOPICS": os.environ["CONSUMER_TOPICS"].split(","),
        "consume.messages": int(os.getenv("CONSUME_MESSAGES", 50)),
        "consume.timeout": int(os.getenv("CONSUME_TIMEOUT", 15)),
    }

    try:
        the_schema = schema.load_schema("schema.avsc")
    except SchemaRepositoryError:
        # in case it is running from the root of the repository
        the_schema = schema.load_schema("lightcurve-step/schema.avsc")

    producer_config = {
        "CLASS": "apf.producers.KafkaProducer",
        "PARAMS": {
            "bootstrap.servers": os.environ["PRODUCER_SERVER"],
            "message.max.bytes": int(os.getenv("PRODUCER_MESSAGE_MAX_BYTES", 6291456)),
        },
        "TOPIC": os.environ["PRODUCER_TOPIC"],
        "SCHEMA": the_schema,
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
        "PROMETHEUS": prometheus,
        "DB_CONFIG": db_config_mongo,
        "DB_CONFIG_SQL": db_config_sql,
        "LOGGING_DEBUG": logging_debug,
        "USE_PROFILING": use_profiling,
        "PYROSCOPE_SERVER": pyroscope_server,
        "COMMIT": False,
    }
    return step_config
