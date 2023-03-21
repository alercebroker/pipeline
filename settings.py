import os


def settings_factory():
    # Set the global logging level to debug
    logging_debug = os.getenv("LOGGING_DEBUG", False)

    excluded_calculators = os.getenv("INCLUDED_CALCULATORS", "").strip().split(",")
    # Consumer configuration
    # Each consumer has different parameters and can be found in the documentation
    consumer_config = {
        "PARAMS": {
            "bootstrap.servers": os.environ["CONSUMER_SERVER"],
            "group.id": os.environ["CONSUMER_GROUP_ID"],
            "auto.offset.reset": "beginning",
            "max.poll.interval.ms": 3600000,
        },
        "consume.timeout": int(os.getenv("CONSUME_TIMEOUT", 10)),
        "consume.messages": int(os.getenv("CONSUME_MESSAGES", 10)),
    }

    if os.getenv("TOPIC_STRATEGY_FORMAT"):
        consumer_config["TOPIC_STRATEGY"] = {
            "CLASS": "apf.core.topic_management.DailyTopicStrategy",
            "PARAMS": {
                "topic_format": os.environ["TOPIC_STRATEGY_FORMAT"].strip().split(","),
                "date_format": "%Y%m%d",
                "change_hour": 23,
            },
        }
    elif os.getenv("CONSUMER_TOPICS"):
        consumer_config["TOPICS"] = os.environ["CONSUMER_TOPICS"].strip().split(",")
    else:
        raise Exception("Add TOPIC_STRATEGY or CONSUMER_TOPICS")

    metrics_config = {
        "CLASS": "apf.metrics.KafkaMetricsProducer",
        "EXTRA_METRICS": [
            {"key": "oid", "alias": "oid"},
        ],
        "PARAMS": {
            "PARAMS": {
                "bootstrap.servers": os.environ["METRICS_HOST"],
                "auto.offset.reset": "smallest",
            },
            "TOPIC": os.environ["METRICS_TOPIC"],
            "SCHEMA": {
                "$schema": "http://json-schema.org/draft-07/schema",
                "$id": "http://example.com/example.json",
                "type": "object",
                "title": "The root schema",
                "description": "The root schema comprises the entire JSON document.",
                "default": {},
                "examples": [
                    {
                        "timestamp_sent": "2020-09-01",
                        "timestamp_received": "2020-09-01",
                    }
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

    # Step Configuration
    step_config = {
        "CONSUMER_CONFIG": consumer_config,
        "METRICS_CONFIG": metrics_config,
        "LOGGING_DEBUG": logging_debug,
        "EXCLUDED_CALCULATORS": filter(bool, excluded_calculators)
    }

    return step_config
