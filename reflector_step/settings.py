import os

##################################################
#       reflector_step   Settings File
##################################################

# Set the global logging level to debug
LOGGING_DEBUG = os.getenv("LOGGING_DEBUG", False)

# Consumer configuration
# Each consumer has different parameters and can be found in the documentation
CONSUMER_CONFIG = {
    "CLASS": "reflector_step.utils.RawKafkaConsumer",
    "PARAMS": {
        "bootstrap.servers": os.environ["CONSUMER_SERVER"],
        "group.id": os.environ["CONSUMER_GROUP_ID"],
        "auto.offset.reset": "beginning",
        "max.poll.interval.ms": 3600000,
    },
    "consume.timeout": int(os.getenv("CONSUME_TIMEOUT", 10)),
    "consume.messages": int(os.getenv("CONSUME_MESSAGES", 1000)),
}

USE_MESSAGE_TOPIC = not bool(os.getenv("PRODUCER_TOPIC"))

if os.getenv("TOPIC_STRATEGY_FORMAT"):
    CONSUMER_CONFIG["TOPIC_STRATEGY"] = {
        "CLASS": "apf.core.topic_management.DailyTopicStrategy",
        "PARAMS": {
            "topic_format": os.environ["TOPIC_STRATEGY_FORMAT"]
            .strip()
            .split(","),
            "date_format": "%Y%m%d",
            "change_hour": 23,
        },
    }
elif os.getenv("CONSUMER_TOPICS"):
    CONSUMER_CONFIG["TOPICS"] = (
        os.environ["CONSUMER_TOPICS"].strip().split(",")
    )
else:
    raise Exception("Add TOPIC_STRATEGY or CONSUMER_TOPICS")

PRODUCER_CONFIG = {
    "CLASS": "reflector_step.utils.RawKafkaProducer",
    "TOPIC": os.getenv("PRODUCER_TOPIC"),
    "PARAMS": {
        "bootstrap.servers": os.environ["PRODUCER_SERVER"],
        "acks": "all",
        "security.protocol": "SASL_SSL",
        "sasl.mechanism": "SCRAM-SHA-512",
        "sasl.username": os.environ["PRODUCER_USERNAME"],
        "sasl.password": os.environ["PRODUCER_PASSWORD"],
    },
}

METRICS_CONFIG = {
    "CLASS": "apf.metrics.KafkaMetricsProducer",
    "EXTRA_METRICS": [],  # This must be kept empty
    "PARAMS": {
        "PARAMS": {
            "bootstrap.servers": os.environ["METRICS_HOST"],
            "security.protocol": "SASL_SSL",
            "sasl.mechanism": "SCRAM-SHA-512",
            "sasl.username": os.environ["METRICS_USERNAME"],
            "sasl.password": os.environ["METRICS_PASSWORD"],
        },
        "TOPIC": os.environ["METRICS_TOPIC"],
        "SCHEMA": {
            "$schema": "http://json-schema.org/draft-07/schema",
            "$id": "http://example.com/example.json",
            "type": "object",
            "title": "ALeRCE reflector metrics schema",
            "description": "Metrics for custom mirrormaker used in ALeRCE pipeline.",
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
        },
    },
}

# Step Configuration
STEP_CONFIG = {
    "CONSUMER_CONFIG": CONSUMER_CONFIG,
    "PRODUCER_CONFIG": PRODUCER_CONFIG,
    "METRICS_CONFIG": METRICS_CONFIG,
    "use_message_topic": USE_MESSAGE_TOPIC,
    "keep_original_timestamp": os.getenv("KEEP_ORIGINAL_TIMESTAMP", False),
}
