##################################################
#       sorting_hat_step   Settings File
##################################################
import os
from schema import SCHEMA

# Set the global logging level to debug
LOGGING_DEBUG = os.getenv("LOGGING_DEBUG", False)

DB_CONFIG = {
    "HOST": os.environ["DB_HOST"],
    "USER": os.getenv("DB_USER", None),
    "PASSWORD": os.getenv("DB_PASSWORD", None),
    "PORT":  int(os.environ["DB_PORT"]),
    "DATABASE": os.environ["DB_NAME"],
}

# Consumer configuration
# Each consumer has different parameters and can be found in the documentation
CONSUMER_CONFIG = {
    "PARAMS": {
        "bootstrap.servers":  os.environ["CONSUMER_SERVER"],
        "group.id": os.environ["CONSUMER_GROUP_ID"],
        "auto.offset.reset": "beginning",
        'max.poll.interval.ms': 3600000
    },
    "consume.timeout": int(os.getenv("CONSUME_TIMEOUT", 10)),
    "consume.messages": int(os.getenv("CONSUME_MESSAGES", 10)),
}

if os.getenv("TOPIC_STRATEGY_FORMAT"):
    CONSUMER_CONFIG["TOPIC_STRATEGY"] = {
        "CLASS": "apf.core.topic_management.DailyTopicStrategy",
        "PARAMS": {
            "topic_format": os.environ["TOPIC_STRATEGY_FORMAT"].strip().split(","),
            "date_format": "%Y%m%d",
            "change_hour": 23,
        },
    }
elif os.getenv("CONSUMER_TOPICS"):
    CONSUMER_CONFIG["TOPICS"] = os.environ["CONSUMER_TOPICS"].strip().split(",")
else:
    raise Exception("Add TOPIC_STRATEGY or CONSUMER_TOPICS")

# Producer configuration
PRODUCER_CONFIG = {
    "TOPIC": os.environ["PRODUCER_TOPIC"],
    "PARAMS": {
        "bootstrap.servers": os.environ["PRODUCER_SERVER"],
    },
    "SCHEMA": SCHEMA
}

STEP_METADATA = {
    "STEP_VERSION": os.getenv("STEP_VERSION", "dev"),
    "STEP_ID": os.getenv("STEP_ID", "preprocess"),
    "STEP_NAME": os.getenv("STEP_NAME", "preprocess"),
    "STEP_COMMENTS": os.getenv("STEP_COMMENTS", ""),
}

# Step Configuration
STEP_CONFIG = {
    "DB_CONFIG": DB_CONFIG,
    "CONSUMER_CONFIG": CONSUMER_CONFIG,
    "PRODUCER_CONFIG": PRODUCER_CONFIG,
    "N_PROCESS": os.getenv("N_PROCESS"),
    "STEP_METADATA": STEP_METADATA,
}
