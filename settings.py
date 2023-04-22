##################################################
#       alert_archiving_step   Settings File
##################################################
import os

# Set the global logging level to debug
LOGGING_DEBUG = os.getenv("LOGGING_DEBUG", False)

## Consumer configuration
### Each consumer has different parameters and can be found in the documentation
CONSUMER_CONFIG = {
    "PARAMS": {
        "bootstrap.servers": os.environ["CONSUMER_SERVER"],
        "group.id": os.environ["CONSUMER_GROUP_ID"],
        "auto.offset.reset": "beginning",
        "max.poll.interval.ms": 3600000,
    },
    "consume.timeout": int(os.getenv("CONSUME_TIMEOUT", 60)),
    "consume.messages": int(os.getenv("CONSUME_MESSAGES", 2200)),
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

STEP_METADATA = {
    "STEP_VERSION": os.getenv("STEP_VERSION", "dev"),
    "STEP_ID": os.getenv("STEP_ID", "archiving"),
    "STEP_NAME": os.getenv("STEP_NAME", "archiving"),
    "STEP_COMMENTS": os.getenv("STEP_COMMENTS", ""),
}

if os.getenv("KAFKA_USERNAME") and os.getenv("KAFKA_PASSWORD"):
    CONSUMER_CONFIG["PARAMS"]["security.protocol"] = "SASL_SSL"
    CONSUMER_CONFIG["PARAMS"]["sasl.mechanism"] = "SCRAM-SHA-512"
    CONSUMER_CONFIG["PARAMS"]["sasl.username"] = os.getenv("KAFKA_USERNAME")
    CONSUMER_CONFIG["PARAMS"]["sasl.password"] = os.getenv("KAFKA_PASSWORD")

## Step Configuration
STEP_CONFIG = {
    "N_PROCESS": os.getenv("N_PROCESS"),  # Number of process for multiprocess script
    "CONSUMER_CONFIG": CONSUMER_CONFIG,
    "STEP_METADATA": STEP_METADATA,
    "FORMAT": os.getenv("ARCHIVE_FORMAT", "avro"),
    "ZTF_BUCKET_NAME": os.getenv("S3_ZTF_BUCKET_NAME", "ztf-avro"),
    "ATLAS_BUCKET_NAME": os.getenv("S3_ATLAS_BUCKET_NAME", "astro-alerts-archive"),
}
