##################################################
#       consolidated_metrics_step   Settings File
##################################################
import os

# Set the global logging level to debug
LOGGING_DEBUG = os.getenv("LOGGING_DEBUG", False)

# Consumer configuration
# Each consumer has different parameters and can be found in the documentation
CONSUMER_CONFIG = {
    "PARAMS": {
        "bootstrap.servers": os.environ["CONSUMER_SERVER"],
        "group.id": os.environ["CONSUMER_GROUP_ID"],
        "auto.offset.reset": "beginning",
        "max.poll.interval.ms": 3600000,
    },
    "consume.timeout": int(os.getenv("CONSUME_TIMEOUT", 10)),
    "consume.messages": int(os.getenv("CONSUME_MESSAGES", 100)),
    "TOPICS": os.environ["CONSUMER_TOPICS"].strip().split(","),
}

DB_CONFIG = {}

PIPELINE_ORDER = {
    "EarlyClassifier": None,
    "S3Step": None,
    "WatchlistStep": None,
    "SortingHatStep": {
        "IngestionStep": {"XmatchStep": {"FeaturesComputer": {"LateClassifier": None}}}
    },
}

# Step Configuration
STEP_CONFIG = {
    # "N_PROCESS": 4,            # Number of process for multiprocess script
    "COMMIT": bool(
        os.getenv("COMMIT", True)
    ),  # Disables commit, useful to debug a KafkaConsumer
    "DB_CONFIG": DB_CONFIG,
    "CONSUMER_CONFIG": CONSUMER_CONFIG,
}
