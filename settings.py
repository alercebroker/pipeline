##################################################
#       features   Settings File
##################################################
import os

FEATURE_VERSION = "v0.1"
STEP_VERSION = "v0.1"

DB_CONFIG = {
    "SQL": {
        "ENGINE": os.environ["DB_ENGINE"],
        "HOST": os.environ["DB_HOST"],
        "USER": os.environ["DB_USER"],
        "PASSWORD": os.environ["DB_PASSWORD"],
        "PORT": int(os.environ["DB_PORT"]),
        "DB_NAME": os.environ["DB_NAME"],
    }
}

CONSUMER_CONFIG = {
    "TOPICS": os.environ["CONSUMER_TOPICS"].strip().split(","),
    "PARAMS": {
        "bootstrap.servers": os.environ["CONSUMER_SERVER"],
        "group.id": os.environ["CONSUMER_GROUP_ID"],
    },
}

PRODUCER_CONFIG = {
    "TOPIC": os.environ["PRODUCER_TOPIC"],
    "PARAMS": {
        "bootstrap.servers": os.environ["PRODUCER_SERVER"],
    },
    "SCHEMA": {
        "doc": "Features",
        "name": "features",
        "type": "record",
        "fields": [
            {"name": "oid", "type": "string"},
            {
                "name": "features",
                "type": {"type": "map", "values": ["float", "int", "string", "null"]},
            },
        ],
    },
}

# ES_CONFIG = {
#    "INDEX_PREFIX": os.environ["ES_PREFIX"],
#    "host": os.environ["ES_NETWORK_HOST"],
#    "port": os.environ["ES_NETWORK_PORT"]
# }

STEP_CONFIG = {
    "CONSUMER_CONFIG": CONSUMER_CONFIG,
    "DB_CONFIG": DB_CONFIG,
    # "ES_CONFIG": ES_CONFIG,
    # "PRODUCER_CONFIG": PRODUCER_CONFIG,
    "FEATURE_VERSION": FEATURE_VERSION,
    "STEP_VERSION": STEP_VERSION,
    "STEP_VERSION_PREPROCESS": "v1",
}
